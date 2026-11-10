import asyncio
import logging
import os
import re
import queue as frame_queue
import threading
import time
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urlsplit, quote

import aiohttp
import discord
import imageio_ffmpeg
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
TOKEN = os.getenv('DISCORD_TOKEN', '').strip()
GUILD_ID = os.getenv('DISCORD_GUILD_ID', '').strip()
FFMPEG = shutil.which('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe()
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='!', intents=intents)

@dataclass
class Player:
    queue: deque = field(default_factory=deque)
    current: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    generation: int = 0

players: dict[int, Player] = {}

def parse_track(url: str) -> tuple[str, str]:
    parts = urlsplit(url.strip())
    if parts.scheme != 'https' or parts.netloc.lower() not in ('grime.fm', 'www.grime.fm'):
        raise ValueError('Нужна ссылка https://grime.fm/track/…')
    match = re.fullmatch(r'/track/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)/?', parts.path)
    if not match:
        raise ValueError('Нужна ссылка на отдельный трек: https://grime.fm/track/артист/трек')
    artist, slug = match.groups()
    return f'https://grime.fm/track/{artist}/{slug}', f'https://grime.fm/api/tracks/{quote(slug, safe="")}/audio'

async def check_track(page: str, audio: str):
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(page, allow_redirects=False) as response:
            if response.status != 200:
                raise ValueError('Страница трека недоступна. Проверь ссылку.')
        # Only check response headers; do not download an entire track.
        async with session.get(audio, headers={'Range': 'bytes=0-0'}, allow_redirects=True) as response:
            if response.status not in (200, 206):
                raise ValueError(f'Аудио недоступно (HTTP {response.status}).')
            if not response.headers.get('Content-Type', '').lower().startswith(('audio/', 'application/octet-stream')):
                raise ValueError('Сайт не вернул аудиофайл. Возможно, адрес плеера изменился.')

async def download_track(audio: str) -> str:
    # The server's connection to grime.fm stalls on large responses. Fetch
    # small byte ranges instead, with bounded concurrency and retries.
    chunk_size = 16384
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        resolved_url = audio
        async def fetch(start: int, end: int):
            nonlocal resolved_url
            for attempt in range(3):
                try:
                    async with session.get(resolved_url, headers={'Range': f'bytes={start}-{end}'}) as response:
                        if response.status != 206:
                            raise ValueError(f'Сайт не поддерживает загрузку частями (HTTP {response.status}).')
                        resolved_url = str(response.url)
                        data = await response.read()
                        if len(data) != end - start + 1:
                            raise ValueError('Сайт вернул неполный фрагмент аудио.')
                        return data, response.headers.get('Content-Range', '')
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if attempt == 2: raise
        first, content_range = await fetch(0, chunk_size - 1)
        match = re.fullmatch(r'bytes 0-16383/(\d+)', content_range)
        if not match:
            raise ValueError('Не удалось определить размер аудиофайла.')
        size = int(match.group(1))
        if size > 50 * 1024 * 1024:
            raise ValueError('Трек слишком большой (лимит 50 МБ).')
        fd, path = tempfile.mkstemp(prefix='grime-', suffix='.audio')
        try:
            with os.fdopen(fd, 'wb') as output:
                output.write(first)
                starts = range(chunk_size, size, chunk_size)
                batch = []
                for start in starts:
                    batch.append((start, min(start + chunk_size, size) - 1))
                    if len(batch) == 32:
                        for data, _ in await asyncio.gather(*(fetch(a, b) for a, b in batch)):
                            output.write(data)
                        batch.clear()
                if batch:
                    for data, _ in await asyncio.gather(*(fetch(a, b) for a, b in batch)):
                        output.write(data)
            return path
        except BaseException:
            os.unlink(path)
            raise

def remove_file(path: str):
    try:
        os.remove(path)
    except OSError:
        pass

class BufferedOpus(discord.AudioSource):
    """Read encoded frames ahead so brief decoding stalls do not interrupt playback."""
    def __init__(self, source):
        self.source = source
        self.frames = frame_queue.Queue(maxsize=250)
        self.stopped = threading.Event()
        self.finished = threading.Event()
        self.started = False
        self.worker = threading.Thread(target=self._fill, daemon=True)
        self.worker.start()

    def _fill(self):
        try:
            while not self.stopped.is_set():
                frame = self.source.read()
                while not self.stopped.is_set():
                    try:
                        self.frames.put(frame, timeout=0.2)
                        break
                    except frame_queue.Full:
                        continue
                if not frame:
                    break
        except Exception:
            logging.exception('Ошибка подготовки аудиобуфера')
        finally:
            self.finished.set()

    def read(self):
        if not self.started:
            self.started = True
            deadline = time.monotonic() + 3
            while self.frames.qsize() < 50 and not self.finished.is_set() and time.monotonic() < deadline:
                time.sleep(0.01)
        if self.frames.empty() and not self.finished.is_set():
            logging.warning('Аудиобуфер опустел: ожидание декодера')
        while not self.stopped.is_set():
            try:
                return self.frames.get(timeout=0.25)
            except frame_queue.Empty:
                if self.finished.is_set():
                    return b''
        return b''

    def is_opus(self):
        return True

    def cleanup(self):
        self.stopped.set()
        self.source.cleanup()

async def play_next(guild: discord.Guild, player: Player):
    async with player.lock:
        voice = guild.voice_client
        if not voice or not voice.is_connected() or voice.is_playing() or voice.is_paused():
            return
        player.current = None
        while player.queue:
            page, audio = player.queue.popleft()
            try:
                source = BufferedOpus(discord.FFmpegOpusAudio(audio, executable=FFMPEG, before_options='-nostdin', options='-vn'))
                generation = player.generation
                def done(error, generation=generation, path=audio):
                    remove_file(path)
                    if error:
                        logging.error('Ошибка воспроизведения: %s', error)
                    if player.generation == generation and not bot.loop.is_closed():
                        asyncio.run_coroutine_threadsafe(play_next(guild, player), bot.loop)
                voice.play(source, after=done)
                player.current = page
                return
            except Exception:
                remove_file(audio)
                logging.exception('Не удалось запустить %s', page)

def same_channel(interaction: discord.Interaction):
    member = interaction.user
    channel = member.voice.channel if isinstance(member, discord.Member) and member.voice else None
    if channel is None:
        return None, 'Сначала зайди в голосовой канал.'
    voice = interaction.guild.voice_client
    if voice and voice.channel != channel:
        return None, 'Бот уже в другом голосовом канале. Зайди в тот же канал.'
    return channel, None

@bot.event
async def setup_hook():
    if GUILD_ID:
        guild = discord.Object(id=int(GUILD_ID))
        bot.tree.copy_global_to(guild=guild)
        await bot.tree.sync(guild=guild)
        logging.info('Команды зарегистрированы на сервере %s', GUILD_ID)
    else:
        await bot.tree.sync()
        logging.info('Глобальные команды зарегистрированы')

@bot.event
async def on_ready():
    logging.info('Бот вошёл как %s', bot.user)

@bot.tree.command(name='play', description='Включить или добавить в очередь трек grime.fm')
@app_commands.describe(url='Необязательно: ссылка на трек или поисковый запрос')
@app_commands.guild_only()
async def play(interaction: discord.Interaction, url: str | None = None):
    if not url or not url.strip().startswith(('https://', 'http://')):
        return await show_catalog(interaction, (url or '').strip())
    channel, error = same_channel(interaction)
    if error:
        return await interaction.response.send_message(error, ephemeral=True)
    try:
        page, audio = parse_track(url)
    except ValueError as exc:
        return await interaction.response.send_message(str(exc), ephemeral=True)
    await interaction.response.defer(thinking=True)
    audio_path = None
    player = players.setdefault(interaction.guild_id, Player())
    try:
        async with player.lock:
            voice = interaction.guild.voice_client
            if voice and voice.channel != channel:
                return await interaction.edit_original_response(content='Бот уже в другом голосовом канале.')
            if not voice:
                await channel.connect(self_deaf=True)
            generation = player.generation
        await interaction.edit_original_response(content=f'Подключился к голосовому каналу. Загружаю трек: {page}')
        await check_track(page, audio)
        audio_path = await download_track(audio)
        async with player.lock:
            voice = interaction.guild.voice_client
            if generation != player.generation or not voice or not voice.is_connected():
                remove_file(audio_path)
                audio_path = None
                return await interaction.followup.send('Загрузка отменена: воспроизведение остановлено.')
            player.queue.append((page, audio_path))
            audio_path = None
        await play_next(interaction.guild, player)
        await interaction.followup.send(f'Трек готов: {page}')
    except (ValueError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
        await interaction.followup.send(f'Не удалось получить трек: {exc}')
    except (discord.ClientException, discord.DiscordException, OSError) as exc:
        logging.exception('Ошибка подключения или воспроизведения')
        await interaction.followup.send(f'Не удалось подключиться или запустить плеер: {exc}')
    finally:
        if audio_path:
            remove_file(audio_path)

class TrackResults(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self.title_parts = None
        self.artist_parts = None
        self.artist_depth = 0
        self.title_depth = 0
        self.current_url = None
        self.result_open = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get('class', '').split()
        if tag == 'a' and 'gh-trow-title' in classes:
            if self.result_open:
                self.finish_result()
            self.current_url = 'https://grime.fm' + attrs.get('href', '')
            self.title_parts = []
            self.artist_parts = []
            self.title_depth = 1
            self.result_open = True
        elif self.title_depth:
            self.title_depth += 1
        elif self.result_open and tag == 'span' and 'gh-trow-artist' in classes:
            self.artist_depth = 1
        elif self.artist_depth:
            self.artist_depth += 1

    def handle_endtag(self, tag):
        if self.title_depth:
            self.title_depth -= 1
        elif self.artist_depth:
            self.artist_depth -= 1

    def handle_data(self, data):
        if self.title_depth and self.title_parts is not None:
            self.title_parts.append(data)
        elif self.artist_depth and self.artist_parts is not None:
            self.artist_parts.append(data)

    def finish_result(self):
        title = ' '.join(''.join(self.title_parts or []).split())
        artist = ' '.join(''.join(self.artist_parts or []).split())
        if title and self.current_url and self.current_url.startswith('https://grime.fm/track/'):
            self.results.append((title, artist, self.current_url))
        self.result_open = False

async def find_tracks(query: str, filters=None, page=1, per=25):
    timeout = aiohttp.ClientTimeout(total=30)
    params = {'q': query, 'per': str(per), 'page': str(page)}
    params.update({k: v for k, v in (filters or {}).items() if v})
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get('https://grime.fm/tracks', params=params) as response:
            if response.status != 200:
                raise ValueError(f'Каталог недоступен (HTTP {response.status}).')
            html = await response.text()
    parser = TrackResults()
    parser.feed(html)
    if parser.result_open:
        parser.finish_result()
    return parser.results

class CatalogState:
    def __init__(self, owner, query=''):
        self.owner = owner
        self.query = query
        self.filters = {'type': '', 'context': '', 'year': '', 'sort': 'hot', 'lyrics': ''}
        self.per = 25
        self.window = 0
        self.cache = {}

    async def get_tracks(self):
        offset = self.window * 5
        site_page = offset // self.per + 1
        within = offset % self.per
        key = (self.query, tuple(self.filters.items()), self.per, site_page)
        if key not in self.cache:
            self.cache[key] = await find_tracks(self.query, self.filters, site_page, self.per)
        page = self.cache[key]
        return page[within:within + 5], len(page) > within + 5 or len(page) == self.per

class CatalogSelect(discord.ui.Select):
    def __init__(self, state, kind, tracks=None):
        self.state = state
        self.kind = kind
        f = state.filters
        if kind == 'category':
            choices = [('Тип: все', 'type:'), ('Тип: треки', 'type:track'), ('Тип: биты', 'type:instrumental'),
                       ('Раздел: все', 'context:'), ('Раздел: баттлы', 'context:battles'),
                       ('Раздел: релизы', 'context:releases'), ('Раздел: альбомы', 'context:albums')]
            options = [discord.SelectOption(label=a, value=b) for a,b in choices]
            placeholder = 'Тип и раздел'
        elif kind == 'year':
            years = [''] + [str(y) for y in range(2026, 2005, -1)]
            options = [discord.SelectOption(label=y or 'Все годы', value=y or 'all', default=f['year'] == y) for y in years]
            placeholder = 'Год'
        elif kind == 'sort':
            choices = [('Актуальные', 'hot'), ('За 30 дней', 'listeners'), ('Популярные', 'plays'), ('По лайкам', 'likes'), ('Новые', 'new'), ('Старые', 'old'), ('Название', 'title')]
            options = [discord.SelectOption(label=a,value=b,default=f['sort']==b) for a,b in choices]
            placeholder = 'Сортировка'
        else:
            self.tracks = tracks
            options = [discord.SelectOption(label=title[:100], description=(artist or 'Трек grime.fm')[:100], value=str(i)) for i,(title,artist,_) in enumerate(tracks)]
            placeholder = '▶ Выбери трек для воспроизведения'
        super().__init__(placeholder=placeholder, options=options)

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.state.owner:
            return await interaction.response.send_message('Открой свой каталог командой /play.', ephemeral=True)
        if self.kind == 'track':
            return await play.callback(interaction, self.tracks[int(self.values[0])][2])
        value = self.values[0]
        if self.kind == 'category':
            key, value = value.split(':', 1)
            self.state.filters[key] = value
        elif self.kind == 'year':
            self.state.filters['year'] = '' if value == 'all' else value
        else:
            self.state.filters['sort'] = value
        self.state.window = 0
        await refresh_catalog(interaction, self.state)

class SearchModal(discord.ui.Modal, title='Поиск на grime.fm'):
    query = discord.ui.TextInput(label='Название трека или артист', placeholder='Например, recept или TUNDER GOD', max_length=100)
    def __init__(self, state):
        super().__init__()
        self.state = state
    async def on_submit(self, interaction: discord.Interaction):
        self.state.query = str(self.query).strip()
        self.state.window = 0
        await refresh_catalog(interaction, self.state, new_message=True)

class CatalogButton(discord.ui.Button):
    def __init__(self, state, action, label, disabled=False):
        self.state, self.action = state, action
        super().__init__(label=label, style=discord.ButtonStyle.secondary, disabled=disabled, row=4)
    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.state.owner:
            return await interaction.response.send_message('Открой свой каталог командой /play.', ephemeral=True)
        if self.action == 'search':
            return await interaction.response.send_modal(SearchModal(self.state))
        if self.action == 'prev': self.state.window = max(0, self.state.window - 1)
        if self.action == 'next': self.state.window += 1
        if self.action == 'lyrics':
            self.state.filters['lyrics'] = '' if self.state.filters['lyrics'] else '1'
            self.state.window = 0
        if self.action == 'per':
            choices = [25, 50, 100, 200]
            self.state.per = choices[(choices.index(self.state.per)+1)%len(choices)]
            self.state.window = 0
        await refresh_catalog(interaction, self.state)

async def refresh_catalog(interaction: discord.Interaction, state: CatalogState, new_message=False):
    await interaction.response.defer(thinking=new_message)
    try:
        tracks, has_more = await state.get_tracks()
    except (ValueError, aiohttp.ClientError, asyncio.TimeoutError):
        logging.exception('Не удалось загрузить каталог grime.fm')
        return await interaction.followup.send('Не удалось загрузить каталог grime.fm. Попробуй позже.', ephemeral=True)
    view = discord.ui.View(timeout=600)
    view.add_item(CatalogSelect(state, 'category'))
    view.add_item(CatalogSelect(state, 'year'))
    view.add_item(CatalogSelect(state, 'sort'))
    if tracks: view.add_item(CatalogSelect(state, 'track', tracks))
    view.add_item(CatalogButton(state, 'search', '🔎 Поиск'))
    view.add_item(CatalogButton(state, 'prev', '←', disabled=state.window == 0))
    view.add_item(CatalogButton(state, 'next', '→', disabled=not has_more))
    view.add_item(CatalogButton(state, 'lyrics', 'Текст ✓' if state.filters['lyrics'] else 'С текстом'))
    view.add_item(CatalogButton(state, 'per', f'{state.per} / стр'))
    f = state.filters
    types = {'': 'все', 'track': 'треки', 'instrumental': 'биты'}
    contexts = {'': 'все', 'battles': 'баттлы', 'releases': 'релизы', 'albums': 'альбомы'}
    lines = [f'**{state.window*5+i}. {discord.utils.escape_markdown(t)[:100]}** — {discord.utils.escape_markdown(a)[:80] or "артист не указан"}' for i,(t,a,_) in enumerate(tracks, 1)]
    description = f"Тип: {types.get(f['type'])} · Раздел: {contexts.get(f['context'])} · Год: {f['year'] or 'все'}"
    if state.query: description += f"\\nПоиск: {discord.utils.escape_markdown(state.query)[:100]}"
    description += '\\n\\n' + ('\\n'.join(lines) if lines else 'Ничего не найдено.')
    embed = discord.Embed(title='ТРЕКИ · grime.fm', description=description, colour=0xb9ff36)
    embed.set_footer(text=f'Показано по 5 треков в Discord · строк на странице сайта: {state.per}')
    if new_message:
        await interaction.followup.send(embed=embed, view=view, allowed_mentions=discord.AllowedMentions.none())
    else:
        await interaction.edit_original_response(embed=embed, view=view, allowed_mentions=discord.AllowedMentions.none())

async def show_catalog(interaction: discord.Interaction, query: str):
    if len(query) > 100:
        return await interaction.response.send_message('Запрос должен быть не длиннее 100 символов.', ephemeral=True)
    await refresh_catalog(interaction, CatalogState(interaction.user.id, query), new_message=True)

@bot.tree.command(name='skip', description='Переключить трек')
@app_commands.guild_only()
async def skip(interaction: discord.Interaction):
    _, error = same_channel(interaction)
    if error:
        return await interaction.response.send_message(error, ephemeral=True)
    voice = interaction.guild.voice_client
    if not voice or not (voice.is_playing() or voice.is_paused()):
        return await interaction.response.send_message('Сейчас ничего не играет.', ephemeral=True)
    voice.stop()
    await interaction.response.send_message('Пропускаю трек.')

@bot.tree.command(name='queue', description='Показать текущий трек и очередь')
@app_commands.guild_only()
async def queue(interaction: discord.Interaction):
    player = players.get(interaction.guild_id)
    if not player or not (player.current or player.queue):
        return await interaction.response.send_message('Очередь пуста.')
    lines = [f'Сейчас: {player.current}'] if player.current else []
    lines += [f'{i}. {item[0]}' for i, item in enumerate(list(player.queue)[:10], 1)]
    if len(player.queue) > 10:
        lines.append(f'И ещё: {len(player.queue) - 10}')
    await interaction.response.send_message('\n'.join(lines), suppress_embeds=True)

@bot.tree.command(name='stop', description='Очистить очередь и отключить бота')
@app_commands.guild_only()
async def stop(interaction: discord.Interaction):
    _, error = same_channel(interaction)
    if error:
        return await interaction.response.send_message(error, ephemeral=True)
    player = players.setdefault(interaction.guild_id, Player())
    async with player.lock:
        player.generation += 1
        for _, path in player.queue:
            remove_file(path)
        player.queue.clear()
        player.current = None
        if interaction.guild.voice_client:
            await interaction.guild.voice_client.disconnect(force=True)
    await interaction.response.send_message('Остановлено, очередь очищена.')

if __name__ == '__main__':
    if not TOKEN:
        raise SystemExit('Укажи DISCORD_TOKEN в файле .env')
    if GUILD_ID and not GUILD_ID.isdigit():
        raise SystemExit('DISCORD_GUILD_ID должен быть числовым ID сервера')
    if not os.path.isfile(FFMPEG):
        raise SystemExit('FFmpeg не найден.')
    bot.run(TOKEN, log_handler=None)
