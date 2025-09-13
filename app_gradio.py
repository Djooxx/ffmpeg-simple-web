import logging
# 设置日志
logger = logging.getLogger(__name__)
logger.propagate = False  # 关键：关闭继承传播
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于将日志消息打印到控制台
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# 创建一个formatter，然后添加到handler中
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# 将handler添加到logger中
logger.addHandler(handler)

logger.info("启动服务中...")

import time
import os
import json
import re
import time
import math
import numpy as np
import soundfile as sf
import ffmpeg
import gradio as gr
import whisper
from faster_whisper import WhisperModel
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import ollama
from pathlib import Path
import srt
from datetime import timedelta
from typing import List, Tuple, Union
import torch
import tqdm
import mysql.connector # 添加 MySQL 连接器
import mysql.connector.pooling # 添加连接池支持
import traceback
from yt_dlp import YoutubeDL
import tempfile
from urllib.parse import urlparse, urlunparse
# 假设Kokoro相关库已安装
from kokoro import KModel, KPipeline  # 需确认实际导入方式
import cv2
import base64
import openai

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
# 全局Ollama客户端实例
ollama_client = ollama.Client(host='http://127.0.0.1:11434')


# 设备选择
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"使用设备: {device}")


is_init_kokoro = False
en_pipeline = None  # 初始化为 None
pipeline_v1_1 = None  # 初始化为 None
pipeline_v1_0 = None  # 初始化为 None
def en_callable(text):
    return next(en_pipeline(text, voice='af_alloy')).phonemes;

# 处理路径
def process_path(path: str) -> str:
    return path.replace('"', '').strip()

# 速度调整
def speed_callable(len_ps: int) -> float:
    speed = 0.9
    if len_ps <= 150:
        speed = 1
    elif len_ps < 250:
        speed = 1 - (len_ps - 150) / 1000
    return speed * 1.1

# 文字生成音频数据（kokoro）
def generate_audio_data(text: str, voice: str) -> List[np.ndarray]:
    global is_init_kokoro, en_pipeline, pipeline_v1_1, pipeline_v1_0
    if not is_init_kokoro:
        # 加载 Kokoro 模型
        logger.info("加载 Kokoro 模型")
        model_v1_0 = KModel(repo_id='hexgrad/Kokoro-82M').to(device).eval()
        en_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', model=model_v1_0)
        model_v1_1 = KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh').to(device).eval()
        pipeline_v1_1 = KPipeline(repo_id='hexgrad/Kokoro-82M-v1.1-zh', lang_code='z', en_callable=en_callable, model=model_v1_1)
        pipeline_v1_0 = KPipeline(repo_id='hexgrad/Kokoro-82M', lang_code='z', en_callable=en_callable, model=model_v1_0)
        is_init_kokoro = True  # 设置初始化标志
        logger.info("加载 Kokoro 模型 完成")
    wavs = []
    try:
        if re.search(r'\d$', voice):
            pipeline = pipeline_v1_1
        else:
            pipeline = pipeline_v1_0
        sentences = []
        logger.info("文本清理中...")
        text = re.sub(r'\r\n|\n|\r', '。', text)
        if len(text) > 100:
            split_pattern = r'([。;；!！?？]|…|\.{2,})'
            sentences_temp = re.split(split_pattern, text)
            MAX_LEN = 100 # 设定我们的目标长度阈值
            for segment in sentences_temp:
                if not segment: # re.split 可能会产生空字符串
                    continue
                # 核心修改点在这里：
                # 检查这个 segment (它要么是句子内容，要么是分隔符如 '。')
                if len(segment) > MAX_LEN:
                    # 这个 segment 是文本，并且它太长了，需要按逗号处理
                    
                    # 1. 使用捕获组 ( ) 来分割，这样逗号也会被保留在列表中
                    # "a,b，c" -> ['a', ',', 'b', '，', 'c']
                    sub_parts = re.split(r'([，,])', segment)
                    
                    current_chunk = "" # 初始化当前块
                    
                    # 2. 迭代这些部分。我们总是尝试组合 (文本 + 紧随其后的分隔符)
                    for i in range(0, len(sub_parts), 2): # 步长为 2，跳过分隔符（我们手动处理它）
                        
                        text_part = sub_parts[i]
                        delimiter = sub_parts[i+1] if i + 1 < len(sub_parts) else "" # 获取逗号（如果存在）
                        
                        segment_with_delim = text_part + delimiter # (例如 "这是第一部分" + ",")

                        # 3. 贪心组合逻辑
                        # 检查如果把这个新部分加进来，是否会超过最大长度
                        if len(current_chunk) + len(segment_with_delim) > MAX_LEN:
                            # 超过了！
                            # A. 我们必须先提交(append)之前积累的 current_chunk (前提是它不为空)
                            if current_chunk:
                                sentences.append(current_chunk)
                            
                            # B. 这个 segment_with_delim 成为新块的开始
                            #    (注意：如果这个 segment_with_delim 本身就>MAX_LEN，我们也别无选择，
                            #     因为它内部没有逗号了。我们也必须接受它作为一整个块。)
                            current_chunk = segment_with_delim
                        else:
                            # 没超过，很好，继续累积
                            current_chunk += segment_with_delim
                    
                    # 4. 循环结束后，不要忘记添加最后一个累积的块
                    if current_chunk:
                        sentences.append(current_chunk)
                        
                else:
                    # 这部分要么是短句子 (<= 100)，要么是分隔符 (如 '。', '！')
                    # 按照原始逻辑添加即可
                    sentences.append(segment)
            
            # (过滤掉任何可能产生的仅包含空白的字符串)
            sentences = [s for s in sentences if s and s.strip()]
        else:
            sentences = [text]
        punctuation_pattern = r'^[\s。;|；!！?>？….,，、\-()（）“”"‘’\'*`\u2018\u2019\u201c\u201d\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+$'
        texts = [(sentence.strip(),) for sentence in sentences if sentence.strip() and not re.match(punctuation_pattern, sentence.strip())]
        if not texts:
            logger.info("没有有效文本内容，跳过音频生成")
            return wavs
        logger.info("文本清理完成")
        for paragraph in tqdm.tqdm(texts):
            for i, sentence in enumerate(paragraph):
                logger.info(f"处理文字: {sentence}")
                generator = pipeline(sentence, voice=voice, speed=speed_callable(len(sentence)))
                result = next(generator)
                wav = result.audio
                if i == 0 and wavs:
                    wav = np.concatenate([np.zeros(5000), wav])
                wavs.append(wav)
        return wavs
    except Exception as e:
        logger.error(f"生成音频错误: {str(e)}")
        return wavs

# 文字转语音（适配Gradio）
def text_to_speech(text: str, voice: str) -> Union[Tuple[int, np.ndarray], None]: # Modified return type
    try:
        wavs = generate_audio_data(text, voice)
        if not wavs:
            logger.error("没有生成音频数据")
            return None # Modified line
        audio_data = np.concatenate(wavs)
        sample_rate = 24000  # Kokoro固定采样率
        return sample_rate, audio_data
    except Exception as e:
        logger.error(f"TTS错误: {str(e)}")
        return None # Modified line

# SRT文件转音频
def srt_to_audio(srt_path: str) -> Union[Tuple[int, np.ndarray], None]: # Modified return type
    srt_path = process_path(srt_path)
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            subtitles = list(srt.parse(f))
        sample_rate = 24000
        full_audio = np.array([])
        last_end = 0
        for subtitle in subtitles:
            text = subtitle.content
            start_time = subtitle.start.total_seconds()
            # 添加静音（从上一个片段结束到当前开始）
            if full_audio.size and start_time > last_end:
                silence = np.zeros(int(sample_rate * (start_time - last_end)))
                full_audio = np.concatenate([full_audio, silence])
            # 生成音频
            temp_audio_tuple = text_to_speech(text, voice="zf_xiaoxiao") # Modified to handle None
            if temp_audio_tuple is None:
                logger.warning(f"Skipping subtitle due to TTS failure for: {text[:50]}...")
                continue
            temp_rate, temp_audio = temp_audio_tuple
            if temp_audio is None or temp_audio.size == 0: # Additional check for safety
                logger.warning(f"Skipping subtitle due to empty audio from TTS for: {text[:50]}...")
                continue
            full_audio = np.concatenate([full_audio, temp_audio]) if full_audio.size else temp_audio
            last_end = start_time + len(temp_audio) / sample_rate
        if full_audio.size == 0:
            logger.warning("SRT转音频后，full_audio为空")
            return None # Modified line
        return sample_rate, full_audio
    except Exception as e:
        logger.error(f"SRT转音频错误: {str(e)}")
        return None # Modified line

# 文件大小单位转换
def convert_size(size_bytes):
    """将字节数转换为合适的单位"""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

# 视频信息获取
def get_video_info(video_path: str) -> str:
    """
    获取指定视频文件的基本信息，包括时长、大小、分辨率、编码格式、比特率和帧率。

    Args:
        video_path (str): 视频文件的路径，支持本地文件路径。

    Returns:
        str: 格式化的视频信息字符串，包含时长、文件大小、分辨率、编码格式、比特率和帧率。
             如果文件不存在、不是文件或无法解析，则返回错误信息。

    Raises:
        Exception: 如果 ffmpeg.probe 无法处理视频文件，可能抛出异常并返回错误信息。
    """
    video_path = process_path(video_path)
    if not os.path.exists(video_path):
        return f"文件路径不存在"
    if not os.path.isfile(video_path):
        return f"路径不是文件"
    try:
        probe = ffmpeg.probe(video_path)
        for stream in probe["streams"]:
            if stream["codec_type"] in ["video"]:
                format_info = probe["format"]
                duration = float(format_info["duration"])
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                duration_str = f"{hours}小时 {minutes}分钟 {seconds}秒"
                size = os.path.getsize(video_path)
                bitrate = (float(format_info["bit_rate"]) / 1000000) if "bit_rate" in format_info else "未知"
                frame_rate = eval(stream["avg_frame_rate"]) if stream.get("avg_frame_rate") else "未知"
                return f"""
                **视频信息**
                - 时长: {duration_str}
                - 文件大小: {convert_size(int(size))}
                - 分辨率: {stream["width"]}x{stream["height"]}
                - 编码格式: {stream["codec_name"]}
                - 比特率: {bitrate:.2f} Mbps
                - 帧率: {frame_rate} fps
                """
        return f"错误: 未识别到视频流"        
    except Exception as e:
        return f"错误: {str(e)}"

# 视频截取
def trim_video(video_path: str, start_time: float, end_time: float) -> str:
    video_path = process_path(video_path)
    if not os.path.exists(video_path):
        return f"文件路径不存在"
    if not os.path.isfile(video_path):
        return f"路径不是文件"
    if start_time < 0 or end_time < 0:
        return f"起始时间和结束时间必须大于0"
    if start_time >= end_time:
        return f"结束时间必须大于起始时间"
    timestamp = int(time.time())
    # 获取输入文件扩展名
    input_ext = video_path.split('.')[-1].lower()
    output_path = video_path.rsplit('.', 1)[0] + f'_{start_time}s-{end_time}s_{timestamp}.{input_ext}'
    bitrate = get_target_video_bitrate(video_path)
    try:
        stream = ffmpeg.input(
                    video_path,
                    ss=start_time,
                    t=end_time - start_time,
                    hwaccel="cuda"
                )
        stream = ffmpeg.output(
                    stream,
                    output_path,
                    vcodec="hevc_nvenc",    # 视频编码器改为 NVENC H.265
                    preset="p5",            # 编码预设
                    rc="vbr",               # 可变比特率模式
                    video_bitrate=bitrate,  # 动态比特率（如 "5M"）
                    acodec="copy"           # 音频直接复制（原 c="copy" 需拆分）
                )
        ffmpeg.run(stream)
        return f"视频截取成功！保存路径：{output_path}"
    except Exception as e:
        return f"错误: {str(e)}"

# 获取目标视频码率
def get_target_video_bitrate(video_path):
    """
    根据视频分辨率返回目标比特率。

    参数:
        video_path (str): 输入视频文件路径

    返回:
        str: 比特率字符串，例如 '15M'、'5M'、'3M'

    默认规则:
        - 4K (≥3840x2160): 15Mbps
        - 1080p (≥1920x1080): 5Mbps
        - 720p 或更低: 3Mbps
    """
    try:
        # 获取视频分辨率
        width, height = get_video_resolution(video_path)
        # 根据分辨率设置比特率
        if width >= 3840 or height >= 2160:  # 4K
            return '15M'
        elif width >= 1920 or height >= 1080:  # 1080p
            return '5M'
        else:  # 720p 或更低
            return '3M'

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"无法获取 {video_path} 的分辨率，错误: {e}，默认使用 5M 比特率")
        return '5M'  # 默认比特率

# 获取视频分辨率
def get_video_resolution(video_path):
    try:
        # 构建流式分析管道
        probe = ffmpeg.probe(
            video_path,
            v='error',  # 日志级别
            select_streams='v:0',  # 选择第一个视频流
            show_entries='stream=width,height',  # 提取宽高字段
            of='json'  # 输出JSON格式
        )

        # 解析流信息（保持与原逻辑一致）
        width = probe['streams'][0]['width']
        height = probe['streams'][0]['height']
        return width, height
    except (ffmpeg.Error, json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"无法获取 {video_path} 的分辨率，错误: {e}，默认使用 1920x1080")
        return 1920, 1080

# 音频提取
def extract_audio(video_path: str) -> str:
    video_path = process_path(video_path)
    if not os.path.exists(video_path):
        return f"文件路径不存在"
    if not os.path.isfile(video_path):
        return f"路径不是文件"
    # 获取输入文件扩展名
    input_ext = video_path.split('.')[-1].lower()
    # 支持常见视频格式
    supported_formats = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']
    if input_ext not in supported_formats:
        return f"不支持的文件格式: {input_ext}"
    # 根据输入格式选择输出格式
    output_ext = 'mp3'  # 默认输出mp3
    timestamp = int(time.time())
    output_path = video_path.rsplit('.', 1)[0] + f'_{timestamp}.{output_ext}'
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(
            stream['a:0'],  # 显式选择第一个音频流
            output_path,
            acodec='libmp3lame',  # 指定 MP3 编码器
            **{'q:a': 0},        # 最高音质参数
            ac=2               # 强制立体声输出
        )
        ffmpeg.run(stream)
        return f"音频提取成功！保存路径：{output_path}"
    except Exception as e:
        return f"错误: {str(e)}"

# 音频截取
def trim_audio(audio_path: str, start_time: float, end_time: float) -> str:
    audio_path = process_path(audio_path)
    if not os.path.exists(audio_path):
        return f"文件路径不存在"
    if not os.path.isfile(audio_path):
        return f"路径不是文件"
    if start_time < 0 or end_time < 0:
        return f"起始时间和结束时间必须大于0"
    if start_time >= end_time:
        return f"结束时间必须大于起始时间"
    timestamp = int(time.time())
    # 获取输入文件扩展名
    input_ext = audio_path.split('.')[-1].lower()
    # 支持常见音频格式
    supported_formats = ['mp3', 'wav', 'flac', 'aac', 'ogg']
    if input_ext not in supported_formats:
        return jsonify({'success': False, 'error': f'不支持的音频格式: {input_ext}'})
    output_path = audio_path.rsplit('.', 1)[0] + f'_{start_time}s-{end_time}s_{timestamp}.{input_ext}'
    try:
        stream = ffmpeg.input(audio_path, ss=start_time, t=end_time - start_time)
        stream = ffmpeg.output(stream, output_path,acodec='libmp3lame', c="copy")
        ffmpeg.run(stream)
        return f"音频截取成功！保存路径：{output_path}"
    except Exception as e:
        return f"错误: {str(e)}"

# 视频格式转换
def convert_video(video_path: str, output_format: str) -> str:
    video_path = process_path(video_path)
    if not os.path.exists(video_path):
        return f"文件路径不存在"
    if not os.path.isfile(video_path):
        return f"路径不是文件"
    output_format = output_format.strip().lstrip('.')
    if not output_format:
        return "目标格式不能为空"
    common_formats = ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm']
    if output_format.lower() not in common_formats:
        return f"不支持的目标视频格式: {output_format}"
    timestamp = int(time.time())
    output_path = video_path.rsplit('.', 1)[0] + f'_{timestamp}.{output_format}'
    bitrate = get_target_video_bitrate(video_path)
    try:
        # 设置输入流并启用 CUDA 硬件加速
        stream = ffmpeg.input(video_path, hwaccel='cuda')  # 启用 GPU 解码

        # 设置输出流，使用 HEVC 编码器、预设、动态比特率，并复制音频流
        stream = ffmpeg.output(
            stream,
            output_path,
            vcodec='hevc_nvenc',    # 使用 NVIDIA HEVC 编码器
            preset='p5',            # NVENC 预设
            video_bitrate=bitrate,  # 设置视频比特率，对应 -b:v
            rc='vbr',               # 可变比特率模式，对应 -rc vbr
            acodec='copy',          # 复制音频流
        )

        # 执行命令
        ffmpeg.run(stream)
        return f"""
        视频转换成功！保存路径：
        {output_path}
        """
    except Exception as e:
        return f"错误: {str(e)}"

# 音频转文字 (Whisper)
def audio_to_text(audio_path: str) -> str:
    audio_path = process_path(audio_path)
    if not os.path.exists(audio_path):
        return f"文件路径不存在"
    if not os.path.isfile(audio_path):
        return f"路径不是文件"
    WHISPER_MODEL = whisper.load_model("large-v3")
    try:
        result = WHISPER_MODEL.transcribe(audio_path)
        timestamp = int(time.time())
        output_path = audio_path.rsplit('.', 1)[0] + f'_{timestamp}' +'.srt'
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], 1):
                start = timedelta(seconds=segment["start"])
                end = timedelta(seconds=segment["end"])
                f.write(f"{i}\n{start} --> {end}\n{segment['text']}\n\n")
        return f"转换成功！字幕文件路径：{output_path}"
    except Exception as e:
        return f"错误: {str(e)}"

# 音频转文字 (Faster Whisper)
def faster_whisper(audio_path: str) -> str:
    audio_path = process_path(audio_path)
    if not os.path.exists(audio_path):
        return f"文件路径不存在"
    if not os.path.isfile(audio_path):
        return f"路径不是文件"
    FASTER_WHISPER_MODEL = WhisperModel("large-v3", device=device)
    try:
        segments, _ = FASTER_WHISPER_MODEL.transcribe(audio_path, beam_size=5)
        timestamp = int(time.time())
        output_path = audio_path.rsplit('.', 1)[0] + f'_{timestamp}' +'.srt'
        subtitles = []
        for i, segment in enumerate(segments, 1):
            start = timedelta(seconds=segment.start)
            end = timedelta(seconds=segment.end)
            subtitles.append(srt.Subtitle(i, start, end, segment.text))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(subtitles))
        return f"转换成功！字幕文件路径：{output_path}"
    except Exception as e:
        return f"错误: {str(e)}"

def clean_bilibili_url(video_url):
    # 判断是否为 B 站链接
    if "bilibili.com" not in video_url:
        return video_url

    # 解析 URL 结构
    parsed = urlparse(video_url)
    # 重构 URL：将 query（参数）和 fragment（锚点）置空
    cleaned_url = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        "",  # 删除所有查询参数
        ""   # 可选：删除锚点（若需要保留锚点则保留 parsed.fragment）
    ))
    return cleaned_url

# 总结视频内容
def summarize_video_url(video_url: str, ollama_model: str, tts_voice: str) -> Tuple[str, Union[Tuple[int, np.ndarray], None], gr.update]: # Modified return type
    try:
        if not video_url.strip():
            return "", None, gr.update(value="错误: 请输入视频URL", visible=True) # Modified line
        if not ollama_model:
            return "", None, gr.update(value="错误: 请选择Ollama模型", visible=True) # Modified line
        if not tts_voice:
            return "", None, gr.update(value="错误: 请选择TTS语音", visible=True) # Modified line

        logger.info(f"开始总结视频URL: {video_url} 使用模型: {ollama_model} 和语音: {tts_voice}")

        video_url = clean_bilibili_url(video_url)
        logger.info(f"清理后的视频URL: {video_url}")

        video_title = "未知标题" # 初始化标题
        # 1. 下载音频
        with tempfile.TemporaryDirectory() as tmpdir:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(tmpdir, '%(title)s.%(ext)s'),
                'keepvideo': False,
                'noplaylist': True,
                'quiet': False,
                'no_warnings': False,
                # 'proxy': 'http://127.0.0.1:7890',  # 添加代理设置
            }
            downloaded_audio_path = None
            with YoutubeDL(ydl_opts) as ydl:
                try:
                    logger.info(f"开始使用 yt-dlp 下载: {video_url}")
                    info_dict = ydl.extract_info(video_url, download=True) # download=True 会执行下载
                    video_title = info_dict.get('title', '未知标题') # <-- 获取标题
                    logger.info(f"提取到的视频标题: {video_title}")

                    # ydl.prepare_filename(info_dict) 返回的是基于模板的文件名，但不一定是实际下载的文件名
                    # 更好的方式是检查 tmpdir 中的文件
                    downloaded_files = os.listdir(tmpdir)
                    if not downloaded_files:
                        logger.error("yt_dlp 未能下载任何文件")
                        return "", None, gr.update(value="错误: 无法从URL下载音频 (无文件)", visible=True)
                    
                    # 假设只有一个音频文件被下载 (因为 noplaylist=True)
                    # 并且文件名可能包含特殊字符，yt-dlp会处理
                    # 我们只需要tmpdir中的第一个（也应该是唯一一个）文件
                    downloaded_audio_path = os.path.join(tmpdir, downloaded_files[0])
                    logger.info(f"音频已下载到: {downloaded_audio_path}")
                except Exception as e:
                    logger.error(f"yt_dlp 下载或提取信息错误: {str(e)}")
                    # 可以尝试从错误信息中获取更具体的yt-dlp错误
                    error_message = str(e)
                    if "Unsupported URL" in error_message:
                         actual_error = "不支持的URL或视频无法访问"
                    elif "Video unavailable" in error_message:
                         actual_error = "视频不可用"
                    else:
                         actual_error = error_message # 保留原始错误的部分信息
                    return "", None, gr.update(value=f"错误 (yt-dlp): {actual_error}", visible=True)

            if not downloaded_audio_path or not os.path.exists(downloaded_audio_path):
                logger.error("下载的音频文件路径无效或文件不存在")
                return "", None, gr.update(value="错误: 下载的音频文件处理失败", visible=True) # Modified line

            # 2. 音频转文字 (SenseVoiceSmall)
            logger.info(f"开始使用SenseVoice进行语音转文字: {downloaded_audio_path}")
            sense_voice_result_str = sense_voice(downloaded_audio_path)
            logger.info(f"SenseVoice 结果: {sense_voice_result_str}")

            # 从结果字符串中提取文本
            transcribed_text = ""
            if "错误:" in sense_voice_result_str:
                logger.error(f"SenseVoice转换失败: {sense_voice_result_str}")
                # Extract the core error message from sense_voice_result_str if it's formatted like "错误: actual error"
                actual_error_message = sense_voice_result_str.split("错误:", 1)[-1].strip() if "错误:" in sense_voice_result_str else sense_voice_result_str
                return "", None, gr.update(value=f"错误 (SenseVoice): {actual_error_message}", visible=True) # Modified line

            # 改进文本提取逻辑
            match = re.search(r"文本内容：(.*?)(\n文件保存路径：|$)", sense_voice_result_str, re.DOTALL)
            if match:
                transcribed_text = match.group(1).strip()
            else:
                # 兼容旧版或仅文本输出的情况
                if sense_voice_result_str.startswith("转换成功！"):
                     # 尝试去除 "转换成功！\n文本内容：" 和 "\n文件保存路径：..."
                    text_content_part = sense_voice_result_str.replace("转换成功！", "").replace("文本内容：", "").strip()
                    path_part_index = text_content_part.find("文件保存路径：")
                    if path_part_index != -1:
                        transcribed_text = text_content_part[:path_part_index].strip()
                    else:
                        transcribed_text = text_content_part # 如果没有路径信息，则认为剩余部分都是文本
                else: # 如果格式不匹配，且不包含错误，则认为整个结果是文本
                    transcribed_text = sense_voice_result_str.strip()

            if not transcribed_text:
                logger.warning("SenseVoice 未能提取有效文本")
                return "", None, gr.update(value="错误: 语音转文字未能提取有效文本", visible=True) # Modified line
            logger.info(f"提取的文本: {transcribed_text[:200]}...")

            # 3. 调用大语言模型进行总结 (chat_with_ollama)
            logger.info(f"开始使用Ollama模型 '{ollama_model}' 进行总结")
            # 准备一个初始的空历史记录
            initial_history = []
            optimized_prompt = (
                f"视频标题：【{video_title}】\n\n"
                f"这是一段来自上述标题视频的语音转录文本（由SenseVoiceSmall识别，源自Bilibili或YouTube）。"
                f"请你用中文为其撰写一份清晰、简洁、易于理解的内容摘要。\n"
                f"摘要应重点突出以下几点：\n"
                f"1. 视频的核心主题或主要内容是什么？（请结合标题和文本内容判断）\n"
                f"2. 视频中讨论了哪些关键的观点、信息、步骤或有趣的亮点？（如果适合，请用分点列出）\n"
                f"3. 视频最终想要传达的核心信息或结论是什么？\n"
                f"请在总结时，尽量忽略原始语音中可能存在的口头禅、不必要的重复或不流畅之处，专注于提炼有价值的信息。"
                f"目标是让未观看视频的人也能快速把握视频的精髓。\n\n"
                f"视频文本如下：\n{transcribed_text}"
            )
            summary_history, audio_tuple, ollama_error = chat_with_ollama(
                message=optimized_prompt,
                model=ollama_model,
                voice=tts_voice,
                history=initial_history
            )

            if ollama_error:
                logger.error(f"Ollama聊天错误: {ollama_error}")
                return "", None, gr.update(value=f"错误 (Ollama): {ollama_error}", visible=True) # Modified line
            
            sample_rate, audio_data = audio_tuple if audio_tuple is not None else (None, None)

            summarized_text = ""
            if summary_history and len(summary_history) > 0 and len(summary_history[-1]) == 2:
                summarized_text = summary_history[-1][1] # 获取Ollama的回复

            if not summarized_text:
                logger.warning("Ollama未能生成总结文本")
                return "", None, gr.update(value="错误: 大语言模型未能生成总结文本", visible=True) # Modified line

            logger.info(f"总结文本: {summarized_text[:200]}...")
            logger.info(f"生成语音数据，采样率: {sample_rate}, 数据长度: {len(audio_data) if audio_data is not None and isinstance(audio_data, np.ndarray) else 0}")

            # 4. 返回总结文本和语音
            return summarized_text, (sample_rate, audio_data) if sample_rate is not None and audio_data is not None else None, gr.update(value="", visible=False) # Modified line

    except Exception as e:
        logger.error(f"视频总结过程中发生意外错误: {str(e)}\n{traceback.format_exc()}")
        return "", None, gr.update(value=f"严重错误: {str(e)}", visible=True) # Modified line

# 音频转文字 (SenseVoiceSmall)
def sense_voice(audio_path: str) -> str:
    audio_path = process_path(audio_path)
    model_dir = "iic/SenseVoiceSmall"
    SENSE_VOICE_MODEL = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        disable_update=True
    )
    try:
        result = SENSE_VOICE_MODEL.generate(
            input=audio_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(result[0]["text"])
        timestamp = int(time.time())
        output_path = audio_path.rsplit('.', 1)[0] + f'_sense_voice_{timestamp}.txt'
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        return f"转换成功！\n文本内容：{text}\n文件保存路径：{output_path}"
    except Exception as e:
        return f"错误: {str(e)}"

# 获取Ollama模型
def get_ollama_models() -> List[str]:
    try:
        # 使用全局Ollama客户端
        models = ollama_client.list()
        return [model["model"] for model in models["models"]]
    except Exception:
        return []

OLLAMA_HOST = 'http://127.0.0.1:11434'
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
def get_all_models() -> List[str]:
    """
    获取本地 Ollama 和 LM Studio 上所有可用的模型。

    此函数会分别尝试连接两个服务：
    - 从 Ollama 获取的模型会以 'ollama:' 作为前缀。
    - 从 LM Studio 获取的模型会以 'lms:' 作为前缀。
    
    如果某个服务无法连接，函数会打印一条错误信息但不会中断，
    并继续尝试获取另一个服务中的模型。

    Returns:
        List[str]: 一个包含所有可用模型名称（带前缀）的列表。
                   如果两个服务都无法访问，则返回一个空列表。
    """
    all_models = []

    # --- 1. 尝试获取 Ollama 模型 ---
    try:
        logger.info(f"正在尝试连接 Ollama 服务于 {OLLAMA_HOST}...")
        client = ollama.Client(host=OLLAMA_HOST)
        models_data = client.list()
        
        # 使用列表推导式为每个模型添加前缀并添加到主列表
        ollama_models = [f"ollama:{model['model']}" for model in models_data["models"]]
        all_models.extend(ollama_models)
        logger.info(f"成功从 Ollama 获取 {len(ollama_models)} 个模型。")

    except Exception as e:
        logger.info(f"获取 Ollama 模型失败。请检查 Ollama 服务是否正在运行。错误: {e}")

    # --- 2. 尝试获取 LM Studio 模型 ---
    try:
        logger.info(f"正在尝试连接 LM Studio 服务于 {LM_STUDIO_BASE_URL}...")
        # api_key 对于本地服务器不是必需的，但 openai 库要求提供
        client = openai.OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="not-needed")
        response = client.with_options(timeout=1, max_retries=0).models.list()
        
        # 为每个模型ID添加前缀并添加到主列表
        lms_models = [f"lms:{model.id}" for model in response.data]
        all_models.extend(lms_models)
        logger.info(f"成功从 LM Studio 获取 {len(lms_models)} 个模型。")

    except Exception as e:
        logger.error(f"获取 LM Studio 模型失败。请检查 LM Studio 服务器是否已启动并在运行。错误: {e}")

    return all_models


def chat_with_ollama(message: str, model: str, voice: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], Tuple[int, np.ndarray], str]:
    try:
        if not model:
            logger.warning("未选择模型，返回错误提示")
            return history, None, "错误: 请先选择一个模型"
        if not message.strip():
            logger.warning("输入消息为空，返回错误提示")
            return history, None, "错误: 请输入有效消息"
        messages = []
        for user_msg, assistant_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": message})
        response = ollama_client.chat(model=model, messages=messages)
        assistant_text = response["message"]["content"]
        logger.info(f"原始回复: {assistant_text}")
        # 改进正则表达式，匹配多行<think>标签
        clean_text = re.sub(r'(?i)<think\s*[^>]*>[\s\S]*?</think\s*>', '', assistant_text).strip()
        logger.info(f"过滤后文本: {clean_text}")
        # 验证过滤
        # think_matches = re.findall(r'(?i)<think\s*[^>]*>[\s\S]*?</think\s*>', assistant_text)
        # logger.info(f"匹配的<think>内容: {think_matches}")


        # 生成语音（使用过滤后的文本）
        sample_rate, audio_data = text_to_speech(clean_text, voice) if clean_text else (0, np.array([]))

        # 更新聊天历史（直接修改history）
        history.append((message, clean_text))
        logger.info(f"更新聊天历史: {history}")

        return history, (sample_rate, audio_data), ""
    except Exception as e:
        logger.error(f"聊天错误: {str(e)}")
        return history, (0, np.array([])), f"错误: {str(e)}"

# 自然语言查数据库
# MySQL 数据库配置
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 9981,
    'user': 'root',
    'password': '1234qwer',
    'database': 'money',  # 请根据实际数据库修改
    'raise_on_warnings': True
}

# 创建数据库连接池
db_pool = None

def get_db_connection():
    global db_pool
    if db_pool is None:
        try:
            db_pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="mypool",
                pool_size=5,
                pool_reset_session=True,
                **DB_CONFIG
            )
            logger.info("数据库连接池创建成功")
        except mysql.connector.Error as err:
            logger.error(f"创建数据库连接池失败: {err}")
            raise
    try:
        conn = db_pool.get_connection()
        if conn.is_connected():
            logger.debug("从连接池获取连接成功")
            return conn
        else:
            logger.error("从连接池获取的连接无效 (is_connected() is False)")
            return None
    except mysql.connector.Error as err:
        logger.error(f"从连接池获取连接失败: {err}")
        return None

# 数据库操作函数
def execute_sql(sql_query):
    """执行 SQL 查询并返回结果"""
    conn = None
    cursor = None
    logger.info(f"开始执行 SQL: {sql_query}")
    try:
        conn = get_db_connection()
        if conn is None:
            error_msg = "无法获取数据库连接"
            logger.error(error_msg)
            return False, error_msg
        cursor = conn.cursor(dictionary=True, buffered=True)
        cursor.execute(sql_query)
        if sql_query.strip().upper().startswith(("SELECT", "SHOW", "DESC")):
            result = cursor.fetchall()
            try:
                result_json = json.dumps(result, ensure_ascii=False, default=str)
            except TypeError as e:
                logger.warning(f"JSON 序列化错误: {e}, 将返回原始结果列表")
                result_json = str(result)
            logger.info(f"执行sql结果 (JSON): {result_json}")
            return True, result_json
        else:
            conn.commit()
            result_message = f"操作成功，影响行数: {cursor.rowcount}"
            logger.info(result_message)
            return True, result_message
    except mysql.connector.Error as err:
        error_msg = f"SQL 执行错误: {err}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"执行 SQL 时发生意外错误: {e}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return False, error_msg
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

def get_tables():
    """获取数据库中的所有表名"""
    logger.info("开始执行获取数据库中的所有表名 get_tables()")
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            error_msg = "无法获取数据库连接"
            logger.error(error_msg)
            return False, error_msg
        cursor = conn.cursor()
        cursor.execute("SELECT DATABASE()")
        current_db = cursor.fetchone()
        if not current_db or not current_db[0]:
            error_msg = "未选择数据库。请在 DB_CONFIG 中指定 'database'"
            logger.error(error_msg)
            return False, error_msg
        db_name = current_db[0]
        logger.info(f"当前数据库: {db_name}")
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        logger.info(f"表名查询结果: {tables}")
        return True, tables
    except mysql.connector.Error as err:
        error_msg = f"获取表列表错误: {err}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"获取表列表时发生意外错误: {e}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return False, error_msg
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
        logger.info("表名查询结束 get_tables()")

def get_columns(table_name):
    """获取指定表的列信息"""
    logger.info(f"开始获取表'{table_name}'的列信息 get_columns(table_name='{table_name}')")
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            error_msg = "无法获取数据库连接"
            logger.error(error_msg)
            return False, error_msg
        cursor = conn.cursor()
        cursor.execute(f"DESCRIBE `{table_name}`")
        columns = [column[0] for column in cursor.fetchall()]
        if not columns:
            error_msg = f"表 '{table_name}' 不存在或没有列"
            logger.warning(error_msg)
            return False, error_msg
        logger.info(f"查询到列信息 '{table_name}': {columns}")
        return True, columns
    except mysql.connector.Error as err:
        error_msg = f"获取列信息错误: {err}"
        logger.error(error_msg)
        if err.errno == mysql.connector.errorcode.ER_NO_SUCH_TABLE:
            return False, f"表 '{table_name}' 不存在"
        return False, error_msg
    except Exception as e:
        error_msg = f"获取列信息时发生意外错误: {e}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return False, error_msg
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
        logger.info(f"列信息查询结束 get_columns(table_name='{table_name}')")

def get_create_table_statement(table_name):
    """获取指定表的 CREATE TABLE 语句"""
    logger.info(f"开始查询表'{table_name}'的 CREATE TABLE 语句 get_create_table_statement(table_name='{table_name}')")
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            error_msg = "无法获取数据库连接"
            logger.error(error_msg)
            return False, error_msg
        cursor = conn.cursor()
        cursor.execute(f"SHOW CREATE TABLE `{table_name}`")
        result = cursor.fetchone()
        if result:
            create_statement = result[1]
            logger.info(f"查询建表语句成功 '{table_name}'")
            return True, create_statement
        else:
            error_msg = f"无法获取表 '{table_name}' 的 CREATE TABLE 语句"
            logger.warning(error_msg)
            return False, error_msg
    except mysql.connector.Error as err:
        error_msg = f"获取 CREATE TABLE 语句错误: {err}"
        logger.error(error_msg)
        if err.errno == mysql.connector.errorcode.ER_NO_SUCH_TABLE:
            return False, f"表 '{table_name}' 不存在"
        return False, error_msg
    except Exception as e:
        error_msg = f"获取 CREATE TABLE 语句时发生意外错误: {e}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return False, error_msg
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
        logger.info(f"查询建表结束 get_create_table_statement(table_name='{table_name}')")

# SQL 生成的系统提示 (GPT-5版)
SQL_SYSTEM_PROMPT = """
你是MySQL查询AI代理。任务：严格按JSON格式响应用户请求。

## 核心指令
1. 输出必须是纯JSON，禁止任何其他文本
2. 未知Schema时禁止生成SQL查询
3. 单次只输出一个JSON指令
4. 基于完整交互历史决策
5. 返回完整结果，禁止省略

## 执行流程
按优先级严格执行：

1. **Schema探索**（最高优先级）
   - 表未知：`{"action": "show_tables"}`
   - 列未知：`{"action": "show_columns", "table_name": "表名"}`
   - 需详细定义：`{"action": "show_create_table", "table_name": "表名"}`

2. **数据查询**
   - `{"sql": "SELECT查询语句"}`

3. **输出答案**
   - `{"answer": "markdown格式的最终答案"}`

## 严格JSON格式
输出必须是以下5种之一：
- `{"action": "show_tables"}`
- `{"action": "show_columns", "table_name": "..."}`
- `{"action": "show_create_table", "table_name": "..."}`
- `{"sql": "..."}`
- `{"answer": "..."}`

输入格式：
- 首次：`{"question": "问题"}`
- 后续：`{"action": "...", "result": [...]}`或`{"sql": "...", "result": "..."}`
"""
# 自然语言查数据库
def nl_to_sql(query: str, model: str) -> Tuple[bool, str]:
    """
    将自然语言查询转换为 SQL，支持 Ollama 和 LM Studio。

    Args:
        query (str): 用户的自然语言查询。
        model (str): 模型标识符，格式为 'prefix:model_name'。
                     - 'ollama:model_name' (例如 'ollama:qwen:4b')
                     - 'lms:model_name' (例如 'lms:local-model/gemma-2-9b-it-q8_0')
                     LM Studio 中的模型名称通常可以在其本地服务器UI上找到。

    Returns:
        Tuple[bool, str]: 一个元组，第一个元素表示是否成功，第二个元素是最终答案或错误信息。
    """
    logger.info(f"接收到用户查询: {query}")
    logger.info(f"使用模型字符串: {model}")

    # 1. 解析模型字符串以确定服务类型和模型名称
    try:
        model_prefix, model_name = model.split(':', 1)
    except ValueError:
        return False, f"错误: 模型字符串 '{model}' 格式不正确。请使用 'ollama:model_name' 或 'lms:model_name' 的格式。"

    max_retries = 50
    retries = 0
    messages = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"question": query})}
    ]

    while retries < max_retries:
        retries += 1
        try:
            response_content_str = None

            # 2. 根据前缀选择并调用相应的模型服务
            if model_prefix == 'ollama':
                logger.info(f"调用 Ollama (模型: {model_name})")
                client = ollama.Client(host='http://127.0.0.1:11434')
                response = client.chat(
                    model=model_name, 
                    messages=messages, 
                    options={
                        "response_format": {"type": "json_object"}
                    }
                )
                response_content_str = response["message"]["content"]

            elif model_prefix == 'lms':
                logger.info(f"调用 LM Studio (模型: {model_name})")
                # LM Studio 提供兼容 OpenAI 的接口，通常在 http://localhost:1234/v1
                # API Key对于本地LM Studio不是必需的，但openai库要求提供
                client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
                response = client.chat.completions.create(
                    model=model_name, # 在LM Studio中，这通常指向已加载的模型
                    messages=messages
                )
                response_content_str = response.choices[0].message.content
            
            else:
                return False, f"错误: 不支持的模型前缀 '{model_prefix}'。请使用 'ollama' 或 'lms'。"

            # --- 从这里开始，后续处理逻辑保持不变 ---
            
            logger.info(f"模型原始回复 ({retries}): {response_content_str}")
            if not response_content_str:
                logger.warning("模型原始回复字符串为空")
                continue

            # 清理响应 (这段逻辑对两种模型的返回都适用)
            cleaned_response_str = re.sub(r'<think>.*?</think>', '', response_content_str, flags=re.DOTALL).strip()
            cleaned_response_str = re.sub(r'^```(?:json)?\s*', '', cleaned_response_str)
            cleaned_response_str = re.sub(r'\s*```$', '', cleaned_response_str).strip()
            logger.info(f"模型格式化回复 ({retries}): {cleaned_response_str}")

            try:
                model_response_json = json.loads(cleaned_response_str)
            except json.JSONDecodeError:
                logger.error(f"无效的 JSON 响应: {cleaned_response_str[:200]}...")
                error_message_to_model = {
                    "error": "错误的json输出格式,输出的响应必须符合输出格式且为合法的JSON",
                    "details": f"Your previous response was: '{cleaned_response_str[:200]}...' which is not valid JSON. Please ensure your entire response is a single, valid JSON object matching the required schema."
                }
                messages.append({"role": "assistant", "content": cleaned_response_str}) 
                messages.append({"role": "user", "content": json.dumps(error_message_to_model, ensure_ascii=False)})
                if retries >= max_retries:
                    return False, f"错误: 达到最大重试次数，仍无法解析JSON响应。最后一次错误响应: {cleaned_response_str}"
                continue

            messages.append({"role": "assistant", "content": cleaned_response_str})

            if 'action' in model_response_json:
                action = model_response_json['action']
                db_result_payload = {"action": action}
                success = False
                result_data = None

                if action == 'show_tables':
                    success, result_data = get_tables()
                elif action == 'show_columns':
                    table_name = model_response_json.get('table_name')
                    if table_name:
                        db_result_payload['table_name'] = table_name
                        success, result_data = get_columns(table_name)
                    else:
                        result_data = "错误：'show_columns' 动作需要 'table_name' 参数"
                elif action == 'show_create_table':
                    table_name = model_response_json.get('table_name')
                    if table_name:
                        db_result_payload['table_name'] = table_name
                        success, result_data = get_create_table_statement(table_name)
                    else:
                        result_data = "错误：'show_create_table' 动作需要 'table_name' 参数"
                else:
                    result_data = f"错误：未知的动作 '{action}'"

                db_result_payload['result'] = result_data if success else f"数据库操作失败: {result_data}"
                messages.append({"role": "user", "content": json.dumps(db_result_payload)})
                continue

            elif 'sql' in model_response_json:
                sql_query = model_response_json['sql']
                success, result_data = execute_sql(sql_query)
                db_result_payload = {
                    "sql": sql_query,
                    "result": result_data if success else f"SQL 执行失败: {result_data}"
                }
                messages.append({"role": "user", "content": json.dumps(db_result_payload)})
                continue

            elif 'answer' in model_response_json:
                final_answer = model_response_json['answer']
                return True, final_answer
            else:
                logger.error(f"无法处理的 JSON 格式: {cleaned_response_str}")
                error_message_to_model = {
                    "error": "Unexpected JSON format received",
                    "details": cleaned_response_str
                }
                messages.append({"role": "user", "content": json.dumps(error_message_to_model)})
                continue

        except Exception as e:
            logger.error(f"处理 NL 到 SQL 时发生错误: {e}\n{traceback.format_exc()}")
            return False, f"错误: {str(e)}"

    logger.error("无法在指定次数内获取有效答案")
    return False, "错误: 无法在指定次数内获取有效答案"

def analyze_videos(frame: np.ndarray, model: str, history: list): # 为frame添加类型提示
    if frame is None:
        logger.error("错误：输入的帧为 None，无法处理。")
        # Return current history and an empty HTML update
        return history, "无新分析结果"

    logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing frame, will work")

    try:
        # 校正图像方向：
        # flipCode = 1：水平翻转（左右）。
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 将图像编码为 PNG 格式的字节并转为 base64
        _, buffer = cv2.imencode('.png', frame_rgb)
        base64_string = base64.b64encode(buffer).decode('utf-8')
        
        # 准备 Ollama 的消息格式
        messages = [
            {
                "role": "user",
                "content": "你是一个实时视频分析助手。请分析当前视频帧，输出一句中文，简洁描述主要内容，突出关键物体或场景。",
                "images": [base64_string]  # 直接传递 base64 字符串
            }
        ]
        # 调用 Ollama 模型
        response = ollama_client.chat(model=model, messages=messages)
        # 提取 Ollama 的回复
        ollama_result = response.get("message", {}).get("content", "Ollama 未返回有效结果")

        logger.info(f"Ollama 分析结果: {ollama_result[:40]}...")
        
        # 更新历史记录
        new_history = [f"[{time.strftime('%H:%M:%S')}] {ollama_result}"] + history
        if len(new_history) > 5:
            new_history = new_history[:5]

        # 格式化为 HTML 输出
        html_output = ""
        for i, result in enumerate(new_history):
            if i == 0:
                # 高亮最新结果
                html_output += f'<p style="background-color: #e0f7fa; color: #006064; padding: 8px; border-radius: 5px; margin-bottom: 5px;"><strong>最新:</strong> {result}</p>'
            else:
                html_output += f'<p style="background-color: #f1f1f1; padding: 8px; border-radius: 5px; margin-bottom: 5px;">{result}</p>'
        
        return new_history, html_output
    except cv2.error as e:
        logger.error(f"OpenCV 错误：无法处理. 错误信息: {e}")
        return history, f"OpenCV 错误: {e}"
    except Exception as e:
        logger.error(f"发生未知错误：无法处理. 错误信息: {e}")
        return history, f"未知错误: {e}" 


# Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 音视频处理工具 (Gradio版)")
    gr.Markdown("提供视频/音频处理、语音识别、文字转语音、SRT处理、大语言模型聊天和自然语言数据库查询功能。")

    model_choices = get_ollama_models()
    with gr.TabItem("视频工具"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                        gr.Markdown("## 通过视频URL总结内容")
                        video_url_input = gr.Textbox(label="视频URL (例如 B站, YouTube)", placeholder="请输入视频链接...")
                        # ---- START: Place dropdowns in a new Row ----
                        with gr.Row():
                            ollama_model_dropdown_video = gr.Dropdown(
                                    label="选择模型",
                                    choices=model_choices,
                                    value="qwen3:14b" if model_choices else None,
                                    interactive=True,
                                    scale=1 # Optional: adjust scale for relative width
                                )
                            tts_voice_dropdown_video =  gr.Dropdown(
                                    label="选择音色",
                                    choices=[
                                        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
                                        "zm_yunxia", "zm_yunjian", "zm_yunxi", "zm_yunyang",
                                        "zf_001", "zf_002", "zf_003", "zf_004", "zf_005", "zf_006",
                                        "zf_007", "zf_008", "zf_017", "zf_018", "zf_019", "zf_021",
                                        "zf_022", "zf_023", "zf_024", "zf_026", "zf_027", "zf_028",
                                        "zf_032", "zf_036", "zf_038", "zf_039", "zf_040", "zf_042",
                                        "zf_043", "zf_044", "zf_046", "zf_047", "zf_048", "zf_049",
                                        "zf_051", "zf_059", "zf_060", "zf_067", "zf_070", "zf_071",
                                        "zf_072", "zf_073", "zf_074", "zf_075", "zf_076", "zf_077",
                                        "zf_078", "zf_079", "zf_083", "zf_084", "zf_085", "zf_086",
                                        "zf_087", "zf_088", "zf_090", "zf_092", "zf_093", "zf_094",
                                        "zf_099", "zm_009", "zm_010", "zm_011", "zm_012", "zm_013",
                                        "zm_014", "zm_015", "zm_016", "zm_020", "zm_025", "zm_029",
                                        "zm_030", "zm_031", "zm_033", "zm_034", "zm_035", "zm_037",
                                        "zm_041", "zm_045", "zm_050", "zm_052", "zm_053", "zm_054",
                                        "zm_055", "zm_056", "zm_057", "zm_058", "zm_061", "zm_062",
                                        "zm_063", "zm_064", "zm_065", "zm_066", "zm_068", "zm_069",
                                        "zm_080", "zm_081", "zm_082", "zm_089", "zm_091", "zm_095",
                                        "zm_096", "zm_097", "zm_098", "zm_100"
                                    ],
                                    value="zf_001",
                                    scale=1 # Optional: adjust scale for relative width
                                )
                        # ---- END: Place dropdowns in a new Row ----
                        summarize_video_button = gr.Button("开始总结视频")
                        video_summary_output_text = gr.Markdown(label="总结结果")
                        video_summary_output_audio = gr.Audio(label="总结语音", interactive=False, autoplay=True)
                        video_summary_status_text = gr.Textbox(label="状态", interactive=False, visible=False)

                        summarize_video_button.click(
                            fn=summarize_video_url,
                            inputs=[
                                video_url_input,
                                ollama_model_dropdown_video,
                                tts_voice_dropdown_video
                            ],
                            outputs=[
                                video_summary_output_text,
                                video_summary_output_audio,
                                video_summary_status_text
                            ]
                        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## 视频信息")
                    video_path_info = gr.Textbox(label="视频路径")
                    info_btn = gr.Button("获取信息")
                    info_output = gr.Markdown()
                    info_btn.click(get_video_info, inputs=video_path_info, outputs=info_output)

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## 音频提取")
                    video_path_extract = gr.Textbox(label="视频路径")
                    extract_btn = gr.Button("提取音频")
                    extract_output = gr.Textbox(label="结果")
                    extract_btn.click(extract_audio, inputs=video_path_extract, outputs=extract_output)

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## 视频格式转换")
                    video_path_convert = gr.Textbox(label="视频路径")
                    output_format = gr.Textbox(label="输出格式")
                    convert_btn = gr.Button("转换视频")
                    convert_output = gr.Textbox(label="结果")
                    convert_btn.click(convert_video, inputs=[video_path_convert, output_format], outputs=convert_output)

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## 视频截取")
                    video_path_trim = gr.Textbox(label="视频路径")
                    start_time_video = gr.Number(label="起始时间(秒)", precision=1)
                    end_time_video = gr.Number(label="结束时间(秒)", precision=1)
                    trim_video_btn = gr.Button("截取视频")
                    trim_video_output = gr.Textbox(label="结果")
                    trim_video_btn.click(trim_video, inputs=[video_path_trim, start_time_video, end_time_video], outputs=trim_video_output)

    with gr.TabItem("音频工具"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## 文字转语音")
                    text_input = gr.Textbox(label="文字", lines=5)
                    voice_select = gr.Dropdown(
                        label="选择音色",
                        choices=[
                            "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
                            "zm_yunxia", "zm_yunjian", "zm_yunxi", "zm_yunyang",
                            "zf_001", "zf_002", "zf_003", "zf_004", "zf_005", "zf_006",
                            "zf_007", "zf_008", "zf_017", "zf_018", "zf_019", "zf_021",
                            "zf_022", "zf_023", "zf_024", "zf_026", "zf_027", "zf_028",
                            "zf_032", "zf_036", "zf_038", "zf_039", "zf_040", "zf_042",
                            "zf_043", "zf_044", "zf_046", "zf_047", "zf_048", "zf_049",
                            "zf_051", "zf_059", "zf_060", "zf_067", "zf_070", "zf_071",
                            "zf_072", "zf_073", "zf_074", "zf_075", "zf_076", "zf_077",
                            "zf_078", "zf_079", "zf_083", "zf_084", "zf_085", "zf_086",
                            "zf_087", "zf_088", "zf_090", "zf_092", "zf_093", "zf_094",
                            "zf_099", "zm_009", "zm_010", "zm_011", "zm_012", "zm_013",
                            "zm_014", "zm_015", "zm_016", "zm_020", "zm_025", "zm_029",
                            "zm_030", "zm_031", "zm_033", "zm_034", "zm_035", "zm_037",
                            "zm_041", "zm_045", "zm_050", "zm_052", "zm_053", "zm_054",
                            "zm_055", "zm_056", "zm_057", "zm_058", "zm_061", "zm_062",
                            "zm_063", "zm_064", "zm_065", "zm_066", "zm_068", "zm_069",
                            "zm_080", "zm_081", "zm_082", "zm_089", "zm_091", "zm_095",
                            "zm_096", "zm_097", "zm_098", "zm_100"
                        ],
                        value="zf_xiaoxiao"
                    )
                    tts_btn = gr.Button("生成语音")
                    tts_output = gr.Audio(label="语音输出")
                    tts_btn.click(text_to_speech, inputs=[text_input, voice_select], outputs=tts_output)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## 音频转文字")
                    audio_path_whisper = gr.Textbox(label="音频路径")
                    whisper_btn = gr.Button("开始转换")
                    whisper_output = gr.Textbox(label="结果")
                    whisper_btn.click(audio_to_text, inputs=audio_path_whisper, outputs=whisper_output)

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## 音频转文字 (Faster Whisper)")
                    audio_path_faster = gr.Textbox(label="音频路径")
                    faster_btn = gr.Button("开始转换")
                    faster_output = gr.Textbox(label="结果")
                    faster_btn.click(faster_whisper, inputs=audio_path_faster, outputs=faster_output)

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## 音频转文字 (SenseVoiceSmall)")
                    audio_path_sense = gr.Textbox(label="音频路径")
                    sense_btn = gr.Button("开始转换")
                    sense_output = gr.Textbox(label="结果")
                    sense_btn.click(sense_voice, inputs=audio_path_sense, outputs=sense_output)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## 音频截取")
                    audio_path_trim = gr.Textbox(label="音频路径")
                    start_time_audio = gr.Number(label="起始时间(秒)", precision=1)
                    end_time_audio = gr.Number(label="结束时间(秒)", precision=1)
                    trim_audio_btn = gr.Button("截取音频")
                    trim_audio_output = gr.Textbox(label="结果")
                    trim_audio_btn.click(trim_audio, inputs=[audio_path_trim, start_time_audio, end_time_audio], outputs=trim_audio_output)

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## SRT文件转音频")
                    srt_path = gr.Textbox(label="SRT文件路径")
                    srt_btn = gr.Button("生成音频")
                    srt_output = gr.Audio(label="音频输出")
                    srt_btn.click(srt_to_audio, inputs=srt_path, outputs=srt_output)

    with gr.TabItem("大模型工具"):
        with gr.Row():
            # 自然语言查数据库模块
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## 自然语言查数据库")
                    sql_model_select = gr.Dropdown(
                        label="选择模型",
                        choices=get_all_models(),
                        value="qwen3:14b" if model_choices else None,
                        interactive=True
                    )
                    sql_input = gr.Textbox(
                        label="输入查询",
                        placeholder="输入自然语言查询，按回车或点击发送",
                        elem_id="sql_input"
                    )
                    sql_btn = gr.Button("发送")
                    sql_output = gr.Markdown(label="查询结果")

                    def update_sql(query, model):
                        logger.info(f"处理查询: {query}, 模型: {model}")
                        if not query.strip():
                            logger.warning("空查询")
                            return "**错误: 请输入有效查询**", query
                        success, result = nl_to_sql(query, model)
                        logger.info(f"nl_to_sql 返回: success={success}, result={result}")
                        if success:
                            return result, query
                        return result, query

                    sql_input.submit(
                        update_sql,
                        inputs=[sql_input, sql_model_select],
                        outputs=[sql_output, sql_input]
                    )
                    sql_btn.click(
                        update_sql,
                        inputs=[sql_input, sql_model_select],
                        outputs=[sql_output, sql_input]
                    )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## 大语言模型音频聊天")
                    # ---- START: Place dropdowns in a new Row ----
                    with gr.Row():
                        model_select = gr.Dropdown(label="选择模型", choices=model_choices, value="qwen3:14b" if model_choices else None, interactive=True)
                        chat_voice_select = gr.Dropdown(
                            label="选择音色",
                            choices=[
                                "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
                                "zm_yunxia", "zm_yunjian", "zm_yunxi", "zm_yunyang",
                                "zf_001", "zf_002", "zf_003", "zf_004", "zf_005", "zf_006",
                                "zf_007", "zf_008", "zf_017", "zf_018", "zf_019", "zf_021",
                                "zf_022", "zf_023", "zf_024", "zf_026", "zf_027", "zf_028",
                                "zf_032", "zf_036", "zf_038", "zf_039", "zf_040", "zf_042",
                                "zf_043", "zf_044", "zf_046", "zf_047", "zf_048", "zf_049",
                                "zf_051", "zf_059", "zf_060", "zf_067", "zf_070", "zf_071",
                                "zf_072", "zf_073", "zf_074", "zf_075", "zf_076", "zf_077",
                                "zf_078", "zf_079", "zf_083", "zf_084", "zf_085", "zf_086",
                                "zf_087", "zf_088", "zf_090", "zf_092", "zf_093", "zf_094",
                                "zf_099", "zm_009", "zm_010", "zm_011", "zm_012", "zm_013",
                                "zm_014", "zm_015", "zm_016", "zm_020", "zm_025", "zm_029",
                                "zm_030", "zm_031", "zm_033", "zm_034", "zm_035", "zm_037",
                                "zm_041", "zm_045", "zm_050", "zm_052", "zm_053", "zm_054",
                                "zm_055", "zm_056", "zm_057", "zm_058", "zm_061", "zm_062",
                                "zm_063", "zm_064", "zm_065", "zm_066", "zm_068", "zm_069",
                                "zm_080", "zm_081", "zm_082", "zm_089", "zm_091", "zm_095",
                                "zm_096", "zm_097", "zm_098", "zm_100"
                            ],
                            value="zf_001"
                        )
                    # ---- END: Place dropdowns in a new Row ----
                    chatbot = gr.Chatbot(label="聊天历史")
                    chat_input = gr.Textbox(label="输入文字", placeholder="输入消息，按回车或点击发送")
                    chat_btn = gr.Button("发送")
                    chat_audio = gr.Audio(label="语音回复", autoplay=True)  # 启用自动播放
                    chat_error = gr.Textbox(label="错误信息", visible=False)

                    # 包装chat_with_ollama，添加清空输入框
                    def update_chat(message, model, voice, history):
                        new_history, audio, error = chat_with_ollama(message, model, voice, history)
                        return new_history, audio, error, ""  # 清空输入框

                    chat_input.submit(
                        update_chat,
                        inputs=[chat_input, model_select, chat_voice_select, chatbot],
                        outputs=[chatbot, chat_audio, chat_error, chat_input]
                    )
                    chat_btn.click(
                        update_chat,
                        inputs=[chat_input, model_select, chat_voice_select, chatbot],
                        outputs=[chatbot, chat_audio, chat_error, chat_input]
                    )

    with gr.TabItem("实时视频分析"):
        analysis_history = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(sources=["webcam"], type="numpy", label="Webcam Input",  mirror_webcam=False)
            with gr.Column(scale=1):
                status_message = gr.HTML(
                    label="分析结果"
                )
        with gr.Row():
            model_select = gr.Dropdown(label="选择模型", choices=model_choices, allow_custom_value=False, value="qwen2.5vl:3b-q8_0" if model_choices else None, interactive=True)
        dep = input_img.stream(
            fn=analyze_videos, 
            inputs=[input_img, model_select, analysis_history],
            outputs=[analysis_history, status_message],
            stream_every=0.1,     # 发送频率:值越小，请求越频繁，待上一次结果返回后,发起下一次请求.(参数必须是stream_every)
            concurrency_limit=1, # 限制并发数为1，逐帧处理。
            time_limit=0.8    # 后端最长处理时间
        )
# 启动Gradio应用
demo.launch(server_port=7860, inbrowser=True, mcp_server=True)
