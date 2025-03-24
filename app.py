from flask import Flask, render_template, request, jsonify, send_file
import subprocess
import numpy as np
import torch
import time
import math
import json
import os
import whisper
from faster_whisper import WhisperModel
import re
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
import soundfile as sf
from kokoro import KModel,KPipeline
import tqdm
import traceback

def convert_size(size_bytes):
    """将字节数转换为合适的单位"""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_video_info', methods=['POST'])
def get_video_info():
    video_path = request.form['video_path']
    if not os.path.exists(video_path):
        return jsonify({'success': False, 'error': '文件路径不存在'})
    if not os.path.isfile(video_path):
        return jsonify({'success': False, 'error': '路径不是文件'})
    try:
        # 使用ffprobe获取视频信息，指定获取第一个视频流
        # 使用列表参数形式避免shell特殊字符问题
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 
            'format=duration,size,bit_rate:stream=width,height,bit_rate,codec_name,avg_frame_rate,r_frame_rate',
            '-of', 'json',
            video_path  # 直接使用原始路径
        ]
        result = subprocess.run(cmd, shell=False, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            # 解析ffprobe输出
            try:
                video_info = json.loads(result.stdout)
                # 转换文件大小
                if 'format' in video_info and 'size' in video_info['format']:
                    size_bytes = int(video_info['format']['size'])
                    video_info['format']['size'] = convert_size(size_bytes)
                return jsonify({'success': True, 'data': video_info})
            except json.JSONDecodeError as e:
                print("JSON解析错误:", e)
                return jsonify({'success': False, 'error': f"JSON解析失败: {str(e)}"})
        else:
            return jsonify({'success': False, 'error': result.stderr})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/extract_audio', methods=['POST'])
def extract_audio():
    video_path = request.form['video_path']
    if not os.path.exists(video_path):
        return jsonify({'success': False, 'error': '文件路径不存在'})
    if not os.path.isfile(video_path):
        return jsonify({'success': False, 'error': '路径不是文件'})
    try:
        timestamp = int(time.time())
        # 获取输入文件扩展名
        input_ext = video_path.split('.')[-1].lower()
        # 支持常见视频格式
        supported_formats = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']
        if input_ext not in supported_formats:
            return jsonify({'success': False, 'error': f'不支持的文件格式: {input_ext}'})
        
        # 根据输入格式选择输出格式
        output_ext = 'mp3'  # 默认输出mp3
        if input_ext in ['wav', 'flac']:  # 如果是无损格式，保持原格式
            output_ext = input_ext
            
        output_path = video_path.rsplit('.', 1)[0] + f'_{timestamp}.{output_ext}'
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-map', '0:a:0',  # 选择第一个音频流
            '-c:a', 'libmp3lame',  # 明确指定MP3编码器
            '-q:a', '0',
            '-ac', '2',  # 确保立体声输出
            output_path
        ]
        result = subprocess.run(cmd, shell=False, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            return jsonify({'success': True, 'output_path': output_path})
        else:
            return jsonify({'success': False, 'error': result.stderr})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/trim_audio', methods=['POST'])
def trim_audio():
    print("收到截取音频请求")
    audio_path = request.form['audio_path']
    if not os.path.exists(audio_path):
        return jsonify({'success': False, 'error': '文件路径不存在'})
    if not os.path.isfile(audio_path):
        return jsonify({'success': False, 'error': '路径不是文件'})
    start_time = request.form['start_time']
    end_time = request.form['end_time']
    print(f"参数解析成功 - 音频路径: {audio_path}, 开始时间: {start_time}, 结束时间: {end_time}")
    try:
        # 参数校验
        start_time = float(start_time)
        end_time = float(end_time)
        if start_time < 0 or end_time < 0:
            return jsonify({'success': False, 'error': '起始时间和结束时间必须大于0'})
        if start_time >= end_time:
            return jsonify({'success': False, 'error': '结束时间必须大于起始时间'})
        timestamp = int(time.time())
        # 获取输入文件扩展名
        input_ext = audio_path.split('.')[-1].lower()
        # 支持常见音频格式
        supported_formats = ['mp3', 'wav', 'flac', 'aac', 'ogg']
        if input_ext not in supported_formats:
            return jsonify({'success': False, 'error': f'不支持的音频格式: {input_ext}'})
            
        output_path = audio_path.rsplit('.', 1)[0] + f'_{start_time}s-{end_time}s_{timestamp}.{input_ext}'
        duration = float(end_time) - float(start_time)
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', audio_path,
            '-t', str(duration),
            '-c', 'copy',
            output_path
        ]
        print(f"准备执行命令: {cmd}")
        result = subprocess.run(cmd, shell=False, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        print(f"命令执行完成，返回码: {result.returncode}")
        if result.returncode == 0:
            print(f"音频截取成功，保存路径: {output_path}")
            return jsonify({'success': True, 'output_path': output_path})
        else:
            print(f"音频截取失败，错误信息: {result.stderr}")
            return jsonify({'success': False, 'error': result.stderr})
    except Exception as e:
        print(f"发生异常: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/trim_video', methods=['POST'])
def trim_video():
    print("收到截取视频请求")
    video_path = request.form['video_path']
    if not os.path.exists(video_path):
        return jsonify({'success': False, 'error': '文件路径不存在'})
    if not os.path.isfile(video_path):
        return jsonify({'success': False, 'error': '路径不是文件'})
    start_time = request.form['start_time']
    end_time = request.form['end_time']
    print(f"参数解析成功 - 视频路径: {video_path}, 开始时间: {start_time}, 结束时间: {end_time}")
    try:
        # 参数校验
        start_time = float(start_time)
        end_time = float(end_time)
        if start_time < 0 or end_time < 0:
            return jsonify({'success': False, 'error': '起始时间和结束时间必须大于0'})
        if start_time >= end_time:
            return jsonify({'success': False, 'error': '结束时间必须大于起始时间'})
        timestamp = int(time.time())
        # 获取输入文件扩展名
        input_ext = video_path.split('.')[-1].lower()
        # 支持常见视频格式
        supported_formats = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']
        if input_ext not in supported_formats:
            return jsonify({'success': False, 'error': f'不支持的视频格式: {input_ext}'})
            
        output_path = video_path.rsplit('.', 1)[0] + f'_{start_time}s-{end_time}s_{timestamp}.{input_ext}'
        duration = float(end_time) - float(start_time)
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),   # 初步快速定位（关键帧对齐）
            '-i', video_path,
            '-ss', '0',               # 精确微调（从已定位的位置开始逐帧解码）
            '-t', str(duration),
        ]
        if start_time > 0:
            cmd.extend(['-c:v', 'hevc_nvenc', '-b:v', '5M'])  # 添加硬件编码和比特率设置
        else:
            cmd.extend(['-c:v', 'copy'])
        cmd.extend(['-c:a', 'copy', output_path])
        print(f"准备执行命令: {cmd}")
        result = subprocess.run(cmd, shell=False, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        print(f"命令执行完成，返回码: {result.returncode}")
        if result.returncode == 0:
            print(f"视频截取成功，保存路径: {output_path}")
            return jsonify({'success': True, 'output_path': output_path})
        else:
            print(f"视频截取失败，错误信息: {result.stderr}")
            return jsonify({'success': False, 'error': result.stderr})
    except Exception as e:
        print(f"发生异常: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/convert_video', methods=['POST'])
def convert_video():
    video_path = request.form['video_path']
    if not os.path.exists(video_path):
        return jsonify({'success': False, 'error': '文件路径不存在'})
    if not os.path.isfile(video_path):
        return jsonify({'success': False, 'error': '路径不是文件'})
    output_format = request.form['output_format']
    try:
        timestamp = int(time.time())
        output_path = video_path.rsplit('.', 1)[0] + f'_{timestamp}.{output_format}'
        cmd = [
            'ffmpeg',
            '-i', video_path,
            output_path
        ]
        result = subprocess.run(cmd, shell=False, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            return jsonify({'success': True, 'output_path': output_path})
        else:
            return jsonify({'success': False, 'error': result.stderr})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/audio_to_text', methods=['POST'])
def audio_to_text():
    audio_path = request.form['audio_path']
    if not os.path.exists(audio_path):
        return jsonify({'success': False, 'error': '文件路径不存在'})
    if not os.path.isfile(audio_path):
        return jsonify({'success': False, 'error': '路径不是文件'})
    try:
        model = whisper.load_model("large")
        result = model.transcribe(audio_path, language="Chinese")
        segments = result["segments"]
        print("segments")
        print(segments)
        # 生成 SRT 内容
        srt_content = ""
        for i, segment in enumerate(segments):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip() # 去除首尾空白
            srt_content += f"{i+1}\n"
            srt_content += f"{format_timestamp(start, always_include_hours=True)} --> {format_timestamp(end, always_include_hours=True)}\n"
            srt_content += f"{text}\n\n"

        # 保存 SRT 文件
        timestamp = int(time.time())
        srt_path = audio_path.rsplit('.', 1)[0] + f'_{timestamp}' +'.srt'
        with open(srt_path, 'w', encoding='utf-8') as f: # 指定 utf-8 编码
            f.write(srt_content)

        return jsonify({'success': True, 'text': "音频转文字成功，并保存为 srt 文件: " + srt_path}) # 修改返回信息
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/faster_whisper', methods=['POST'])
def faster_whisper():
    audio_path = request.form['audio_path']
    if not os.path.exists(audio_path):
        return jsonify({'success': False, 'error': '文件路径不存在'})
    if not os.path.isfile(audio_path):
        return jsonify({'success': False, 'error': '路径不是文件'})
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        compute_type = 'int8_float16' if device == 'cuda' else 'int8'
        model = WhisperModel("large-v3", device=device, compute_type=compute_type)
        
        segments, info = model.transcribe(audio_path, beam_size=5, language="zh")
        
        srt_content = ""
        for i, segment in enumerate(segments):
            srt_content += f"{i+1}\n"
            srt_content += f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
            srt_content += f"{segment.text.strip()}\n\n"
        
        timestamp = int(time.time())
        srt_path = audio_path.rsplit('.', 1)[0] + f'_faster_whisper_{timestamp}.srt'
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        return jsonify({'success': True, 'text': "转换成功", 'srt_path': srt_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def format_timestamp(seconds: float, always_include_hours: bool = False):
    if seconds is not None:
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    else:
        return None

@app.route('/sense_voice', methods=['POST'])
def sense_voice():
    audio_path = request.form['audio_path']
    if not os.path.exists(audio_path):
        return jsonify({'success': False, 'error': '文件路径不存在'})
    if not os.path.isfile(audio_path):
        return jsonify({'success': False, 'error': '路径不是文件'})
    try:
        model_dir = "iic/SenseVoiceSmall"
        model = AutoModel(
            model=model_dir,
            trust_remote_code=True, 
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
        )

        res = model.generate(
            input=audio_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )

        text = rich_transcription_postprocess(res[0]["text"])
        
        # 保存为txt文件
        timestamp = int(time.time())
        txt_path = audio_path.rsplit('.', 1)[0] + f'_sense_voice_{timestamp}.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        return jsonify({'success': True, 'text': text, 'txt_path': txt_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    

en_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', model=False)
def en_callable(text):
    return next(en_pipeline(text)).phonemes
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_v1_1 = KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh').to(device).eval()
model_v1_0 = KModel(repo_id='hexgrad/Kokoro-82M').to(device).eval()
pipeline_v1_1 = KPipeline(repo_id='hexgrad/Kokoro-82M-v1.1-zh', lang_code='z' , en_callable=en_callable) 
pipeline_v1_0 = KPipeline(repo_id='hexgrad/Kokoro-82M', lang_code='z', en_callable=en_callable) 

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    try:
        text = request.form['text']
        voice = request.form['voice']
        wavs = generate_audio_data(text, voice)
        if not wavs:
            return jsonify({'success': False, 'error': 'No audio generated'})
        # Concatenate audio segments
        audio_data = np.concatenate(wavs)
        # 返回音频数据
        timestamp = int(time.time())
        filename = f'audio_{voice}_{timestamp}.wav'
        audio_buffer = BytesIO()
        sf.write(audio_buffer, audio_data, 24000, format='WAV')
        audio_buffer.seek(0)
        return send_file(
            audio_buffer,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

def generate_audio_data(text, voice):
    wavs = []
    try:
        if re.search(r'\d$', voice):
            pipeline = pipeline_v1_1
        else:
            pipeline = pipeline_v1_0
        # 处理text
        # 分割模式
        split_pattern = r'\n+|[。;；!！?？]|…|\.{2,}'
        # 使用re.split()方法根据split_pattern分割文本
        sentences = re.split(split_pattern, text)
        # 创建结果列表，并确保过滤掉任何空字符串或仅包含空白字符的情况
        texts = [(sentence.strip(),) for sentence in sentences if sentence.strip()]
        # 打印结果
        print(texts)
        for paragraph in tqdm.tqdm(texts):
            for i, sentence in enumerate(paragraph):
                print("处理文字: " + sentence)
                generator = pipeline(sentence, voice=voice, speed=speed_callable)
                result = next(generator)
                wav = result.audio
                if i == 0 and wavs:
                    wav = np.concatenate([np.zeros(5000), wav])
                wavs.append(wav)
        return wavs
    except Exception as e:
        print(traceback.format_exc())
        return wavs

def speed_callable(len_ps):
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1

@app.route('/srt_to_audio', methods=['POST'])
def srt_to_audio():
    print("收到SRT转音频请求")
    srt_path = request.form['srt_path']
    print(srt_path+"==测试存在")
    if not os.path.exists(srt_path):
        print(srt_path+"==不存在")
        return jsonify({'success': False, 'error': 'SRT文件路径不存在'})
    if not os.path.isfile(srt_path):
        print(srt_path+"==不是文件")
        return jsonify({'success': False, 'error': 'SRT路径不是文件'})
    print(srt_path+"==存在")
    # 读取SRT文件内容
    with open(srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    print("文本内容=="+srt_content)
    # 分段解析SRT内容
    segments = parse_srt(srt_content)
    print("分段=="+str(segments))
    # 调用text_to_speech方法生成音频
    wavs = []
    for segment in segments:
        text = segment['text']
        voice = 'zf_xiaoxiao'
        wavTemp = generate_audio_data(text=text, voice=voice)
        if wavTemp:
            wavs.append(np.concatenate(wavTemp))
        else:
            return jsonify({'success': False, 'error': '音频生成失败'})
    print("wavs=="+wavs.__len__().__str__())
    # 按时间戳拼接音频
    audio_data = concatenate_audio(wavs, segments)
    print("拼接音频完成")
    # 保存音频文件
    timestamp = int(time.time())
    srt_name = os.path.splitext(os.path.basename(srt_path))[0]
    filename = f'audio_srt_{srt_name}_{timestamp}.wav'
    audio_buffer = BytesIO()
    sf.write(audio_buffer, audio_data, 24000, format='WAV')
    audio_buffer.seek(0)
    return send_file(
        audio_buffer,
        mimetype='audio/wav',
        as_attachment=True,
        download_name=filename
    )


def parse_srt(srt_content):
    segments = []
    lines = srt_content.strip().split('\n\n')
    for line in lines:
        parts = line.split('\n')
        if len(parts) >= 3:
            index = parts[0]
            time_range = parts[1]
            text = '\n'.join(parts[2:])
            start_time, end_time = parse_time_range(time_range)
            segments.append({
                'index': index,
                'start_time': start_time,
                'end_time': end_time,
                'text': text
            })
    return segments


def parse_time_range(time_range):
    start_str, end_str = time_range.split(' --> ')
    start_time = convert_time_to_seconds(start_str)
    end_time = convert_time_to_seconds(end_str)
    return start_time, end_time


def convert_time_to_seconds(time_str):
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_parts = parts[2].split(',')
    seconds = int(seconds_parts[0])
    milliseconds = int(seconds_parts[1])
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return total_seconds


def concatenate_audio(wavs, segments):
    total_duration = segments[-1]['end_time']
    # 计算每个音频段应有的采样点数
    sample_rate = 24000
    audio_data = np.zeros(int(total_duration * sample_rate))
    
    for i, (wav, segment) in enumerate(zip(wavs, segments)):
        start_time = segment['start_time']
        end_time = segment['end_time']
        expected_samples = int((end_time - start_time) * sample_rate)
        
        # 裁剪或填充音频数据到预期长度
        if len(wav) > expected_samples:
            wav = wav[:expected_samples]
        elif len(wav) < expected_samples:
            wav = np.pad(wav, (0, expected_samples - len(wav)), 'constant')
        
        start_index = int(start_time * sample_rate)
        audio_data[start_index:start_index + expected_samples] = wav
    return audio_data

if __name__ == '__main__':
    app.run(debug=False)
