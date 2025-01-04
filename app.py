from flask import Flask, render_template, request, jsonify
import subprocess
import time
import math
import json
import os

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
        result = subprocess.run(cmd, shell=False, capture_output=True, text=True)
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
            '-q:a', '0',
            '-map', 'a',
            output_path
        ]
        result = subprocess.run(cmd, shell=False, capture_output=True, text=True)
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
        result = subprocess.run(cmd, shell=False, capture_output=True, text=True)
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

if __name__ == '__main__':
    app.run(debug=True)
