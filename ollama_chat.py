from flask import Blueprint, request, jsonify, send_file
import requests
import json
import time
import os
import numpy as np
import torch
import traceback
from io import BytesIO
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from kokoro import KModel, KPipeline
import re
import tqdm

# 创建蓝图
ollama_chat_bp = Blueprint('ollama_chat', __name__)

# 设置Ollama API地址
OLLAMA_API_URL = "http://localhost:11434/api"


# 复用app.py中的语音识别模型
sense_voice_model = None

def get_sense_voice_model():
    global sense_voice_model
    if sense_voice_model is None:
        model_dir = "iic/SenseVoiceSmall"
        sense_voice_model = AutoModel(
            model=model_dir,
            trust_remote_code=True, 
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
    return sense_voice_model

# 获取Ollama可用模型列表
@ollama_chat_bp.route('/get_ollama_models', methods=['GET'])
def get_ollama_models():
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            # 提取模型名称列表
            model_names = [model.get('name') for model in models]
            return jsonify({'success': True, 'models': model_names})
        else:
            return jsonify({'success': False, 'error': f'获取模型列表失败: {response.text}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# 音频转文字 (使用SenseVoiceSmall)
@ollama_chat_bp.route('/audio_to_text_for_chat', methods=['POST'])
def audio_to_text_for_chat():
    try:
        # 检查是否有音频文件上传
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': '没有上传音频文件'})
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': '没有选择音频文件'})
        
        # 保存上传的音频文件
        timestamp = int(time.time())
        temp_audio_path = f"temp_audio_{timestamp}.wav"
        audio_file.save(temp_audio_path)
        
        # 使用SenseVoiceSmall模型进行语音识别
        model = get_sense_voice_model()
        res = model.generate(
            input=temp_audio_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        
        # 处理识别结果
        text = rich_transcription_postprocess(res[0]["text"])
        
        # 删除临时音频文件
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        return jsonify({'success': True, 'text': text})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

# 与Ollama模型聊天
@ollama_chat_bp.route('/chat_with_ollama', methods=['POST'])
def chat_with_ollama():
    try:
        data = request.json
        model_name = data.get('model', 'llama3')
        user_message = data.get('message', '')
        
        # 构建请求数据
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": user_message}
            ],
            "stream": False
        }
        
        # 发送请求到Ollama API
        response = requests.post(f"{OLLAMA_API_URL}/chat", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            assistant_message = result.get('message', {}).get('content', '')
            return jsonify({'success': True, 'response': assistant_message})
        else:
            return jsonify({'success': False, 'error': f'模型响应失败: {response.text}'})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

# 文字转语音 (使用kokoro)
@ollama_chat_bp.route('/text_to_speech_for_chat', methods=['POST'])
def text_to_speech_for_chat():
    try:
        # 在函数内部导入generate_audio_data，避免循环导入
        from app import generate_audio_data
        
        data = request.json
        text = data.get('text', '')
        voice = data.get('voice', 'zf_xiaoxiao')
        
        wavs = generate_audio_data(text, voice)
        if not wavs:
            return jsonify({'success': False, 'error': '没有生成音频'})
        
        # 合并音频片段
        audio_data = np.concatenate(wavs)
        
        # 返回音频数据
        timestamp = int(time.time())
        filename = f'audio_chat_{voice}_{timestamp}.wav'
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