import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
import asyncio
import json
import secrets # 用于 msg_id
from pathlib import Path
import tempfile # 用于临时音频文件
import wave # 用于写入wav元数据

import gradio as gr
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastrtc import (
    AdditionalOutputs,
    AsyncStreamHandler,
    Stream,
    wait_for_item,
)

import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import soundfile as sf # 用于写入wav文件
import re

# VAD 相关导入
import webrtcvad

# Ollama 相关导入
import ollama
# 全局Ollama客户端实例
ollama_client = ollama.Client(host='http://127.0.0.1:11434')

cur_dir = Path(__file__).parent


class LocalSpeechHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            input_sample_rate=16_000, # SenseVoice 和 VAD 通常使用 16kHz
            # output_sample_rate=24_000, # 不需要音频输出了
        )
        self.output_queue = asyncio.Queue()
        self.vad = webrtcvad.Vad(3)  # VAD 级别 0-3, 3最激进(最容易判断为语音)
        
        # 音频缓冲相关
        self.audio_buffer = bytearray()
        self.is_speaking = False
        self.frames_since_last_speech = 0
        self.min_silence_frames = int(0.8 * (self.input_sample_rate / (160 * 2))) # 0.8秒的静音 (假设VAD帧为20ms)
                                                                                # VAD帧长10, 20, 30ms. 16000Hz, 16bit -> 20ms = 16000*0.02*2 = 640 bytes
                                                                                # (160*2) 是假设的fastrtc每次给的块是160个样本点
        self.vad_frame_duration_ms = 20 # VAD 期望的帧时长 (10, 20, or 30 ms)
        self.vad_frame_bytes = int(self.input_sample_rate * (self.vad_frame_duration_ms / 1000.0) * 2) # 16-bit PCM

        # 初始化 SenseVoice 模型
        model_dir = "iic/SenseVoiceSmall" # 确保模型路径正确
        print(f"Loading SenseVoice model from: {model_dir}")
        try:
            self.stt_model = AutoModel(
                model=model_dir,
                trust_remote_code=True,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                disable_update=True
            )
            print(f"SenseVoice model loaded on {'cuda' if torch.cuda.is_available() else 'cpu'}.")
        except Exception as e:
            print(f"Error loading SenseVoice model: {e}")
            self.stt_model = None


        # 初始化 Ollama 客户端
        self.ollama_model_name = "qwen3:4b"
        print(f"Initializing Ollama client with model: {self.ollama_model_name}")
        try:
            # 检查ollama服务是否可达以及模型是否存在
            ollama_client.list() # 会抛出异常如果服务不可达
            found = False
            for model_info in ollama_client.list()['models']:
                if self.ollama_model_name in model_info['model']:
                    found = True
                    break
            if not found:
                print(f"Ollama model '{self.ollama_model_name}' not found. Please pull it first: `ollama pull {self.ollama_model_name}`")
                self.ollama_client = None
            else:
                self.ollama_client = ollama.AsyncClient(host='http://127.0.0.1:11434') # 使用异步客户端
                print("Ollama client initialized.")
        except Exception as e:
            print(f"Error initializing Ollama or model not found: {e}")
            print(f"Please ensure Ollama is running and you have pulled the model: `ollama pull {self.ollama_model_name}`")
            self.ollama_client = None
            
        self.conversation_history = []


    def copy(self):
        return LocalSpeechHandler() # fastrtc 需要这个

    @staticmethod
    def msg_id() -> str:
        return f"event_{secrets.token_hex(10)}"

    async def start_up(self):
        # 本地处理，不需要连接外部服务
        print("LocalSpeechHandler started up for a new connection.")
        # 清空历史记录，为新会话做准备
        self.conversation_history = []
        self.audio_buffer = bytearray()
        self.is_speaking = False
        self.frames_since_last_speech = 0
        await self.output_queue.put(
            AdditionalOutputs({"role": "assistant", "content": "你好！请开始说话。"})
        )


    def _is_speech_frame(self, frame_data: bytes) -> bool:
        try:
            return self.vad.is_speech(frame_data, self.input_sample_rate)
        except Exception as e:
            # print(f"VAD error: {e}") # 可能因为帧长度不对
            return False

    async def _process_audio_buffer(self):
        if not self.audio_buffer or not self.stt_model:
            self.audio_buffer = bytearray() # 清空buffer
            return

        print(f"Processing audio buffer of length: {len(self.audio_buffer)} bytes")
        
        try:
            # 使用临时文件来保存音频数据
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
                audio_path = tmp_audio_file.name
                # soundfile 需要 numpy 数组
                audio_np = np.frombuffer(bytes(self.audio_buffer), dtype=np.int16)
                sf.write(tmp_audio_file, audio_np, self.input_sample_rate, format='WAV', subtype='PCM_16')
            
            self.audio_buffer = bytearray() # 清空buffer

            print(f"Audio saved to temporary file: {audio_path}")
            result = self.stt_model.generate(
                input=audio_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )

            os.remove(audio_path) # 删除临时文件

            if result and result[0]["text"]:
                transcribed_text = rich_transcription_postprocess(result[0]["text"])
                print(f"User STT: {transcribed_text}")
                if not transcribed_text.strip(): # 忽略空文本
                    return

                await self.output_queue.put(
                    AdditionalOutputs({"role": "user", "content": transcribed_text})
                )
                self.conversation_history.append({"role": "user", "content": transcribed_text})

                # 调用 Ollama LLM
                if self.ollama_client:
                    try:
                        print(f"Sending to Ollama: {self.conversation_history}")
                        llm_response = await self.ollama_client.chat(
                            model=self.ollama_model_name,
                            messages=self.conversation_history
                        )
                        assistant_text = llm_response['message']['content']
                        assistant_reply = re.sub(r'(?i)<think\s*[^>]*>[\s\S]*?</think\s*>', '', assistant_text).strip()
                        print(f"Ollama Assistant: {assistant_reply}")
                        self.conversation_history.append({"role": "assistant", "content": assistant_reply})
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": assistant_reply})
                        )
                    except Exception as e:
                        print(f"Ollama inference error: {e}")
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": f"Ollama 模型出错: {e}"})
                        )
                else:
                     await self.output_queue.put(
                        AdditionalOutputs({"role": "assistant", "content": "Ollama 客户端未初始化。"})
                    )
            else:
                print("STT result is empty or invalid.")

        except Exception as e:
            print(f"Error in _process_audio_buffer: {e}")
            self.audio_buffer = bytearray() # 确保清空
            # 可以在这里向用户发送错误信息
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": f"语音处理出错: {e}"})
            )
        finally:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                 os.remove(audio_path)


    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.stt_model or not self.ollama_client : # 如果模型未加载，则不处理
            if not self.is_speaking: # 避免重复发送
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": "模型未成功加载，无法处理语音。"})
                )
                self.is_speaking = True # 标记一下，避免刷屏
            return
        
        # frame[0] 是采样率, frame[1] 是 numpy 数组 (int16)
        audio_data_np = frame[1].squeeze() # 移除多余维度
        audio_bytes = audio_data_np.tobytes()

        # 将接收到的音频块分割成VAD期望的帧大小进行处理
        num_frames = len(audio_bytes) // self.vad_frame_bytes
        has_speech_in_this_block = False

        for i in range(num_frames):
            start = i * self.vad_frame_bytes
            end = start + self.vad_frame_bytes
            vad_audio_chunk = audio_bytes[start:end]

            if len(vad_audio_chunk) < self.vad_frame_bytes: # 不足一个VAD帧，跳过
                continue

            if self._is_speech_frame(vad_audio_chunk):
                # print("Speech detected in VAD frame.")
                self.audio_buffer.extend(vad_audio_chunk)
                self.is_speaking = True
                self.frames_since_last_speech = 0
                has_speech_in_this_block = True
            elif self.is_speaking: # 当前块没有语音，但之前有语音
                self.audio_buffer.extend(vad_audio_chunk) # 仍然缓冲一小段尾音
                self.frames_since_last_speech += 1
                # print(f"Silence after speech, count: {self.frames_since_last_speech}")

        # 如果整个块都没有语音，但之前有语音，也增加静音计数
        if not has_speech_in_this_block and self.is_speaking:
            self.frames_since_last_speech += num_frames # 近似增加静音帧数

        # 如果检测到足够长的静音，并且之前有语音，则处理缓冲的音频
        if self.is_speaking and self.frames_since_last_speech >= self.min_silence_frames:
            print("End of speech detected by VAD silence.")
            await self._process_audio_buffer()
            self.is_speaking = False
            self.frames_since_last_speech = 0
            # audio_buffer 已经在 _process_audio_buffer 中清空

        # 防止buffer过长，设置一个最大长度（例如15秒）
        max_buffer_bytes = self.input_sample_rate * 2 * 15 # 15 seconds
        if len(self.audio_buffer) > max_buffer_bytes:
            print("Max audio buffer length reached, processing.")
            await self._process_audio_buffer()
            self.is_speaking = False # 重置状态
            self.frames_since_last_speech = 0


    async def emit(self) -> AdditionalOutputs | None: # 只返回 AdditionalOutputs
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        print("LocalSpeechHandler shutting down.")
        # 清空队列
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        # 清理其他资源 (模型通常由Python GC处理，但可以显式del)
        if hasattr(self, 'stt_model'): del self.stt_model

        if hasattr(self, 'ollama_client') and self.ollama_client:
            self.ollama_client = None # 显式解除引用，帮助GC

        if len(self.audio_buffer) > 0 and self.is_speaking: # 如果关闭时仍有未处理的音频
            print("Processing remaining audio buffer on shutdown...")
            await self._process_audio_buffer()


def update_chatbot(chatbot: list[dict], response: dict):
    chatbot.append(response)
    return chatbot


chatbot_ui = gr.Chatbot(type="messages", label="本地语音助手")

rtc_config = None

stream = Stream(
    LocalSpeechHandler(),
    mode="send-receive", # 仍然需要接收音频
    modality="audio",    # 输入是音频
    additional_inputs=[], # voice_dropdown 移除了, chatbot本身不作为input给handler的start_up
    additional_outputs=[chatbot_ui], # 输出到聊天框
    additional_outputs_handler=update_chatbot,
    rtc_configuration=rtc_config,
)

app = FastAPI()
stream.mount(app) # fastrtc 会处理 WebRTC 信令和 Gradio UI

@app.get("/outputs") # 这个端点用于非Gradio客户端订阅聊天输出，可以保留
def _(webrtc_id: str):
    async def output_stream():
        import json
        async for output in stream.output_stream(webrtc_id): # output.args[0] 是聊天消息dict
            s = json.dumps(output.args[0])
            yield f"event: output\ndata: {s}\n\n"
    return StreamingResponse(output_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    print("Starting Gradio UI and FastAPI server...")
    stream.ui.launch(server_name="0.0.0.0", server_port=7860)