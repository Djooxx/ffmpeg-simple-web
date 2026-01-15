@chcp 65001 >nul
@echo off

:: 激活虚拟环境
call venv\Scripts\activate

set HF_ENDPOINT=https://hf-mirror.com

python app_gradio.py