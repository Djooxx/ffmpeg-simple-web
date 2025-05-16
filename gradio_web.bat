@chcp 65001 >nul
@echo off

:: 激活虚拟环境
call venv\Scripts\activate

python app_gradio.py