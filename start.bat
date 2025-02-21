@chcp 65001 >nul
@echo off

:: 检查虚拟环境是否存在
if not exist "venv\Scripts\activate" (
    echo 创建虚拟环境...
    python -m venv venv
)

:: 激活虚拟环境
call venv\Scripts\activate

:: 检查 requirements.txt 是否存在并安装依赖
if exist "requirements.txt" (
    echo 检查依赖项...
    pip install -r requirements.txt
) else (
    echo requirements.txt 文件不存在，跳过依赖安装。
)

:: 运行应用
echo 启动应用...
python app.py