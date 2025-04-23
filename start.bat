@chcp 65001 >nul
@echo off

:: 检查虚拟环境是否存在
if not exist "venv\Scripts\activate" (
    echo 创建虚拟环境...
    python -m venv venv
)

:: 激活虚拟环境
call venv\Scripts\activate

if exist "requirements.txt" (
    echo 安装基础依赖...
    pip install -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com -r requirements.txt
) else (
    echo requirements.txt 文件不存在，跳过依赖安装。
)

:: 预检NVIDIA显卡是否存在
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo 检测到NVIDIA显卡，安装CUDA版本...
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
) else (
    echo 无NVIDIA显卡，直接安装CPU版本...
    pip install torch==2.3 torchaudio==2.3
)

:: 运行应用
echo 启动应用...
waitress-serve --listen=0.0.0.0:5000 app:app