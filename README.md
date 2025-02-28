# ffmpeg-simple-web

## 使用

### 1.执行以下命令
```
git clone git@github.com:Djooxx/ffmpeg-simple-web.git
cd ffmpeg-simple-web
.\start.bat
```
#### start.bat将自动执行以下命令
```
# 创建虚拟环境
python -m venv venv
# 激活虚拟环境
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### 2.访问:
```
http://localhost:5000/
```

## 功能
### 1.显示视频信息
### 2.截取视频
### 3.提取视频的音频
### 4.截取音频
### 5.语音转文字(whisper)
### 6.语音转文字(SenseVoiceSmall)
### 6.文字转语音(kokoro)

## 备注:
无法访问huggingface时,kokoro会进行连接重试,非常耗时,直接设置host
```
127.0.0.1 huggingface.co
```
![image](https://github.com/user-attachments/assets/15455295-37b3-4ef4-a375-239dc560302f)
