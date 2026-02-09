# ffmpeg-simple-web

## 使用
1.执行命令
```
gradio_web.bat
```
2.访问
http://127.0.0.1:7860

## 功能
### 1.显示视频信息
### 2.截取视频
### 3.提取视频的音频
### 4.截取音频
### 5.视频格式转换
### 6.语音转文字(whisper)
### 7.语音转文字(faster-whisper)
### 8.语音转文字(SenseVoiceSmall)
### 9.文字转语音(kokoro)
### 10.SRT文件转音频

## 备注:
无法访问huggingface时,kokoro会进行连接重试,非常耗时,直接设置hosts
```
127.0.0.1 huggingface.co
```