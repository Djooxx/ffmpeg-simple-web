// 用于处理与大语言模型的实时音频聊天功能

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let selectedModel = '';
let chatHistory = [];

// 页面加载时获取可用的Ollama模型列表
document.addEventListener('DOMContentLoaded', function() {
    // 获取模型列表
    fetchOllamaModels();
    
    // 初始化录音按钮事件
    const recordButton = document.getElementById('recordButton');
    if (recordButton) {
        recordButton.addEventListener('click', toggleRecording);
    }
    
    // 初始化发送文本按钮事件
    const sendTextButton = document.getElementById('sendTextButton');
    if (sendTextButton) {
        sendTextButton.addEventListener('click', sendTextMessage);
    }
});

// 获取Ollama可用模型列表
function fetchOllamaModels() {
    fetch('/get_ollama_models')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const modelSelect = document.getElementById('modelSelect');
                // 清空现有选项
                modelSelect.innerHTML = '';
                
                // 添加模型选项
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                });
                
                // 如果有模型，默认选择第一个
                if (data.models.length > 0) {
                    selectedModel = data.models[0];
                }
            } else {
                showError('获取模型列表失败: ' + data.error);
            }
        })
        .catch(error => {
            showError('请求失败: ' + error);
        });
}

// 切换录音状态
function toggleRecording() {
    const recordButton = document.getElementById('recordButton');
    
    if (!isRecording) {
        // 开始录音
        startRecording();
        recordButton.textContent = '停止录音';
        recordButton.classList.add('recording');
    } else {
        // 停止录音
        stopRecording();
        recordButton.textContent = '开始录音';
        recordButton.classList.remove('recording');
    }
    
    isRecording = !isRecording;
}

// 开始录音
function startRecording() {
    audioChunks = [];
    
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = processRecording;
            
            mediaRecorder.start();
        })
        .catch(error => {
            showError('无法访问麦克风: ' + error);
        });
}

// 停止录音
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
}

// 处理录音结果
function processRecording() {
    // 显示处理中的消息
    updateChatHistory('正在处理您的语音...', 'user-processing');
    
    // 创建音频Blob
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    
    // 创建FormData对象
    const formData = new FormData();
    formData.append('audio', audioBlob);
    
    // 发送到服务器进行语音识别
    fetch('/audio_to_text_for_chat', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 移除处理中的消息
            removeTempMessages();
            
            // 显示识别出的文字
            const userText = data.text;
            updateChatHistory(userText, 'user');
            
            // 发送文字到Ollama进行对话
            chatWithOllama(userText);
        } else {
            showError('语音识别失败: ' + data.error);
        }
    })
    .catch(error => {
        showError('请求失败: ' + error);
    });
}

// 发送文本消息
function sendTextMessage() {
    const textInput = document.getElementById('chatTextInput');
    const text = textInput.value.trim();
    
    if (text) {
        // 显示用户消息
        updateChatHistory(text, 'user');
        
        // 清空输入框
        textInput.value = '';
        
        // 发送文字到Ollama进行对话
        chatWithOllama(text);
    }
}

// 与Ollama模型聊天
function chatWithOllama(message) {
    // 获取选中的模型
    const modelSelect = document.getElementById('modelSelect');
    const selectedModel = modelSelect.value;
    
    // 显示处理中的消息
    updateChatHistory('正在思考...', 'assistant-processing');
    
    // 发送请求到服务器
    fetch('/chat_with_ollama', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: selectedModel,
            message: message
        })
    })
    .then(response => response.json())
    .then(data => {
        // 移除处理中的消息
        removeTempMessages();
        
        if (data.success) {
            // 显示模型回复
            const assistantText = data.response;
            updateChatHistory(assistantText, 'assistant');
            
            // 将回复转为语音
            textToSpeech(assistantText);
        } else {
            showError('模型响应失败: ' + data.error);
        }
    })
    .catch(error => {
        showError('请求失败: ' + error);
    });
}

// 文字转语音
function textToSpeech(text) {
    // 获取选中的语音
    const voiceSelect = document.getElementById('chatVoiceSelect');
    const selectedVoice = voiceSelect.value;
    
    // 发送请求到服务器
    fetch('/text_to_speech_for_chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: text,
            voice: selectedVoice
        })
    })
    .then(response => {
        if (response.ok) {
            return response.blob();
        }
        throw new Error('语音生成失败');
    })
    .then(blob => {
        // 创建音频URL并播放
        const audioUrl = URL.createObjectURL(blob);
        const audioPlayer = document.getElementById('chatAudioPlayer');
        const audioControls = document.getElementById('chatAudioControls');
        
        audioPlayer.src = audioUrl;
        audioControls.style.display = 'block';
        audioPlayer.play();
        
        // 设置下载链接
        const downloadLink = document.getElementById('chatDownloadLink');
        downloadLink.href = audioUrl;
        downloadLink.download = 'chat_response.wav';
    })
    .catch(error => {
        showError('语音生成失败: ' + error);
    });
}

// 更新聊天历史
function updateChatHistory(message, role) {
    const chatHistoryDiv = document.getElementById('chatHistory');
    
    // 创建消息元素
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;
    
    // 处理临时消息（正在处理中的消息）
    if (role === 'user-processing' || role === 'assistant-processing') {
        messageDiv.classList.add('processing');
    }
    
    // 设置消息内容
    messageDiv.textContent = message;
    
    // 添加到聊天历史
    chatHistoryDiv.appendChild(messageDiv);
    
    // 滚动到底部
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
    
    // 保存到聊天历史数组（除了处理中的消息）
    if (!role.includes('processing')) {
        chatHistory.push({ role, message });
    }
}

// 移除临时消息
function removeTempMessages() {
    const processingMessages = document.querySelectorAll('.chat-message.processing');
    processingMessages.forEach(msg => msg.remove());
}

// 显示错误消息
function showError(message) {
    const errorDiv = document.getElementById('chatError');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    
    // 3秒后隐藏错误消息
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 3000);
}

// 音频播放器控制
document.addEventListener('DOMContentLoaded', function() {
    const audioPlayer = document.getElementById('chatAudioPlayer');
    const playPauseBtn = document.getElementById('chatPlayPauseBtn');
    const progressBar = document.getElementById('chatProgressBar');
    const timeDisplay = document.getElementById('chatTimeDisplay');
    
    if (audioPlayer && playPauseBtn) {
        // 播放/暂停按钮
        playPauseBtn.addEventListener('click', function() {
            if (audioPlayer.paused) {
                audioPlayer.play();
                playPauseBtn.textContent = '暂停';
            } else {
                audioPlayer.pause();
                playPauseBtn.textContent = '播放';
            }
        });
        
        // 音频播放结束
        audioPlayer.addEventListener('ended', function() {
            playPauseBtn.textContent = '播放';
        });
        
        // 更新进度条
        audioPlayer.addEventListener('timeupdate', function() {
            const currentTime = audioPlayer.currentTime;
            const duration = audioPlayer.duration;
            
            if (!isNaN(duration)) {
                // 更新进度条
                progressBar.value = (currentTime / duration) * 100;
                
                // 更新时间显示
                const currentMinutes = Math.floor(currentTime / 60);
                const currentSeconds = Math.floor(currentTime % 60);
                const durationMinutes = Math.floor(duration / 60);
                const durationSeconds = Math.floor(duration % 60);
                
                timeDisplay.textContent = `${currentMinutes.toString().padStart(2, '0')}:${currentSeconds.toString().padStart(2, '0')} / ${durationMinutes.toString().padStart(2, '0')}:${durationSeconds.toString().padStart(2, '0')}`;
            }
        });
        
        // 进度条点击
        progressBar.addEventListener('input', function() {
            const seekTime = (progressBar.value / 100) * audioPlayer.duration;
            audioPlayer.currentTime = seekTime;
        });
    }
});