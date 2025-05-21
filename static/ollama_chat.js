// 用于处理与大语言模型的实时音频聊天功能

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let selectedModel = '';
let chatHistory = []; // 存储对话历史的数组，用于实现对话记忆功能
let audioContext;
let analyser;
let silenceTimer = null;
let isSpeaking = false;
let audioStream = null;
const SILENCE_THRESHOLD = 30; // 静音阈值
const SILENCE_DURATION = 1500; // 静音持续时间（毫秒）

// 页面加载时获取可用的Ollama模型列表
document.addEventListener('DOMContentLoaded', function() {
    // 获取模型列表
    fetchOllamaModels();
    
    // 初始化录音按钮事件
    const recordButton = document.getElementById('recordButton');
    if (recordButton) {
        recordButton.textContent = '语音';
        recordButton.addEventListener('click', toggleRecording);
    }
    
    // 初始化发送文本按钮事件
    const sendTextButton = document.getElementById('sendTextButton');
    if (sendTextButton) {
        sendTextButton.addEventListener('click', sendTextMessage);
    }
    
    // 初始化文本输入框回车事件
    const chatTextInput = document.getElementById('chatTextInput');
    if (chatTextInput) {
        chatTextInput.addEventListener('keydown', function(event) {
            // 检查是否按下了回车键 (keyCode 13) 并且没有按下 Shift 键
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault(); // 阻止默认的回车行为（例如换行）
                sendTextMessage(); // 调用发送消息函数
            }
        });
    }
    
    // 初始化清除聊天按钮事件
    const clearChatButton = document.getElementById('clearChatButton');
    if (clearChatButton) {
        clearChatButton.addEventListener('click', clearChatHistory);
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

// 切换对话状态
function toggleRecording() {
    const recordButton = document.getElementById('recordButton');
    
    if (!isRecording) {
        // 开始对话
        startRecording();
        recordButton.textContent = '结束';
        recordButton.classList.add('recording');
    } else {
        // 结束对话
        stopRecording();
        recordButton.textContent = '语音';
        recordButton.classList.remove('recording');
    }
    
    isRecording = !isRecording;
}

// 开始录音和语音监听
function startRecording() {
    audioChunks = [];
    
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            audioStream = stream;
            
            // 创建音频分析器
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            const microphone = audioContext.createMediaStreamSource(stream);
            microphone.connect(analyser);
            
            // 配置分析器
            analyser.fftSize = 256;
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            
            // 创建录音机但不立即开始录音
            mediaRecorder = new MediaRecorder(stream, {mimeType: 'audio/webm'});
            console.log('已创建MediaRecorder，初始状态:', mediaRecorder.state);
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                    console.log('收到音频数据，当前块数:', audioChunks.length, '大小:', event.data.size, '字节');
                }
            };
            
            mediaRecorder.onstop = () => {
                console.log('MediaRecorder.onstop事件触发，准备处理录音');
                processRecording();
            };
            
            // 不立即开始录音，而是等待检测到声音后再开始
            // 开始监听音量
            detectSpeech(dataArray);
        })
        .catch(error => {
            showError('无法访问麦克风: ' + error);
        });}

// 检测语音活动
function detectSpeech(dataArray) {
    if (!isRecording) return;
    
    // 获取音频数据
    analyser.getByteFrequencyData(dataArray);
    
    // 计算平均音量
    let sum = 0;
    for(let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i];
    }
    const average = sum / dataArray.length;
    
    // 检测是否有声音
    if (average > SILENCE_THRESHOLD) {
        // 有声音
        if (!isSpeaking) {
            isSpeaking = true;
            console.log('检测到用户开始说话');
            
            // 如果录音机未启动，则开始录音
            if (mediaRecorder && mediaRecorder.state === 'inactive') {
                audioChunks = [];
                mediaRecorder.start(1000); // 每1秒触发一次ondataavailable事件
                console.log('检测到声音，开始录音，状态:', mediaRecorder.state);
            } 
            // 如果已经在录音但有之前的数据，重新开始录音
            else if (mediaRecorder && mediaRecorder.state === 'recording' && audioChunks.length > 0) {
                audioChunks = [];
                mediaRecorder.stop();
                setTimeout(() => {
                    if (isRecording) {
                        mediaRecorder.start(1000);
                        console.log('重置录音，状态:', mediaRecorder.state);
                    }
                }, 100);
            }
        }
        
        // 清除静音计时器
        if (silenceTimer) {
            clearTimeout(silenceTimer);
            silenceTimer = null;
        }
    } else if (isSpeaking) {
        // 检测到静音，但之前在说话
        if (!silenceTimer) {
            console.log('检测到用户停止说话，开始计时');
            silenceTimer = setTimeout(() => {
                // 静音持续了指定时间，处理录音
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    // 确保在停止录音前已经收集到数据
                    if (audioChunks.length === 0) {
                        console.log('等待音频数据收集...');
                        // 强制触发一次数据收集
                        mediaRecorder.requestData();
                        // 给一点时间让数据被收集
                        setTimeout(() => {
                            if (audioChunks.length > 0) {
                                console.log('成功收集到音频数据，音频块数量:', audioChunks.length);
                                try {
                                    mediaRecorder.stop();
                                    console.log('已停止录音，等待处理');
                                } catch (e) {
                                    console.error('停止录音时出错:', e);
                                }
                            } else {
                                console.log('仍未收集到音频数据，放弃本次录音');
                                // 重置录音状态，但不立即开始新录音
                                try {
                                    mediaRecorder.stop();
                                } catch (e) {
                                    console.error('停止录音时出错:', e);
                                }
                            }
                        }, 500);
                    } else {
                        console.log('检测到静音2秒，自动处理录音');
                        console.log('当前录音状态:', mediaRecorder.state);
                        console.log('音频块数量:', audioChunks.length);
                        // 确保在这里停止录音，这将触发onstop事件并调用processRecording
                        try {
                            mediaRecorder.stop();
                            console.log('已停止录音，等待处理');
                        } catch (e) {
                            console.error('停止录音时出错:', e);
                        }
                        // processRecording会通过onstop事件自动调用
                    }
                }
                isSpeaking = false;
                silenceTimer = null;
                console.log('静音计时器已重置，准备下一次录音');
            }, SILENCE_DURATION);
        }
    }
    
    // 继续检测
    if (isRecording) {
        requestAnimationFrame(() => detectSpeech(dataArray));
    }
}

// 停止录音和语音监听
function stopRecording() {
    // 停止录音
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // 清除静音计时器
    if (silenceTimer) {
        clearTimeout(silenceTimer);
        silenceTimer = null;
    }
    
    // 停止音频分析
    if (audioContext) {
        audioContext.close().catch(e => console.error('关闭音频上下文失败:', e));
        audioContext = null;
        analyser = null;
    }
    
    // 停止音频流
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }
    
    isSpeaking = false;
}

// 处理录音结果
function processRecording() {
    console.log('进入processRecording函数，准备处理录音');
    console.log('当前audioChunks长度:', audioChunks.length);
    
    // 显示处理中的消息
    updateChatHistory('正在处理您的语音...', 'user-processing');
    
    if (audioChunks.length === 0) {
        console.log('警告：audioChunks为空，没有录音数据可处理');
        showError('没有录音数据');
        removeTempMessages();
        return;
    }
    
    // 创建音频Blob
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    console.log('已创建音频Blob，大小:', audioBlob.size, '字节');
    
    // 创建FormData对象
    const formData = new FormData();
    formData.append('audio', audioBlob);
    
    console.log('准备发送请求到/audio_to_text_for_chat');
    
    // 清空录音数据，准备下一次录音
    audioChunks = [];
    
    // 发送到服务器进行语音识别
    fetch('/audio_to_text_for_chat', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log('收到服务器响应，状态码:', response.status);
        return response.json();
    })
    .then(data => {
        console.log('解析响应数据:', data);
        if (data.success) {
            // 移除处理中的消息
            removeTempMessages();
            
            const userText = data.text;
            console.log('语音识别成功，文本:', userText);
            if (userText.trim() && !/^[\s.,!?;，。！？；]*$/.test(userText)) {
                // 显示识别出的文字
                updateChatHistory(userText, 'user');
                
                // 发送文字到Ollama进行对话
                console.log('准备发送文本到Ollama进行对话');
                chatWithOllama(userText);
            } else {
                console.log('识别结果为空或仅包含标点，不发送到Ollama');
            }
        } else {
            console.error('语音识别失败:', data.error);
            showError('语音识别失败: ' + data.error);
        }
    })
    .catch(error => {
        console.error('请求失败:', error);
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
    
    // 准备消息历史数据
    // 将chatHistory数组转换为Ollama API所需的格式
    const formattedMessages = chatHistory.map(item => {
        // 将'user'和'assistant'角色映射到Ollama API所需的格式
        const role = item.role === 'user' ? 'user' : 'assistant';
        return {
            role: role,
            content: item.message
        };
    });
    
    // 发送请求到服务器
    fetch('/chat_with_ollama', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: selectedModel,
            message: message,
            messages: formattedMessages
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
            
            // 如果服务器返回了更新后的消息历史，则更新本地历史
            if (data.messages) {
                // 我们不直接替换chatHistory，因为updateChatHistory已经添加了最新的消息
                // 这里只是记录一下服务器返回的完整历史，如果需要可以使用
                console.log('服务器返回的消息历史:', data.messages);
            }
            
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
        console.log('聊天历史已更新，当前消息数:', chatHistory.length);
    }
}

// 清除聊天历史
function clearChatHistory() {
    // 清空聊天历史数组
    chatHistory = [];
    
    // 清空聊天历史显示
    const chatHistoryDiv = document.getElementById('chatHistory');
    chatHistoryDiv.innerHTML = '';
    
    console.log('聊天历史已清除');
    
    // 显示提示消息
    updateChatHistory('聊天历史已清除，开始新的对话吧！', 'system');
    
    // 系统消息不保存到历史中
    chatHistory = [];
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