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
import mysql.connector # 添加 MySQL 连接器
import mysql.connector.pooling # 添加连接池支持

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
            disable_update=True
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
        model_name = data.get('model', 'qwen3:14b')
        user_message = data.get('message', '')
        messages_history = data.get('messages', [])
        
        # 如果没有提供历史消息，则创建只包含当前消息的列表
        if not messages_history:
            messages = [
                {"role": "user", "content": user_message}
            ]
        else:
            # 使用提供的历史消息列表，并添加当前消息
            messages = messages_history.copy()
            messages.append({"role": "user", "content": user_message})
        
        # 构建请求数据
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False
        }
        
        # 发送请求到Ollama API
        response = requests.post(f"{OLLAMA_API_URL}/chat", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            assistant_message = result.get('message', {}).get('content', '')
            cleaned_response_str = re.sub(r'<think>.*?</think>', '', assistant_message, flags=re.DOTALL).strip()
            # 移除常见的 Markdown 代码块标记 (```json ... ```, ``` ... ```)
            cleaned_response_str = re.sub(r'^```(?:json)?s*', '', cleaned_response_str)
            cleaned_response_str = re.sub(r's*```$', '', cleaned_response_str).strip()
            # 将助手回复添加到消息历史中
            messages.append({"role": "assistant", "content": cleaned_response_str})
            
            return jsonify({
                'success': True, 
                'response': cleaned_response_str,
                'messages': messages  # 返回更新后的消息历史
            })
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

# --- 自然语言转 SQL 相关 --- #

def execute_sql(sql_query):
    """执行 SQL 查询并返回结果"""
    conn = None
    cursor = None
    print(f"--- Executing SQL ---\n{sql_query}\n---------------------") # 添加日志：打印 SQL
    try:
        conn = get_db_connection()
        if conn is None:
            error_msg = "无法获取数据库连接"
            print(f"SQL Error: {error_msg}") # 添加日志：打印错误
            return False, error_msg
        cursor = conn.cursor(dictionary=True) # 使用字典游标
        cursor.execute(sql_query)
        if sql_query.strip().upper().startswith(("SELECT", "SHOW", "DESC")):
            result = cursor.fetchall()
            # 尝试将结果转换为 JSON 字符串，处理 datetime 等特殊类型
            try:
                result_json = json.dumps(result, ensure_ascii=False, default=str) # 使用 default=str 处理无法序列化的类型
            except TypeError as e:
                print(f"JSON 序列化错误: {e}, 将返回原始结果列表")
                result_json = str(result) # 如果序列化失败，返回原始列表的字符串表示
            print(f"SQL Result (JSON): {result_json}") # 添加日志：打印 JSON 结果
            return True, result_json
        else:
            conn.commit()
            result_message = f"操作成功，影响行数: {cursor.rowcount}"
            print(f"SQL Result: {result_message}") # 添加日志：打印操作结果
            return True, result_message
    except mysql.connector.Error as err:
        error_msg = f"SQL 执行错误: {err}"
        print(f"SQL Error: {error_msg}") # 添加日志：打印 SQL 错误
        return False, error_msg
    except Exception as e:
        error_msg = f"执行 SQL 时发生意外错误: {e}"
        print(f"SQL Error: {error_msg}") # 添加日志：打印其他错误
        print(traceback.format_exc()) # 打印详细堆栈信息
        return False, error_msg
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

def get_tables():
    """获取数据库中的所有表名"""
    print("--- Calling get_tables() ---") # 添加日志：函数入口
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            error_msg = "无法获取数据库连接"
            print(f"get_tables Error: {error_msg}") # 添加日志：错误
            return False, error_msg
        cursor = conn.cursor()
        # 尝试获取当前连接的数据库名称
        cursor.execute("SELECT DATABASE()")
        current_db = cursor.fetchone()
        if not current_db or not current_db[0]:
            error_msg = "未选择数据库。请在 DB_CONFIG 中指定 'database' 或在连接后使用 'USE database_name;'"
            print(f"get_tables Error: {error_msg}") # 添加日志：错误
            return False, error_msg
        db_name = current_db[0]
        print(f"当前数据库: {db_name}")

        # 获取该数据库的表
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        print(f"get_tables Success: Found tables: {tables}") # 添加日志：成功
        return True, tables
    except mysql.connector.Error as err:
        error_msg = f"获取表列表错误: {err}"
        print(f"get_tables Error: {error_msg}") # 添加日志：SQL 错误
        return False, error_msg
    except Exception as e:
        error_msg = f"获取表列表时发生意外错误: {e}"
        print(f"get_tables Error: {error_msg}") # 添加日志：其他错误
        print(traceback.format_exc())
        return False, error_msg
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
        print("--- Exiting get_tables() ---") # 添加日志：函数出口

def get_columns(table_name):
    """获取指定表的列信息"""
    print(f"--- Calling get_columns(table_name='{table_name}') ---") # 添加日志：函数入口和参数
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            error_msg = "无法获取数据库连接"
            print(f"get_columns Error for table '{table_name}': {error_msg}") # 添加日志：错误
            return False, error_msg
        cursor = conn.cursor()
        # 检查表是否存在以及获取列信息
        # 使用 DESCRIBE 语句，更通用
        cursor.execute(f"DESCRIBE `{table_name}`") # 使用反引号处理特殊表名
        columns = [column[0] for column in cursor.fetchall()]
        if not columns:
             error_msg = f"表 '{table_name}' 不存在或没有列"
             print(f"get_columns Warning for table '{table_name}': {error_msg}") # 添加日志：警告/未找到
             return False, error_msg
        print(f"get_columns Success for table '{table_name}': Found columns: {columns}") # 添加日志：成功
        return True, columns
    except mysql.connector.Error as err:
        error_msg = f"获取列信息错误: {err}"
        print(f"get_columns Error for table '{table_name}': {error_msg}") # 添加日志：SQL 错误
        # 检查是否是表不存在的错误
        if err.errno == mysql.connector.errorcode.ER_NO_SUCH_TABLE:
             return False, f"表 '{table_name}' 不存在"
        return False, f"获取列信息错误: {err}"
    except Exception as e:
        error_msg = f"获取列信息时发生意外错误: {e}"
        print(f"get_columns Error for table '{table_name}': {error_msg}") # 添加日志：其他错误
        print(traceback.format_exc())
        return False, error_msg
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
        print(f"--- Exiting get_columns(table_name='{table_name}') ---") # 添加日志：函数出口

def get_create_table_statement(table_name):
    """获取指定表的 CREATE TABLE 语句"""
    print(f"--- Calling get_create_table_statement(table_name='{table_name}') ---") # 添加日志：函数入口和参数
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            error_msg = "无法获取数据库连接"
            print(f"get_create_table_statement Error for table '{table_name}': {error_msg}") # 添加日志：错误
            return False, error_msg
        cursor = conn.cursor()
        cursor.execute(f"SHOW CREATE TABLE `{table_name}`")
        result = cursor.fetchone()
        if result:
            create_statement = result[1]
            print(f"get_create_table_statement Success for table '{table_name}'") # 添加日志：成功
            return True, create_statement # CREATE TABLE 语句在第二个位置
        else:
            error_msg = f"无法获取表 '{table_name}' 的 CREATE TABLE 语句"
            print(f"get_create_table_statement Warning for table '{table_name}': {error_msg}") # 添加日志：警告/未找到
            return False, error_msg
    except mysql.connector.Error as err:
        error_msg = f"获取 CREATE TABLE 语句错误: {err}"
        print(f"get_create_table_statement Error for table '{table_name}': {error_msg}") # 添加日志：SQL 错误
        if err.errno == mysql.connector.errorcode.ER_NO_SUCH_TABLE:
             return False, f"表 '{table_name}' 不存在"
        return False, f"获取 CREATE TABLE 语句错误: {err}"
    except Exception as e:
        error_msg = f"获取 CREATE TABLE 语句时发生意外错误: {e}"
        print(f"get_create_table_statement Error for table '{table_name}': {error_msg}") # 添加日志：其他错误
        print(traceback.format_exc())
        return False, error_msg
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
        print(f"--- Exiting get_create_table_statement(table_name='{table_name}') ---") # 添加日志：函数出口

@ollama_chat_bp.route('/nl_to_sql', methods=['POST'])
def nl_to_sql():
    data = request.json
    user_question = data.get('question')
    print(f"接收到用户命令: {user_question}")
    model_name = data.get('model', 'qwen3:14b') # 可以让前端选择模型
    print(f"使用模型: {model_name}")

    max_retries = 50 # 防止无限循环
    retries = 0
    messages = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"question": user_question})}
    ]

    while retries < max_retries:
        retries += 1
        try:
            # 调用 Ollama API
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "response_format": {"type": "json_object"},
            }
            print("messages:\n", json.dumps(messages, indent=2, ensure_ascii=False))
            response = requests.post(f"{OLLAMA_API_URL}/chat", json=payload)
            response.raise_for_status() # 检查 HTTP 错误
            print(f"Ollama API Response text ({retries}): {response.text}")
            ollama_response_str = response.json().get('message', {}).get('content', '')
            print(f"Ollama Raw Response ({retries}): {ollama_response_str}")

            # 清理响应字符串：移除 <think>...</think> 标签和 Markdown 代码块标记
            cleaned_response_str = re.sub(r'<think>.*?</think>', '', ollama_response_str, flags=re.DOTALL).strip()
            # 移除常见的 Markdown 代码块标记 (```json ... ```, ``` ... ```)
            cleaned_response_str = re.sub(r'^```(?:json)?s*', '', cleaned_response_str)
            cleaned_response_str = re.sub(r's*```$', '', cleaned_response_str).strip()
            print(f"Cleaned Response String ({retries}): {cleaned_response_str}")

            try:
                # 使用清理后的字符串进行 JSON 解析
                ollama_response = json.loads(cleaned_response_str)
            except json.JSONDecodeError:
                 print(f"错误：Ollama 未返回有效的 JSON (即使清理后): {cleaned_response_str}")
                 # 将错误信息包装成用户消息，让模型知道出错了
                 error_message_to_model = {
                     "error": "Invalid JSON response received",
                     "details": cleaned_response_str # 发送清理后的字符串给模型参考
                 }
                 messages.append({"role": "user", "content": json.dumps(error_message_to_model)})
                 continue # 继续循环，让模型重试

            # 将模型的原始回复（清理后）添加到历史记录中，以便进行下一轮对话
            messages.append({"role": "assistant", "content": cleaned_response_str})

            # 处理 Ollama 的响应 (使用解析后的 ollama_response)
            if 'action' in ollama_response:
                action = ollama_response['action']
                db_result_payload = {"action": action}
                success = False
                result_data = None

                if action == 'show_tables':
                    success, result_data = get_tables()
                elif action == 'show_columns':
                    table_name = ollama_response.get('table_name')
                    if table_name:
                        db_result_payload['table_name'] = table_name
                        success, result_data = get_columns(table_name)
                    else:
                        result_data = "错误：'show_columns' 动作需要 'table_name' 参数"
                elif action == 'show_create_table':
                    table_name = ollama_response.get('table_name')
                    if table_name:
                        db_result_payload['table_name'] = table_name
                        success, result_data = get_create_table_statement(table_name)
                    else:
                        result_data = "错误：'show_create_table' 动作需要 'table_name' 参数"
                else:
                    result_data = f"错误：未知的动作 '{action}'"

                db_result_payload['result'] = result_data if success else f"数据库操作失败: {result_data}"
                # 将数据库操作结果作为用户消息发回给 Ollama
                messages.append({"role": "user", "content": json.dumps(db_result_payload)})
                continue # 继续循环，让模型根据数据库信息生成下一步

            elif 'sql' in ollama_response:
                sql_query = ollama_response['sql']
                print(f"Executing SQL: {sql_query}")
                success, result_data = execute_sql(sql_query)
                db_result_payload = {
                    "sql": sql_query,
                    "result": result_data if success else f"SQL 执行失败: {result_data}"
                }
                # 将 SQL 执行结果作为用户消息发回给 Ollama
                messages.append({"role": "user", "content": json.dumps(db_result_payload)})
                continue # 继续循环，让模型根据 SQL 结果生成最终答案

            elif 'answer' in ollama_response:
                # 模型生成了最终答案
                final_answer = ollama_response['answer']
                print(f"处理结果: {final_answer}")
                return jsonify({'success': True, 'answer': final_answer})
            else:
                # 模型返回了未知格式的 JSON
                print(f"错误：Ollama 返回了无法处理的 JSON 格式: {ollama_response_str}")
                error_message_to_model = {
                     "error": "Unexpected JSON format received",
                     "details": ollama_response_str
                 }
                messages.append({"role": "user", "content": json.dumps(error_message_to_model)})
                continue # 继续循环，让模型重试

        except requests.exceptions.RequestException as e:
            print(traceback.format_exc())
            return jsonify({'success': False, 'error': f'调用 Ollama API 失败: {e}'}), 500
        except Exception as e:
            print(traceback.format_exc())
            return jsonify({'success': False, 'error': f'处理 NL 到 SQL 时发生内部错误: {e}'}), 500

    # 如果循环结束仍未得到答案
    return jsonify({'success': False, 'error': '无法在指定次数内从模型获取有效答案'}), 500
# MySQL 数据库配置
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 9981,
    'user': 'root',
    'password': '1234qwer',
    'database': 'employees', # 数据库名称，如果需要指定
    'raise_on_warnings': True
}

# 创建数据库连接池
db_pool = None

def get_db_connection():
    global db_pool
    if db_pool is None:
        try:
            db_pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="mypool",
                pool_size=5, # 根据需要调整连接池大小
                **DB_CONFIG
            )
            print("数据库连接池创建成功")
        except mysql.connector.Error as err:
            print(f"创建数据库连接池失败: {err}")
            raise # 重新抛出异常，以便上层处理
    try:
        # 从连接池获取连接
        conn = db_pool.get_connection()
        if conn.is_connected():
            return conn
        else:
            print("从连接池获取的连接无效")
            return None # 或者尝试重新获取
    except mysql.connector.Error as err:
        print(f"从连接池获取连接失败: {err}")
        return None

# SQL 生成的系统提示
SQL_SYSTEM_PROMPT = """# 角色与目标
你是一个专门分析用户自然语言需求并将其转化为 SQL 查询的专家系统。你的核心任务是：
1.  理解用户的自然语言问题。
2.  **如有必要，通过特定的指令查询数据库的结构（表和列）。**
3.  根据用户需求和获取到的数据库结构信息，生成准确的 SQL 查询以获取数据。
4.  在接收到数据查询的执行结果后，生成一个自然的语言回答返回给用户。

# 工作流程与规则
你将遵循以下严格的多轮交互流程：

1.  **接收用户请求**: 你会收到一个包含用户自然语言问题的 JSON 对象。
    ```json
    {
        "question": "用户的自然语言问题"
    }
    ```

2.  **分析与决策**:
    *   分析 `question` 中的用户意图。
    *   **判断**: 你是否需要了解数据库中有哪些表，或者某个特定表有哪些列才能构建最终的 SQL 查询？
        *   **如果需要了解有哪些表**: 输出 `show_tables` 指令。
        *   **如果需要了解特定表的列**: 输出 `show_columns` 指令 (或 `show_create_table`)。
        *   **如果已掌握足够信息**: 直接生成数据查询 SQL。

3.  **输出指令/SQL**: 根据你的决策，输出以下**其中一种** JSON 格式。**不允许包含任何其他字符、文字、解释或注释。**

    *   **3.1 请求表列表**:
        ```json
        {
            "action": "show_tables"
        }
        ```
    *   **3.2 请求特定表的列信息**: (选择一种你更容易处理的方式，`show_columns` 通常更直接)
        ```json
        {
            "action": "show_columns",
            "table_name": "目标表名"
        }
        ```
        *或者*
        ```json
        {
            "action": "show_create_table",
            "table_name": "目标表名"
        }
        ```
    *   **3.3 生成数据查询 SQL**: (只有在你确认了表和字段存在后才能执行此步骤)
        ```json
        {
            "sql": "SELECT column FROM table WHERE condition;"
        }
        ```

4.  **接收指令/SQL 执行结果**: 外部系统执行你的指令或 SQL 后，你会收到包含结果的 JSON 对象。

    *   **4.1 `show_tables` 结果**:
        ```json
        {
            "action": "show_tables",
            "result": ["table1", "students", "orders"] // 表名列表
        }
        ```
    *   **4.2 `show_columns` 结果**:
        ```json
        {
            "action": "show_columns",
            "table_name": "students",
            "result": ["student_id", "name", "major", "age"] // 列名列表
        }
        ```
    *   **4.3 `show_create_table` 结果**:
        ```json
        {
            "action": "show_create_table",
            "table_name": "students",
            "result": "CREATE TABLE `students` (\n  `student_id` int,\n  `name` varchar(50),\n  `major` varchar(50),\n  `age` int\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;" // DDL 语句
        }
        ```
    *   **4.4 数据查询 SQL 结果**:
        ```json
        {
            "sql": "执行过的 SELECT SQL 查询",
            "result": "SQL 执行返回的结果 (例如: '500', '[{\"name\":\"张三\", \"age\":20}]', '[]')"
        }
        ```

5.  **处理结果与循环/生成最终答案**:
    *   **如果是 Schema 信息结果 (4.1, 4.2, 4.3)**: 记录下这些信息，然后**返回步骤 2**，根据更新后的知识重新决策，判断是否还需要更多信息或可以生成数据查询 SQL。
    *   **如果是数据查询 SQL 结果 (4.4)**:
        *   分析收到的 `sql` 和 `result`。
        *   基于**仅限于**提供的 `result`，构建一个简洁、自然的语言回答给最终用户。
        *   将最终答案封装在**严格**如下的 JSON 格式中输出。**不允许包含任何其他文字、解释或注释。**
        ```json
        {
            "answer": "最终的自然语言回答"
        }
        ```

# 严格的输出格式要求
*   你的**所有**响应**必须**是 JSON 格式。
*   **绝对不允许**在 JSON 对象之外包含任何字符、引导性文字、结束语、解释、道歉或任何其他字符。你的输出必须**仅**是定义的几种 JSON 格式之一 (`{"action": ...}`, `{"sql": ...}`, 或 `{"answer": ...}`)。

"""