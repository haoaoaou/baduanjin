import os
import time
import torch
import numpy as np
import ollama
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# 引入自定义模块 (保持不变)
from src.rtmpose_tran import RTM_Pose_Tran
from src.datapro import PreProcess
from src.score import Score
from src.model import ST_GCN

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 动作类别映射
ACTION_CLASSES = {
    0: "双手托天理三焦", 1: "左右开弓似射雕", 2: "调理脾胃须单举",
    3: "五劳七伤往后瞧", 4: "摇头摆尾去心火", 5: "双手攀足固肾腰",
    6: "攒拳怒目增气力", 7: "背后七颠百病消", 8: "虎戏", 9: "鹿戏",
    10: "熊戏", 11: "猿戏", 12: "鸟戏", 13: "收势", 14: "无法识别/其他"
}

# 全局变量
session_histories = {}
model, device = None, None


def load_global_model():
    """初始化模型，只加载一次"""
    global model, device
    if model is None:
        # 修改为你的实际路径
        model_path = r"model/bestbest/best_model_3.pth"
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA 可用，使用 GPU 加载模型")
        else:
            device = torch.device("cpu")
            print("CUDA 不可用，使用 CPU 加载模型")

        model = ST_GCN(num_classes=15, in_channels=2, t_kernel_size=9, hop_size=1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()


# 启动时加载模型
load_global_model()


# --- 辅助逻辑函数 ---

def estimate_heart_rate(keypoints):
    """估算心率逻辑"""
    if keypoints is None or len(keypoints) < 2: return None
    total_movement = 0
    num_points = keypoints.shape[1]
    for i in range(1, len(keypoints)):
        frame_diff = np.abs(keypoints[i] - keypoints[i - 1])
        total_movement += np.sum(frame_diff)
    avg_movement = total_movement / (len(keypoints) * num_points)
    base_hr = 70
    movement_factor = min(avg_movement * 500, 50)
    return max(60, int(min(base_hr + movement_factor, 180)))


def extract_section(text, start_marker, end_marker=None):
    """从文本中提取特定段落"""
    try:
        start = text.find(start_marker)
        if start == -1: return ""
        start += len(start_marker)
        end = text.find(end_marker, start) if end_marker else len(text)
        if end == -1: end = len(text)
        return text[start:end].strip()
    except:
        return ""


def generate_feedback(action_id, score, heart_rate=None):
    """调用 Ollama 生成并解析反馈"""
    hr_info = f"心率: {heart_rate} BPM" if heart_rate else "心率: 未检测"
    prompt = f"""
    你是一个专业的健身教练和中医养生专家。
    动作名称: {ACTION_CLASSES.get(action_id, '未知')}
    动作评分: {score:.2f} (满分1.00)
    {hr_info}
    请按以下格式给出反馈：
    [动作评价]...[评分分析]...[心率评估]...[改进建议]...[鼓励话语]...
    """
    try:
        response = ollama.chat(model='deepseek-r1:8b', messages=[{'role': 'user', 'content': prompt}])
        full_text = response['message']['content']

        # 在后端直接解析好结构化数据，前端直接取用
        return {
            'raw': full_text,
            'evaluation': extract_section(full_text, '[动作评价]', '[评分分析]'),
            'analysis': extract_section(full_text, '[评分分析]', '[心率评估]'),
            'hr_eval': extract_section(full_text, '[心率评估]', '[改进建议]'),
            'suggestion': extract_section(full_text, '[改进建议]', '[鼓励话语]'),
            'encouragement': extract_section(full_text, '[鼓励话语]')
        }
    except Exception as e:
        print(f"Ollama Error: {e}")
        return {'raw': "无法生成建议", 'evaluation': "生成失败", 'analysis': "", 'hr_eval': "", 'suggestion': "",
                'encouragement': ""}


# --- 路由 ---

@app.route('/')
def index():
    # 渲染首页，result 为 None 表示未上传状态
    return render_template('index.html', result=None)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return render_template('index.html', error="未选择文件")

    file = request.files['video']
    if file.filename == '':
        return render_template('index.html', error="文件名为空")

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # 1. 视频推理
            start_time = time.time()
            good_vid, keypoints = RTM_Pose_Tran(filepath, display_pose=False)

            if not good_vid:
                # 处理无效视频
                return render_template('index.html', error="无法提取骨骼关键点，请确保视频包含人物全身")

            pp_keypoints = torch.tensor(PreProcess(keypoints), dtype=torch.float32, device=device)
            action, conf = model.predict(pp_keypoints)
            action_id = action[0][0]
            score = Score(keypoints, action_id, conf[0][0])

            if score < 0.5 and conf[0][0] < 0.5:
                action_id = 14
                score = 0

            heart_rate = estimate_heart_rate(keypoints)
            duration = time.time() - start_time

            # 2. 生成反馈
            feedback_data = generate_feedback(action_id, score, heart_rate)

            # 3. 构造结果数据传给模板
            result_data = {
                'filename': filename,
                'action_id': action_id,
                'action_name': ACTION_CLASSES[action_id],
                'score': score,
                'heart_rate': heart_rate,
                'duration': duration,
                'frame_count': keypoints.shape[0],
                'feedback': feedback_data
            }

            # 清理文件
            try:
                os.remove(filepath)
            except:
                pass

            return render_template('index.html', result=result_data)

        except Exception as e:
            try:
                os.remove(filepath)
            except:
                pass
            return render_template('index.html', error=f"处理发生错误: {str(e)}")


@app.route('/chat', methods=['POST'])
def chat():
    """保持原有的聊天接口逻辑"""
    try:
        data = request.get_json()
        user_msg = data.get('message')
        action_id = int(data.get('action'))
        score = float(data.get('score'))

        session_id = request.remote_addr
        if session_id not in session_histories:
            bg_info = f"用户动作：{ACTION_CLASSES.get(action_id)}，评分：{score:.2f}。"
            session_histories[session_id] = [
                {'role': 'system', 'content': f'你是一个健身教练。{bg_info} 请回答用户问题。'}]

        session_histories[session_id].append({'role': 'user', 'content': user_msg})
        response = ollama.chat(model='deepseek-r1:8b', messages=session_histories[session_id])
        reply = response['message']['content']
        session_histories[session_id].append({'role': 'assistant', 'content': reply})

        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'reply': '系统繁忙，请重试'}), 500


if __name__ == '__main__':
    print("应用启动: http://localhost:4000")
    app.run(host='0.0.0.0', port=4000, debug=True)