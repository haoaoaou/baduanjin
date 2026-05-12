import os
import time
import cv2  # 新增：用于处理图像可视化
import torch
import numpy as np
import ollama
from flask import Flask, request, render_template, jsonify,send_from_directory
from werkzeug.utils import secure_filename

# 引入自定义模块
from src.rtmpose_tran import RTM_Pose_Tran
from src.datapro import PreProcess
from src.score import Score
from src.model import ST_GCN

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

VIDEO_DIR = r"D:\桌面\配套视频"

@app.route('/local_videos/<path:filename>')
def serve_video(filename):

    return send_from_directory(VIDEO_DIR, filename)

ACTION_CLASSES = {
    0: "双手托天理三焦", 1: "左右开弓似射雕", 2: "调理脾胃须单举",
    3: "五劳七伤往后瞧", 4: "摇头摆尾去心火", 5: "双手攀足固肾腰",
    6: "攒拳怒目增气力", 7: "背后七颠百病消", 8: "虎戏", 9: "鹿戏",
    10: "熊戏", 11: "猿戏", 12: "鸟戏", 13: "收势", 14: "无法识别/其他"
}

session_histories = {}
model, device = None, None


def load_global_model():
    global model, device
    if model is None:

        model_path = r"model/best_model_7_exchange_val_and_test.pth"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用 {device} 加载模型")

        # 注意：如果一会儿运行报错 size mismatch，大概率是因为这个模型是 14 分类的，把 15 改成 14 即可
        model = ST_GCN(num_classes=15, in_channels=2, t_kernel_size=9, hop_size=1)

        print(f"准备加载模型权重: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("加载模型权重成功")

        model.to(device)
        model.eval()


load_global_model()


# --- 辅助逻辑函数 ---

def estimate_heart_rate(keypoints):
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
    try:
        start = text.find(start_marker)
        if start == -1: return ""
        start += len(start_marker)
        end = text.find(end_marker, start) if end_marker else len(text)
        if end == -1: end = len(text)
        return text[start:end].strip().replace("*", "")
    except:
        return ""


def generate_feedback(action_id, score, heart_rate=None):
    hr_info = f"心率: {heart_rate} BPM" if heart_rate else "心率: 未检测"
    prompt = f"""
    你是一个专业的健身教练和中医养生专家。
    动作名称: {ACTION_CLASSES.get(action_id, '未知')}
    动作评分: {score:.2f} (满分1.00)
    {hr_info}
    请按以下格式给出反馈，语气要温和：
    [动作评价]...[评分分析]...[心率评估]...[改进建议]...[鼓励话语]...
    """
    try:
        response = ollama.chat(model='deepseek-r1:8b', messages=[{'role': 'user', 'content': prompt}])
        full_text = response['message']['content']

        return {
            'raw': full_text,
            'evaluation': extract_section(full_text, '[动作评价]', '[评分分析]'),
            'analysis': extract_section(full_text, '[评分分析]', '[心率评估]'),
            'hr_eval': extract_section(full_text, '[心率评估]', '[改进建议]'),
            'suggestion': extract_section(full_text, '[改进建议]', '[鼓励话语]'),
            'encouragement': extract_section(full_text, '[鼓励话语]')
        }
    except Exception as e:
        return {'raw': "无法生成建议", 'evaluation': "生成失败", 'analysis': "", 'hr_eval': "", 'suggestion': "",
                'encouragement': ""}


def create_visualization(video_path, keypoints, filename):
    """新增：截取中间帧并画出骨骼连线，用于前端显示"""
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
        ret, frame = cap.read()
        cap.release()

        if ret and keypoints is not None:
            # 取中间帧的骨骼点
            kp_frame = keypoints[len(keypoints) // 2]
            # 简单的连线逻辑 (基于COCO 17点)
            skeleton = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12),
                        (5, 11), (6, 12)]
            for p1, p2 in skeleton:
                pt1 = (int(kp_frame[p1][0]), int(kp_frame[p1][1]))
                pt2 = (int(kp_frame[p2][0]), int(kp_frame[p2][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                cv2.circle(frame, pt1, 4, (0, 0, 255), -1)
                cv2.circle(frame, pt2, 4, (0, 0, 255), -1)

            out_filename = f"vis_{int(time.time())}.jpg"
            out_path = os.path.join(app.config['RESULT_FOLDER'], out_filename)
            cv2.imwrite(out_path, frame)
            return f"/{app.config['RESULT_FOLDER']}/{out_filename}"
    except Exception as e:
        print(f"Visualization error: {e}")
    return ""


# --- 路由 ---

@app.route('/')
def index():
    return render_template('index.html', result=None)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files: return render_template('index.html', error="未选择文件")
    file = request.files['video']
    if file.filename == '': return render_template('index.html', error="文件名为空")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        start_time = time.time()
        good_vid, keypoints = RTM_Pose_Tran(filepath, display_pose=False)

        if not good_vid: return render_template('index.html', error="无法提取骨骼关键点")

        pp_keypoints = torch.tensor(PreProcess(keypoints), dtype=torch.float32, device=device)
        action, conf = model.predict(pp_keypoints)
        action_id = action[0][0]
        score = Score(keypoints, action_id, conf[0][0])

        if score < 0.5 and conf[0][0] < 0.5:
            action_id = 14;
            score = 0

        heart_rate = estimate_heart_rate(keypoints)
        duration = time.time() - start_time

        # 截取可视化图像
        vis_image_path = create_visualization(filepath, keypoints, filename)
        feedback_data = generate_feedback(action_id, score, heart_rate)

        result_data = {
            'filename': filename,
            'action_id': action_id,
            'action_name': ACTION_CLASSES[action_id],
            'score': score,
            'heart_rate': heart_rate,
            'duration': duration,
            'frame_count': keypoints.shape[0],
            'feedback': feedback_data,
            'vis_image': vis_image_path  # 传回前端渲染
        }

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
    app.run(host='0.0.0.0', port=4000, debug=True)