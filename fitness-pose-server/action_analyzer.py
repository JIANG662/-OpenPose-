import numpy as np

def analyze_squat(landmarks):
    """
    分析深蹲动作并提供反馈。

    Args:
        landmarks (list): 从 MediaPipe 获取的关键点列表。

    Returns:
        A tuple containing:
        - feedback (str): 对当前姿态的文本反馈。
        - angle (float): 计算出的膝盖角度。
    """
    if not landmarks:
        return "未检测到人体", 0

    # MediaPipe 关键点索引
    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27
    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28

    # 提取左右两侧的髋、膝、踝关节坐标
    try:
        left_hip = np.array([landmarks[LEFT_HIP]['x'], landmarks[LEFT_HIP]['y']])
        left_knee = np.array([landmarks[LEFT_KNEE]['x'], landmarks[LEFT_KNEE]['y']])
        left_ankle = np.array([landmarks[LEFT_ANKLE]['x'], landmarks[LEFT_ANKLE]['y']])

        right_hip = np.array([landmarks[RIGHT_HIP]['x'], landmarks[RIGHT_HIP]['y']])
        right_knee = np.array([landmarks[RIGHT_KNEE]['x'], landmarks[RIGHT_KNEE]['y']])
        right_ankle = np.array([landmarks[RIGHT_ANKLE]['x'], landmarks[RIGHT_ANKLE]['y']])
    except IndexError:
        return "关键点缺失，无法分析", 0

    # 计算膝盖角度
    def calculate_angle(a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle
        return angle

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # 简单评估逻辑
    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

    if avg_knee_angle > 160:
        feedback = "请下蹲"
    elif avg_knee_angle < 90:
        feedback = "蹲得太深了"
    else:
        feedback = "动作标准"

    return feedback, avg_knee_angle
