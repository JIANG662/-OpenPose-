import numpy as np

def calculate_angle(a, b, c):
    """
    计算三个坐标点之间的夹角（角度制）。
    a: 第一个点 (x, y)
    b: 中间点 (x, y) - 顶点
    c: 最后一个点 (x, y)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

class SquatAnalyzer:
    def __init__(self):
        # 动作计数
        self.counter = 0
        # 动作阶段: "up" (站立/上升), "down" (下蹲)
        self.stage = "up"
        # 平滑后的角度 (EMA 滤波)
        self.smoothed_angle = None
        # 滤波系数 (0.1 ~ 0.5, 越小越平滑但延迟越高)
        self.alpha = 0.3

    def process(self, landmarks):
        """
        处理姿态关键点，进行动作分析和计数。
        """
        if not landmarks:
            return "未检测到人体", 0, self.counter

        # MediaPipe 关键点索引
        LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
        RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28

        try:
            # 获取关节坐标
            l_hip = [landmarks[LEFT_HIP]['x'], landmarks[LEFT_HIP]['y']]
            l_knee = [landmarks[LEFT_KNEE]['x'], landmarks[LEFT_KNEE]['y']]
            l_ankle = [landmarks[LEFT_ANKLE]['x'], landmarks[LEFT_ANKLE]['y']]
            
            r_hip = [landmarks[RIGHT_HIP]['x'], landmarks[RIGHT_HIP]['y']]
            r_knee = [landmarks[RIGHT_KNEE]['x'], landmarks[RIGHT_KNEE]['y']]
            r_ankle = [landmarks[RIGHT_ANKLE]['x'], landmarks[RIGHT_ANKLE]['y']]
            
            # 计算双膝角度并取平均
            l_angle = calculate_angle(l_hip, l_knee, l_ankle)
            r_angle = calculate_angle(r_hip, r_knee, r_ankle)
            current_angle = (l_angle + r_angle) / 2
            
            # --- 优化点 1: 平滑滤波 (EMA) ---
            if self.smoothed_angle is None:
                self.smoothed_angle = current_angle
            else:
                self.smoothed_angle = self.alpha * current_angle + (1 - self.alpha) * self.smoothed_angle
            
            # --- 优化点 2: 状态机动作计数 ---
            feedback = "请下蹲"
            
            # 下蹲检测 (下蹲到 90 度以下)
            if self.smoothed_angle < 90:
                if self.stage == "up":
                    self.stage = "down"
                feedback = "动作标准"
            
            # 起身检测 (回到 160 度以上且之前处于下蹲状态)
            if self.smoothed_angle > 160:
                if self.stage == "down":
                    self.stage = "up"
                    self.counter += 1  # 完成一次动作，计数加一
                feedback = "请下蹲"
            
            # 过程中的反馈
            if 90 <= self.smoothed_angle <= 160:
                feedback = "正在下蹲..." if self.stage == "down" else "继续下蹲"

            return feedback, self.smoothed_angle, self.counter

        except (IndexError, KeyError):
            return "关键点提取错误", 0, self.counter

# 为了保持 main.py 的兼容性，创建一个默认实例
default_analyzer = SquatAnalyzer()

def analyze_squat(landmarks):
    # 调用默认实例的 process 方法
    return default_analyzer.process(landmarks)
