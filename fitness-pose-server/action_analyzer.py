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

class PushupAnalyzer:
    def __init__(self):
        self.counter = 0
        self.stage = "up"
        self.smoothed_angle = None
        self.alpha = 0.3

    def process(self, landmarks):
        if not landmarks:
            return "未检测到人体", 0, self.counter

        # MediaPipe 关键点索引: 肩、肘、腕
        LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 11, 13, 15
        RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 12, 14, 16

        try:
            l_sh = [landmarks[LEFT_SHOULDER]['x'], landmarks[LEFT_SHOULDER]['y']]
            l_el = [landmarks[LEFT_ELBOW]['x'], landmarks[LEFT_ELBOW]['y']]
            l_wr = [landmarks[LEFT_WRIST]['x'], landmarks[LEFT_WRIST]['y']]
            
            r_sh = [landmarks[RIGHT_SHOULDER]['x'], landmarks[RIGHT_SHOULDER]['y']]
            r_el = [landmarks[RIGHT_ELBOW]['x'], landmarks[RIGHT_ELBOW]['y']]
            r_wr = [landmarks[RIGHT_WRIST]['x'], landmarks[RIGHT_WRIST]['y']]
            
            # 计算肘部角度
            l_angle = calculate_angle(l_sh, l_el, l_wr)
            r_angle = calculate_angle(r_sh, r_el, r_wr)
            current_angle = (l_angle + r_angle) / 2
            
            if self.smoothed_angle is None:
                self.smoothed_angle = current_angle
            else:
                self.smoothed_angle = self.alpha * current_angle + (1 - self.alpha) * self.smoothed_angle
            
            feedback = "请向下俯卧"
            
            # 下压检测 (肘部角度小于 90 度)
            if self.smoothed_angle < 90:
                if self.stage == "up":
                    self.stage = "down"
                feedback = "动作标准"
            
            # 撑起检测 (回到 160 度以上)
            if self.smoothed_angle > 160:
                if self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
                feedback = "请向下俯卧"
            
            if 90 <= self.smoothed_angle <= 160:
                feedback = "正在撑起..." if self.stage == "up" else "继续下压"

            return feedback, self.smoothed_angle, self.counter

        except (IndexError, KeyError):
            return "关键点提取错误", 0, self.counter

class JumpingJackAnalyzer:
    def __init__(self):
        self.counter = 0
        self.stage = "down" # "down" 表示手臂在下，"up" 表示手臂在上
        self.smoothed_angle = None
        self.alpha = 0.3

    def process(self, landmarks):
        if not landmarks:
            return "未检测到人体", 0, self.counter

        # MediaPipe 关键点索引: 肩、髋、手腕、脚踝
        LEFT_SHOULDER, LEFT_HIP, LEFT_WRIST = 11, 23, 15
        RIGHT_SHOULDER, RIGHT_HIP, RIGHT_WRIST = 12, 24, 16
        LEFT_ANKLE, RIGHT_ANKLE = 27, 28

        try:
            # 获取关键点坐标
            l_sh = [landmarks[LEFT_SHOULDER]['x'], landmarks[LEFT_SHOULDER]['y']]
            r_sh = [landmarks[RIGHT_SHOULDER]['x'], landmarks[RIGHT_SHOULDER]['y']]
            l_wr = [landmarks[LEFT_WRIST]['x'], landmarks[LEFT_WRIST]['y']]
            r_wr = [landmarks[RIGHT_WRIST]['x'], landmarks[RIGHT_WRIST]['y']]
            l_hip = [landmarks[LEFT_HIP]['x'], landmarks[LEFT_HIP]['y']]
            r_hip = [landmarks[RIGHT_HIP]['x'], landmarks[RIGHT_HIP]['y']]
            l_ank = [landmarks[LEFT_ANKLE]['x'], landmarks[LEFT_ANKLE]['y']]
            r_ank = [landmarks[RIGHT_ANKLE]['x'], landmarks[RIGHT_ANKLE]['y']]
            
            # 1. 计算手臂角度 (髋-肩-腕)
            l_angle = calculate_angle(l_hip, l_sh, l_wr)
            r_angle = calculate_angle(r_hip, r_sh, r_wr)
            current_angle = (l_angle + r_angle) / 2
            
            # 2. 计算腿部开合程度 (双脚间距 vs 肩宽)
            # 使用欧式距离
            def dist(p1, p2):
                return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
            
            shoulder_width = dist(l_sh, r_sh)
            ankle_dist = dist(l_ank, r_ank)
            # 归一化比例：双脚间距 / 肩宽
            leg_ratio = ankle_dist / (shoulder_width + 1e-6)
            
            # --- 角度平滑 (EMA) ---
            if self.smoothed_angle is None:
                self.smoothed_angle = current_angle
            else:
                self.smoothed_angle = self.alpha * current_angle + (1 - self.alpha) * self.smoothed_angle
            
            feedback = "请开始开合跳"
            
            # 判定标准：
            # 手臂打开 (角度 > 150) 且 双腿跳开 (比例 > 1.3)
            # 手臂收回 (角度 < 40) 且 双腿并拢 (比例 < 1.1)
            
            # 检测跳起阶段 (Up)
            if self.smoothed_angle > 140:
                if leg_ratio > 1.3:
                    if self.stage == "down":
                        self.stage = "up"
                    feedback = "动作标准: 跳起"
                else:
                    feedback = "请注意: 腿部要跳开"
            
            # 检测收回阶段 (Down)
            if self.smoothed_angle < 40:
                if leg_ratio < 1.2: # 稍微放宽并拢的标准
                    if self.stage == "up":
                        self.stage = "down"
                        self.counter += 1
                    feedback = "动作标准: 收回"
                else:
                    if self.stage == "up":
                        feedback = "请收回双腿"

            return feedback, self.smoothed_angle, self.counter

        except (IndexError, KeyError):
            return "关键点提取错误", 0, self.counter

# 为了保持 main.py 的兼容性，创建一个默认实例
default_squat_analyzer = SquatAnalyzer()
default_pushup_analyzer = PushupAnalyzer()
default_jumping_jack_analyzer = JumpingJackAnalyzer()

def analyze_squat(landmarks):
    return default_squat_analyzer.process(landmarks)

def analyze_pushup(landmarks):
    return default_pushup_analyzer.process(landmarks)

def analyze_jumping_jack(landmarks):
    return default_jumping_jack_analyzer.process(landmarks)
