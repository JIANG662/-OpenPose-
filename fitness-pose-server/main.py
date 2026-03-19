from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import uvicorn
import sys
import os
from PIL import Image, ImageDraw, ImageFont

# 导入自定义模块
try:
    from pose_detector import detect_pose
    from action_analyzer import analyze_squat, analyze_pushup, analyze_jumping_jack
    print("--- Custom modules imported successfully ---")
except Exception as e:
    print(f"--- Error importing custom modules: {e} ---")
    sys.exit(1)

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 尝试加载中文字体，如果失败则回退
font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
if not os.path.exists(font_path):
    font_path = "C:/Windows/Fonts/simhei.ttf" # 黑体

# 缓存不同大小的字体对象
font_cache = {}

def get_font(size):
    if size not in font_cache:
        try:
            font_cache[size] = ImageFont.truetype(font_path, size)
        except Exception:
            return None
    return font_cache[size]

def draw_chinese_text(img, text, position, color=(0, 255, 0), font_size=32):
    """使用 PIL 在图片上绘制中文"""
    font = get_font(font_size)
    
    if font is None:
        # 如果字体加载失败，退回到 cv2 (会显示问号，但至少不崩溃)
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        return img
    
    # 转换 OpenCV 图像 (BGR) 到 PIL 图像 (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # 绘制文字
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0])) # PIL 使用 RGB
    # 转换回 OpenCV 图像
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class ImageData(BaseModel):
    image: str
    exercise_type: str = "squat" # 默认深蹲

@app.post("/analyze")
def analyze(data: ImageData):
    try:
        image_data = data.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 1. 姿态检测与骨架绘制
        image_with_skeleton, landmarks = detect_pose(frame)

        # 2. 根据 exercise_type 选择动作分析函数
        if data.exercise_type == "pushup":
            feedback, angle, counter = analyze_pushup(landmarks)
            exercise_name = "俯卧撑"
        elif data.exercise_type == "jumping_jack":
            feedback, angle, counter = analyze_jumping_jack(landmarks)
            exercise_name = "开合跳"
        else:
            feedback, angle, counter = analyze_squat(landmarks)
            exercise_name = "深蹲"

        # 在图像上显示反馈、角度和计数
        # 使用自定义函数绘制中文
        image_with_skeleton = draw_chinese_text(image_with_skeleton, f"项目: {exercise_name}", (10, 10), font_size=28)
        image_with_skeleton = draw_chinese_text(image_with_skeleton, f"角度: {int(angle)}°", (10, 45), font_size=28)
        # 反馈信息使用较小的字体，防止过长溢出
        image_with_skeleton = draw_chinese_text(image_with_skeleton, f"反馈: {feedback}", (10, 80), font_size=22)
        image_with_skeleton = draw_chinese_text(image_with_skeleton, f"计数: {counter}", (10, 115), color=(0, 0, 255), font_size=32)

        # 编码返回
        _, buffer = cv2.imencode('.jpg', image_with_skeleton)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        return {
            "processed_image": "data:image/jpeg;base64," + encoded_image,
            "feedback": feedback,
            "angle": angle,
            "counter": counter
        }
    except Exception as e:
        print(f"Error during analysis: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("--- Starting server via uvicorn.run ---")
    # 直接运行，不使用 reload 模式，这样更稳定
    uvicorn.run(app, host="0.0.0.0", port=8000)
