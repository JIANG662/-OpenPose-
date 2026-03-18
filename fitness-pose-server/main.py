from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import uvicorn
import sys

# 导入自定义模块
try:
    from pose_detector import detect_pose
    from action_analyzer import analyze_squat
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

class ImageData(BaseModel):
    image: str

@app.post("/analyze")
def analyze(data: ImageData):
    try:
        image_data = data.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 1. 姿态检测与骨架绘制
        image_with_skeleton, landmarks = detect_pose(frame)

        # 2. 动作分析
        feedback, angle = analyze_squat(landmarks)

        # 在图像上显示反馈
        cv2.putText(image_with_skeleton, f"Knee Angle: {int(angle)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image_with_skeleton, f"Feedback: {feedback}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 编码返回
        _, buffer = cv2.imencode('.jpg', image_with_skeleton)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        return {
            "processed_image": "data:image/jpeg;base64," + encoded_image,
            "feedback": feedback,
            "angle": angle
        }
    except Exception as e:
        print(f"Error during analysis: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("--- Starting server via uvicorn.run ---")
    # 直接运行，不使用 reload 模式，这样更稳定
    uvicorn.run(app, host="0.0.0.0", port=8000)
