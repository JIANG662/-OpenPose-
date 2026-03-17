from fastapi import FastAPI
import base64
import cv2
import numpy as np

app = FastAPI()

@app.post("/analyze")
def analyze(data: dict):

    image_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)

    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 这里未来调用OpenPose
    result = "pose detected"

    return {"result": result}