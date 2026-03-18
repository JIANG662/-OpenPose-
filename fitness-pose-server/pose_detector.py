import cv2
import numpy as np
import os
import urllib.request

# 优先尝试导入 Legacy API
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    USE_TASKS_API = False
except (ImportError, AttributeError):
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        USE_TASKS_API = False
    except (ImportError, AttributeError):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        USE_TASKS_API = True

# --- 优化点 1: 持久化 Pose 对象，避免每帧重复初始化 ---
_pose_instance = None
_landmarker_instance = None

def get_pose_instance():
    global _pose_instance, _landmarker_instance
    if not USE_TASKS_API:
        if _pose_instance is None:
            # --- 优化点 2: 使用 model_complexity=0 (Lite 模型)，牺牲极小精度换取大幅速度提升 ---
            _pose_instance = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0, 
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return _pose_instance
    else:
        if _landmarker_instance is None:
            download_model()
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False)
            _landmarker_instance = vision.PoseLandmarker.create_from_options(options)
        return _landmarker_instance

def detect_pose(image: np.ndarray):
    """
    姿态检测函数，已优化性能。
    """
    instance = get_pose_instance()
    
    if not USE_TASKS_API:
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 直接使用持久化的实例处理
        results = instance.process(image_rgb)
        image.flags.writeable = True
        
        # 优化：直接在原图上绘制，减少一次 copy() 操作
        landmarks_list = []
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            for landmark in results.pose_landmarks.landmark:
                landmarks_list.append({
                    'x': landmark.x, 'y': landmark.y, 'z': landmark.z, 
                    'visibility': landmark.visibility
                })
        return image, landmarks_list
    else:
        # Tasks API 逻辑同样使用持久化实例
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = instance.detect(mp_image)
        
        # 优化：直接在原图上绘制
        landmarks_list = []
        
        if results.pose_landmarks:
            for pose_landmarks in results.pose_landmarks:
                for landmark in pose_landmarks:
                    landmarks_list.append({
                        'x': landmark.x, 'y': landmark.y, 'z': landmark.z,
                        'visibility': getattr(landmark, 'visibility', 1.0)
                    })
                for lm in pose_landmarks:
                    px, py = int(lm.x * image.shape[1]), int(lm.y * image.height)
                    cv2.circle(image, (px, py), 5, (0, 255, 0), -1)
        
        return image, landmarks_list
