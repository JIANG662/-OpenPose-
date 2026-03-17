import cv2
import numpy as np
import os
import requests

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")

def run_openpose():
    # Paths for the model
    protoFile = "openpose/models/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "openpose/models/pose/coco/pose_iter_440000.caffemodel"
    
    # Download weights if not exist (This is the COCO model)
    if not os.path.exists(weightsFile):
        print("OpenPose weights file not found!")
        print("Please run the following command to download models:")
        print("cd openpose/models && ./getModels.sh")
        # Alternatively, download manually if needed:
        # url = "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel"
        # download_file(url, weightsFile)
        return

    # Check for image
    imagePath = "openpose/examples/media/COCO_val2014_000000000192.jpg"
    if not os.path.exists(imagePath):
        print(f"Image {imagePath} not found!")
        return

    # Load the network
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # Read image
    frame = cv2.imread(imagePath)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    # Pre-process image
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    # Set input to the network
    net.setInput(inpBlob)

    # Forward pass
    output = net.forward()

    # Get keypoints
    H = output.shape[2]
    W = output.shape[3]
    
    # COCO Keypoints: 0-Nose, 1-Neck, 2-RShoulder, 3-RElbow, 4-RWrist, 5-LShoulder, 6-LElbow, 7-LWrist, 
    # 8-RHip, 9-RKnee, 10-RAnkle, 11-LHip, 12-LKnee, 13-LAnkle, 14-REye, 15-LEye, 16-REar, 17-LEar, 18-Background
    points = []
    for i in range(18):
        # Confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > 0.1 :
            cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else :
            points.append(None)

    # Save output
    cv2.imwrite("output.jpg", frame)
    print("OpenPose finished! Output saved to output.jpg")

if __name__ == "__main__":
    run_openpose()
