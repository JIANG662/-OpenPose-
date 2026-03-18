<template>
  <div id="main-container">
    <h2>Fitness Pose Detection</h2>

    <div id="video-container">
      <div>
        <h3>Live Camera</h3>
        <video ref="video" width="640" height="480" autoplay playsinline></video>
      </div>
      <div>
        <h3>Processed Output</h3>
        <img :src="processedImage" width="640" height="480" />
      </div>
    </div>

    <div id="feedback-container">
      <h3>Real-time Feedback</h3>
      <p>{{ feedback }}</p>
      <p>Knee Angle: <span>{{ angle }}</span></p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';

// Refs for DOM elements
const video = ref(null);

// Reactive state for data
const processedImage = ref('');
const feedback = ref('-');
const angle = ref('-');

let intervalId = null;
let isProcessing = false; // 请求锁

// All requests will be proxied by Vite. No need for full IP address.
const BACKEND_URL = "/api/analyze";

const captureAndAnalyze = () => {
  if (!video.value || video.value.videoWidth === 0) return;

  const canvas = document.createElement("canvas");
  // 优化：降低画布尺寸以减少数据量
  const scale = 0.5; // 将尺寸缩小为原来的 50%
  // const scale = 0.3; // 进一步缩小图片尺寸，降低延迟
  canvas.width = video.value.videoWidth * scale;
  canvas.height = video.value.videoHeight * scale;
  
  canvas.getContext("2d").drawImage(video.value, 0, 0, canvas.width, canvas.height);
  
  // 优化：降低 JPEG 质量（0.7 表示 70% 的质量）
  const image = canvas.toDataURL("image/jpeg", 0.7);

  // 优化：如果上一个请求还没回来，就不发送新的
  if (isProcessing) return;
  isProcessing = true;

  fetch(BACKEND_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ image: image })
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      console.error("Backend Error:", data.error);
      feedback.value = `Error: ${data.error}`;
      return;
    }
    processedImage.value = data.processed_image;
    feedback.value = data.feedback;
    angle.value = Math.round(data.angle);
  })
  .catch((error) => {
    console.error("Fetch Error:", error);
    feedback.value = "Connection to backend failed. Is it running?";
  })
  .finally(() => {
    isProcessing = false; // 释放锁
  });
};

onMounted(async () => {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (video.value) {
        video.value.srcObject = stream;
        // 优化：提高尝试频率，但实际频率由请求锁控制
        intervalId = setInterval(captureAndAnalyze, 200); // 每 200ms 尝试一次
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
      feedback.value = "Could not access camera. Please grant permission.";
    }
  } else {
    feedback.value = "getUserMedia not supported on this browser.";
  }
});

onUnmounted(() => {
  // Clean up the interval when the component is destroyed
  if (intervalId) {
    clearInterval(intervalId);
  }
});
</script>

<style>
body {
  font-family: Arial, sans-serif;
  background-color: #f0f2f5;
  color: #333;
}

#main-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
  padding: 20px;
}

#video-container {
  display: flex;
  flex-wrap: wrap; /* Allow wrapping on smaller screens */
  justify-content: center;
  gap: 20px;
}

#feedback-container {
  margin-top: 20px;
  padding: 15px;
  border: 1px solid #ccc;
  border-radius: 8px;
  background-color: #fff;
  min-width: 300px;
  text-align: center;
}

video, img {
  border-radius: 8px;
  background-color: #000;
}
</style>
