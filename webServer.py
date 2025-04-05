import streamlit as st
import requests
from PIL import Image
import io

# Main server configuration
MAIN_SERVER_URL = "http://localhost:5000/process-image"

st.title("Vehicle License Plate Recognition System")

# 파일 업로더 타입 변경
uploaded_file = st.file_uploader("Upload media", type=["jpg", "png", "jpeg", "mp4", "avi"])

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Send to main server
        files = {'image': uploaded_file.getvalue()}
        response = requests.post(MAIN_SERVER_URL, files=files)
        
        if response.status_code == 200:
            # Receive processed image
            processed_image = Image.open(io.BytesIO(response.content))
            st.image(processed_image, caption='Processed Result', use_column_width=True)
        else:
            st.error("Error processing image")
    elif uploaded_file.type.startswith('video'):
        # 동영상 처리 로직
        video_bytes = uploaded_file.read()
        
        # 메인 서버로 전송
        files = {'video': video_bytes}
        response = requests.post("http://localhost:5000/process-video", files=files)
        
        if response.status_code == 200:
            # 처리된 동영상 다운로드 링크 제공
            st.video(response.content)
        else:
            st.error("Error processing video")