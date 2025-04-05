# streamlit: UI(사용자 인터페이스)를 생성하는 데 사용.
# requests: 업로드된 파일을 메인 서버로 전송하기 위해 HTTP 요청(POST)을 보내는 데 사용.
# PIL.Image: 이미지 파일 처리를 위한 라이브러리.
# io: 입출력 처리 (io.BytesIO)
import streamlit as st
import requests
from PIL import Image
import io

# 메인 서버 URL (이미지 처리와 동영상 처리 페이지)
MAIN_SERVER_IMAGE = "http://localhost:5000/process-image"
MAIN_SERVER_VIDIO = "http://localhost:5000/process-video"

st.title("Vehicle License Plate Recognition System")

# file_uploader : 파일을 업로드 하는 UI 및 기능을 제공
# 허용 하는 파일 형식 설정. (jpg, png, jpeg, mp4, avi)
uploaded_file = st.file_uploader("Upload media", type=["jpg", "png", "jpeg", "mp4", "avi"])

# 파일을 업로드 했는지 확인 (jpg, png, jpeg, mp4, avi 이라면 실행)
if uploaded_file is not None:

    if uploaded_file.type.startswith('image'):   # 업로드 된 파일 유형이 이미지 일때
        # 업로드 한 이미지 읽고, 웹 화면에 표시
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # 업로드 된 이미지 파일 데이터 files 딕셔너리로 저장 (서버 전송을 위해)
        files = {'image': uploaded_file.getvalue()}

        # 메인 서버로 POST 요청 (mainServer.py로 files 전송)
        response = requests.post(MAIN_SERVER_IMAGE, files=files)

        # 어떤 응답이 반환을 했는지 (200: 성공, 그 외는 실패)
        if response.status_code == 200:   # 성공 했을때
            # response.content로 서버로 전송된 이미지 받고, io.BytesIO를 사용해 바이너리 데이터를 메모리 스트림 객체로 처리 (파일로 저장하지 않아도 되도록...)
            # 서버에서 반환된 이미지 열고, 화면에 표시
            processed_image = Image.open(io.BytesIO(response.content))
            st.image(processed_image, caption='Processed Result', use_column_width=True)
        else:   # 그 외(실패 했을때)
            st.error("Error processing image")
    elif uploaded_file.type.startswith('video'):   # 업로드 된 파일 유형이 영상 일때
        # 업로드 한 동영상 파일 읽기 (프레임 하나 하나 딕셔너리로 저장)
        video_bytes = uploaded_file.read()

        # 메인 서버로 전송
        files = {'video': video_bytes}
        response = requests.post(MAIN_SERVER_VIDIO, files=files)

        # 어떤 응답이 반환을 했는지 (200: 성공, 그 외는 실패)
        if response.status_code == 200:   # 성공 했을때
            st.video(response.content)
        else:   # 그 외(실패 했을때)
            st.error("Error processing video")