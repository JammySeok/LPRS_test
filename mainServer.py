# 대충 ai 돌려서 짠 코드, 솔직히 어떻게 작동 하는지 잘 모르고 코드 해석도 안해봄 (뭐 욜로 모델에 있는 기능 썻겠지)
# 대충 웹 서버(webServer.py)에서 전송 받은 파일(이미지, 영상)을 욜로 모델5(yolov5s)로 돌려서 차량을 확인하고 바운딩 박스로 표시 (차량 인식 기능도 개구림)
# 대충 처리 완료한 파일을 다시 웹 서버로 전송
# 사실 여기서 웹 서버에서 받은 파일의 차량이나 번호판 욜로로 딴 다음에 품질 구지면 다시 스케일 업 서버(/v1/enhance)에 파일 전송하고 품질 좋은 이미지 받아와야 됨
# (Real-ESRGAN 쓰거나, Diffusion으로 이미지 학습시켜서 품질 업 한다는데 지금 내실력으론 못하겠음 더 공부하든 해야지)
# 아직 OCR 기능 없음 나중에 추가 하던가 해야지 (번호판 이미지 텍스트 인식해서 딕셔너리로 보내기)

# 그러므로 웹 관련 코드(데이터 받기, 이미지 인식, 동영상 인식... 은 잘 모르겠고, 데이터 보내기)만 주석 처리 함

from flask import Flask, request, send_file
import cv2
import numpy as np
import io
import torch
import tempfile
import os
import time

app = Flask(__name__)

# 대충 욜로 모델(yolov5s) 처리
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [2, 5, 7]

# localhost:5000/process-image
# 자바로 따지면 PostMapping("/process-image")
@app.route('/process-image', methods=['POST'])
def process_image():   # 이미지 처리
    # 이미지 파일이 요청(request)에 포함 되어 있지 않으면 400 반환
    if 'image' not in request.files:
        return 'No image uploaded', 400
    
    # 웹서버에서 받은 파일 이미지로 읽기
    file = request.files['image'].read()

    # 대충 바이트 데이터를 NumPy 배열로 변환하고 openCV로 디코딩
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # 대충 욜로 처리. BRG를 RGB로 어쩌구저쩌구
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    
    # 대충 draw_bounding_boxes 함수 호출
    processed_img = draw_bounding_boxes(img, results)
    
    # 대충 openCV로 이미지 엔코딩
    _, img_encoded = cv2.imencode('.jpg', processed_img)

    return send_file(
        io.BytesIO(img_encoded.tobytes()),   # 이미지 데이터를 바이트 스트림으로 변환
        mimetype='image/jpeg'   # 타입 지정 (이미지)
    )

# 대충 바운딩 박스 처리하는 코드
def draw_bounding_boxes(img, results):
    if len(results.xyxy[0]) == 0:
        return img

    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection
        if conf < 0.5:
            continue
        cv2.rectangle(img, 
                     (int(x1), int(y1)), 
                     (int(x2), int(y2)), 
                     (0, 255, 0), 2)
    return img

# localhost:5000/process-video
# 자바로 따지면 PostMapping("/process-video")
@app.route('/process-video', methods=['POST'])
def process_video():   # 동영상 처리
    # 동영상 파일이 요청(request)에 포함 되어 있지 않으면 400 반환
    if 'video' not in request.files:
        return 'No video uploaded', 400
    
    # 대충 임시 파일 저장 처리
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
        request.files['video'].save(temp_input.name)   # 비디오 파일 저장
        temp_path = temp_input.name
    
    # 대충 process_video_frames 함수 호출
    output_path = process_video_frames(temp_path)

    try:
        return send_file(
            output_path,   # MIME 타입 지정 (비디오)
            mimetype='video/mp4',   # 타입 지정 (비디오)
            as_attachment=True,   # 파일을 다운로드 가능한 첨부 파일로 처리
            download_name='processed_video.mp4'   # 이름 설정
        )
    finally:
        try:
            os.unlink(output_path)   # 파일 삭제
        except:
            pass

# 대충 비디오를 프레임으로 나눠서 처리하는 코드
def process_video_frames(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Error opening video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        raise RuntimeError("Error opening video writer")

    try:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            processed_frame = draw_bounding_boxes(frame, results)

            out.write(processed_frame)

            if frame_count == 0:
                cv2.imwrite('debug_frame.jpg', processed_frame)
            
            frame_count += 1

            print(f"Processing frame {frame_count}/{total_frames}")
    finally:
        cap.release()
        out.release()

    while True:
        file_size = os.path.getsize(temp_output.name)
        time.sleep(0.5)
        if file_size == os.path.getsize(temp_output.name):
            break
    
    return temp_output.name

# 독립적으로 실행
if __name__ == '__main__':
    # Flask 웹 애플리케이션 실행
    app.run(host='0.0.0.0', port=5000)
