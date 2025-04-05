from flask import Flask, request, send_file
import cv2
import numpy as np
import io
import torch
import tempfile
import os
import time

app = Flask(__name__)

# Load YOLO model (modify path as needed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [2, 5, 7]  # car, bus, truck 클래스만 검출

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    
    # Read image
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # YOLO processing
    # BGR -> RGB 변환 추가
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    
    # 원본 이미지에 바운딩 박스 그리기
    processed_img = draw_bounding_boxes(img, results)
    
    # 회색조 변환 코드 제거 (아래 줄 주석 처리)
    # processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert back to bytes
    _, img_encoded = cv2.imencode('.jpg', processed_img)
    return send_file(
        io.BytesIO(img_encoded.tobytes()),
        mimetype='image/jpeg'
    )

def draw_bounding_boxes(img, results):
    # 결과가 없는 경우 원본 이미지 반환
    if len(results.xyxy[0]) == 0:
        return img
    
    # 박스 그리기
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection
        if conf < 0.5:  # 신뢰도 50% 이상만 표시
            continue
        cv2.rectangle(img, 
                     (int(x1), int(y1)), 
                     (int(x2), int(y2)), 
                     (0, 255, 0), 2)
    return img

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return 'No video uploaded', 400
    
    # 임시 파일 저장 (with 문 사용으로 자동 정리)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
        request.files['video'].save(temp_input.name)
        temp_path = temp_input.name
    
    # 비디오 처리
    output_path = process_video_frames(temp_path)
    
    # 파일 핸들 닫기 대기
    try:
        return send_file(
            output_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name='processed_video.mp4'
        )
    finally:
        # 결과 파일 정리
        try:
            os.unlink(output_path)
        except:
            pass

def process_video_frames(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Error opening video file")
    
    # 비디오 정보 추출
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 메모리 기반 비디오 작성기 사용
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output.close()  # 파일 핸들 닫기
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 코덱 사용
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
                
            # 프레임 처리 (수정 부분)
            results = model(frame)  # BGR 직접 사용
            processed_frame = draw_bounding_boxes(frame, results)
            
            # 컬러 변환 코드 제거 (이 부분 삭제)
            out.write(processed_frame)
            
            # 디버깅용 프레임 저장
            if frame_count == 0:
                cv2.imwrite('debug_frame.jpg', processed_frame)
            
            frame_count += 1
            
            # 진행률 표시 (디버깅용)
            print(f"Processing frame {frame_count}/{total_frames}")
            
    finally:
        cap.release()
        out.release()
    
    # 파일이 완전히 기록되었는지 확인
    while True:
        file_size = os.path.getsize(temp_output.name)
        time.sleep(0.5)
        if file_size == os.path.getsize(temp_output.name):
            break
    
    return temp_output.name

if __name__ == '__main__':
    # Initialize YOLO model here
    app.run(host='0.0.0.0', port=5000)
