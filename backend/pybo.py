from flask import Flask, request, jsonify, render_template, send_file
from ultralytics import YOLO
import cv2
import json

app = Flask(__name__)

# YOLO 모델 로드
model = YOLO('model/weights/best.pt')

# 절대 위치 그룹 정보 로드 및 해상도 조정
with open('absolute_positions .json') as f:
    absolute_positions = json.load(f)

# 원본 그룹 바운딩 박스의 해상도
original_width = 2304
original_height = 1296

# 그룹 색상 정의
group_colors = {
    "Group_1": (0, 0, 255),
    "Group_2": (255, 0, 0),
    "Group_3": (0, 255, 0),
    "Group_4": (0, 255, 255),
    "Group_5": (0, 165, 255),
    "Group_6": (128, 0, 128)
}

# 겹치는 면적을 계산하는 함수
def get_overlap_area(box1, box2):
    """Calculates the overlap area between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x1 < x2 and y1 < y2:
        return (x2 - x1) * (y2 - y1)
    return 0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        video_path = 'uploaded_video.mp4'
        video_file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 400

        # 현재 비디오의 해상도 추출
        target_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        target_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 스케일 계수 계산
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        # 그룹 좌표 조정
        for group in absolute_positions:
            group['xmin'] = int(group['xmin'] * scale_x)
            group['ymin'] = int(group['ymin'] * scale_y)
            group['xmax'] = int(group['xmax'] * scale_x)
            group['ymax'] = int(group['ymax'] * scale_y)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter('output_with_bboxes.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_width, target_height))

        # 결과를 저장할 리스트
        results = []

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results_frame = model(frame)
            boxes = results_frame[0].boxes.xyxy.cpu().numpy()
            classes = results_frame[0].boxes.cls.cpu().numpy()
            confidences = results_frame[0].boxes.conf.cpu().numpy()

            # 그룹 별 바운딩 박스 그리기
            for group in absolute_positions:
                gx1, gy1, gx2, gy2 = group['xmin'], group['ymin'], group['xmax'], group['ymax']
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), group_colors[group['name']], 2)

            # 프레임별 그룹 결과 초기화
            frame_results = {group['name']: {'table': 0, 'chair': 0, 'person': 0, 'useable': True} for group in absolute_positions}

            # 객체 탐지 및 그룹 할당
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box[:4])
                max_overlap = 0
                assigned_group = None

                for group in absolute_positions:
                    gx1, gy1, gx2, gy2 = group['xmin'], group['ymin'], group['xmax'], group['ymax']
                    overlap_area = get_overlap_area([x1, y1, x2, y2], [gx1, gy1, gx2, gy2])

                    if overlap_area > max_overlap:
                        max_overlap = overlap_area
                        assigned_group = group['name']

                if assigned_group:
                    color = group_colors[assigned_group]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{model.names[int(cls)]} {conf:.2f} Group: {assigned_group}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # 그룹별 객체 수 카운트
                    if model.names[int(cls)] == 'table':
                        frame_results[assigned_group]['table'] += 1
                    elif model.names[int(cls)] == 'chair':
                        frame_results[assigned_group]['chair'] += 1
                    elif model.names[int(cls)] == 'person':
                        frame_results[assigned_group]['person'] += 1

            # 사람 수에 따른 useable 속성 설정
            for group_name, group_data in frame_results.items():
                group_data['useable'] = group_data['person'] == 0

            # 초 단위로 결과 저장
            if frame_index % fps == 0:
                second_index = frame_index // fps
                results.append({'second': second_index, 'groups': frame_results})

            out.write(frame)
            frame_index += 1

        cap.release()
        out.release()

        # result.json 저장
        with open('result.json', 'w') as json_file:
            json.dump(results, json_file, indent=4)

        return send_file('output_with_bboxes.mp4', as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
