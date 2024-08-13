import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. 모델 로드 및 설정
model = tf.lite.Interpreter(model_path='movenet_thunder.tflite')
model.allocate_tensors()

# 입력 및 출력 텐서 정보 얻기
input_details = model.get_input_details()
output_details = model.get_output_details()

# 입력 크기 확인
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
print("Input shape expected by the model:", input_shape)
print("Input data type expected by the model:", input_dtype)

# 두 점 사이의 각도 계산 함수 (도 단위)
def calculate_angle(point1, point2):
    angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
    return np.degrees(angle)

# 사람의 자세를 판별하는 함수
def determine_posture(keypoints, angle_threshold, position_threshold):
    nose = keypoints[0]
    neck = keypoints[1]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    mid_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
    mid_ankle = [(left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2]

    body_inclination = calculate_angle(nose, right_ankle)
    hip_to_ankle_angle = calculate_angle(mid_hip, mid_ankle)
    relative_position = abs(neck[0] - mid_ankle[0])

    if relative_position >= position_threshold:
        if abs(body_inclination) < angle_threshold or abs(hip_to_ankle_angle) < angle_threshold:
            return "Standing"
    else:
        return "Lying Down"

# 이미지 전처리 함수
def preprocess_image(image, input_shape):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_height, input_width = input_shape[1:3]
    image = cv2.resize(image, (input_width, input_height))
    image = image.astype(np.uint8)  # UINT8 형식으로 변환
    image = np.expand_dims(image, axis=0)
    return image

# 이미지 로드 및 전처리
image_path = 'stand2.jpg'
original_image = cv2.imread(image_path)
input_image = preprocess_image(original_image, input_shape)

# 모델에 이미지 입력
model.set_tensor(input_details[0]['index'], input_image)
model.invoke()
keypoints_with_scores = model.get_tensor(output_details[0]['index'])

# 키포인트 좌표와 점수 추출
keypoints = keypoints_with_scores[0, 0, :, :2]
scores = keypoints_with_scores[0, 0, :, 2]

# 키포인트를 원본 이미지 크기로 조정
h, w, _ = original_image.shape
keypoints *= [w, h]

# 신뢰할 수 있는 키포인트만 필터링
threshold = 0.5
keypoints = np.array([keypoint if score > threshold else [0, 0] for keypoint, score in zip(keypoints, scores)])

# 트랙바 콜백 함수
def update_posture(x):
    angle_threshold = cv2.getTrackbarPos('Angle Threshold', 'Posture Estimation')
    position_threshold = cv2.getTrackbarPos('Position Threshold', 'Posture Estimation')
    posture = determine_posture(keypoints, angle_threshold, position_threshold)

    updated_image = original_image.copy()
    for keypoint in keypoints:
        x, y = keypoint
        if x > 0 and y > 0:  # 신뢰할 수 있는 키포인트만 그림
            cv2.circle(updated_image, (int(y), int(x)), 5, (0, 255, 0), -1)

    cv2.putText(updated_image, posture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Posture Estimation', updated_image)

# 트랙바 윈도우 생성
cv2.namedWindow('Posture Estimation', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('Angle Threshold', 'Posture Estimation', 30, 180, update_posture)
cv2.createTrackbar('Position Threshold', 'Posture Estimation', 50, 500, update_posture)

# 초기 호출
update_posture(0)

# waitKey()를 지속적으로 호출하여 트랙바 이벤트를 처리
while True:
    key = cv2.waitKey(1)
    if key == 27:  # ESC 키를 누르면 종료
        break

cv2.destroyAllWindows()
