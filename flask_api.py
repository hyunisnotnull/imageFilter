from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

CORS(app)

# 모델 로딩 함수
def load_model():
    model = models.resnet34(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 7)  # 7개의 클래스 (술, 담배, 약, 현금, 총, 통장, 신분증)
    model.load_state_dict(torch.load('D:/kjh/python/download/model.pth'))  # 학습된 모델 로드
    model.eval()
    return model

# 이미지 전처리 함수
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # image가 이미 PIL 이미지 객체라면, BytesIO로 감싸지 않음
    if isinstance(image, Image.Image):
        image = image.convert("RGB")  # RGB로 강제 변환 (RGBA나 다른 색상 공간을 처리하기 위해)
        image = transform(image).unsqueeze(0)  # 배치 차원 추가
    else:
        image = Image.open(io.BytesIO(image)).convert("RGB")  # 바이트 스트림을 이미지로 변환하고 RGB로 변환
        image = transform(image).unsqueeze(0)  # 배치 차원 추가
    
    return image

# 금지된 항목 이미지 경로 (업로드된 이미지와 비교할 금지된 항목들)
banned_image_paths = {
    '술': 'D:/kjh/python/pjt/goodbuy/banned_images/image2.jpg',
    '담배': 'D:/kjh/python/pjt/goodbuy/banned_images/image1.jpg',
    '약': 'D:/kjh/python/pjt/goodbuy/banned_images/image4.jpg',
    '현금': 'D:/kjh/python/pjt/goodbuy/banned_images/image7.jpg',
    '총': 'D:/kjh/python/pjt/goodbuy/banned_images/image5.jpg',
    '통장': 'D:/kjh/python/pjt/goodbuy/banned_images/image6.jpg',
    '신분증': 'D:/kjh/python/pjt/goodbuy/banned_images/image3.png',
}

# 금지된 이미지들의 특징 벡터 저장
banned_image_features = {}

# 금지된 이미지들의 특징 벡터 추출
def extract_banned_image_features():
    model = load_model()  # 모델 로드
    for category, image_path in banned_image_paths.items():
        if os.path.exists(image_path):  # 파일 존재 여부 확인
            image = Image.open(image_path).convert("RGB")  # 이미지를 열 때 RGB로 변환
            image_tensor = process_image(image)  # 이미지 처리
            # 특징 벡터 추출
            with torch.no_grad():
                features = model(image_tensor)
            banned_image_features[category] = features.cpu().numpy().flatten()  # 1D 벡터로 변환
        else:
            print(f"경고: 금지된 이미지 파일이 존재하지 않습니다: {image_path}")

# 업로드된 이미지와 금지된 이미지 간의 유사도 계산
def calculate_similarity(uploaded_image_tensor):
    model = load_model()  # 모델 로드
    similarities = {}
    
    try:
        with torch.no_grad():
            uploaded_features = model(uploaded_image_tensor)
        uploaded_embedding = uploaded_features.cpu().numpy().flatten()  # 1D 벡터로 변환
        
        for category, banned_embedding in banned_image_features.items():
            similarity = cosine_similarity([uploaded_embedding], [banned_embedding])
            similarities[category] = similarity[0][0]
            print(f"{category} 유사도: {similarity[0][0]:.4f}")  # 유사도 출력 (디버깅용)
    except Exception as e:
        print(f"유사도 계산 중 오류 발생: {e}")
    
    return similarities


# 예측 함수
def predict(image_bytes):
    model = load_model()  # 모델 로드
    image_tensor = process_image(image_bytes)  # 바이트 스트림을 PIL 이미지로 변환하여 처리
    
    # 예측
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
    
    # 클래스 이름 (예: 술, 담배, ... )
    class_names = ['술', '담배', '약', '현금', '총', '통장', '신분증']
    predicted_class_name = class_names[predicted_class.item()]
    
    return predicted_class_name

# 이미지 업로드 및 예측 API
@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # 파일을 바이트 스트림으로 읽어 예측 수행
        img_bytes = file.read()
        
        # 업로드된 이미지의 특징 벡터와 금지된 이미지들의 특징 벡터 간의 유사도 계산
        uploaded_image_tensor = process_image(img_bytes)
        similarities = calculate_similarity(uploaded_image_tensor)
        
        # 유사도가 높은 금지된 항목 판단 (예시: 유사도 0.8 이상이면 거부)
        threshold = 0.8
        for category, similarity in similarities.items():
            if similarity >= threshold:
                return jsonify({"message": f"이미지가 {category}와 유사하여 등록이 거부되었습니다."}), 400
        
        # 예측 결과 반환
        return jsonify({"message": "상품이 성공적으로 등록되었습니다."}), 200


if __name__ == "__main__":
    # 금지된 이미지들의 특징 벡터 추출 (서버가 시작될 때 한 번만 실행)
    extract_banned_image_features()
    
    app.run(debug=True, host="0.0.0.0", port=5000)
