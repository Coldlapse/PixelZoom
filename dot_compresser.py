import cv2
import numpy as np

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

def preprocess_image(image):
    # 투명한 배경이 있는 경우
    if image.shape[2] == 4:
        alpha_channel = image[:, :, 3]
        _, mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
        image_no_bg = image[:, :, :3]
    else:
        # 회색조로 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 하얀 배경을 검출
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        image_no_bg = image

    return image_no_bg, mask

def extract_content(image, mask):
    # 외곽선을 찾아서 그 외곽선의 경계 상자를 구함
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    content = image[y:y+h, x:x+w]
    return content, (x, y, w, h)

def create_restored_image(content, bbox, original_image_shape, alpha_channel):
    x, y, w, h = bbox
    
    # 복원할 이미지 초기화
    restored_image = np.zeros((h, w, 4), dtype=np.uint8)
    
    # 원래 이미지가 투명한 배경을 포함할 때
    restored_image[:, :, :3] = content
    restored_image[:, :, 3] = alpha_channel[y:y+h, x:x+w]  # 알파 채널 설정
    
    return restored_image

def save_image(image, path):
    cv2.imwrite(path, image)

# 이미지 경로 설정
image_path = 'res_compresser/dot4.png'
output_path = 'res_compresser/output_image4.png'

# 이미지 로드
image = load_image(image_path)

# 이미지 전처리
image_no_bg, mask = preprocess_image(image)

# 내용물 추출
content, bbox = extract_content(image_no_bg, mask)

# 알파 채널 추출
alpha_channel = image[:, :, 3]

# 내용물을 포함하는 새 이미지 생성 (투명 배경 유지)
restored_image = create_restored_image(content, bbox, image.shape, alpha_channel)

height, width = restored_image.shape[:2]



# 복원된 이미지 저장
save_image(restored_image, output_path)

print(f"복원된 이미지가 {output_path}에 저장되었습니다.")
