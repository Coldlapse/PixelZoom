import cv2
import numpy as np

# 
def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

def preprocess_image(image):
    # 이미지를 회색조로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 투명한 배경이 있는 경우
    if image.shape[2] == 4:
        alpha_channel = image[:, :, 3]
        _, mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
    else:
        # 하얀 배경을 검출
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # 배경을 제거한 이미지 생성
    image_no_bg = cv2.bitwise_and(gray, gray, mask=mask)
    
    return image_no_bg, mask

def extract_content(image, mask):
    # 외곽선을 찾아서 그 외곽선의 경계 상자를 구함
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    content = image[y:y+h, x:x+w]
    return content

def normalize_image(image, size=(100, 100)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def compare_images(image1, image2):
    # 두 이미지를 비교
    difference = cv2.absdiff(image1, image2)
    _, diff = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
    
    # 차이가 있는 픽셀 수를 계산
    non_zero_count = np.count_nonzero(diff)
    
    return non_zero_count == 0

# 이미지 경로 설정
image_path1 = 'Test/dot1.png'
image_path2 = 'Test/dot2.png'

# 이미지 로드
image1 = load_image(image_path1)
image2 = load_image(image_path2)

# 이미지 전처리
image1_no_bg, mask1 = preprocess_image(image1)
image2_no_bg, mask2 = preprocess_image(image2)

# 내용물 추출
content1 = extract_content(image1_no_bg, mask1)
content2 = extract_content(image2_no_bg, mask2)

# 내용물 정규화
normalized_content1 = normalize_image(content1)
normalized_content2 = normalize_image(content2)

# 이미지 비교
are_images_equal = compare_images(normalized_content1, normalized_content2)

print("이미지가 일치합니다." if are_images_equal else "이미지가 일치하지 않습니다.")
