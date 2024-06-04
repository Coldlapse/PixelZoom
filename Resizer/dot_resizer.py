import cv2
import sys
import numpy as np
from pathlib import Path

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

def create_restored_image(content, bbox, alpha_channel):
    x, y, w, h = bbox
    
    # 복원할 이미지 초기화
    restored_image = np.zeros((h, w, 4), dtype=np.uint8)
    
    # 원래 이미지가 투명한 배경을 포함할 때
    restored_image[:, :, :3] = content
    restored_image[:, :, 3] = alpha_channel[y:y+h, x:x+w]  # 알파 채널 설정
    
    return restored_image

def compare_images(image1, image2):
    # 두 이미지를 비교
    difference = cv2.absdiff(image1, image2)
    _, diff = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
    
    # 차이가 있는 픽셀 수를 계산
    non_zero_count = np.count_nonzero(diff)
    
    return non_zero_count == 0

def save_image(image, path):
    cv2.imwrite(path, image)

def find_chunk_size(image):
    height, width = image.shape[:2]
    min_val = min(height, width)
        
    # 공약수를 저장할 리스트 초기화
    common_divisors = []

    # 1부터 두 정수 중 작은 값까지 반복하여 공약수 찾기
    for i in range(1, min_val + 1):
        if height % i == 0 and width % i == 0:
            common_divisors.append(i)

    for element in reversed(common_divisors):
        new_height = int(height / element)
        new_width = int(width / element)
        minchunk_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        restored_minchunk_image = cv2.resize(minchunk_image, (width, height), interpolation=cv2.INTER_NEAREST)
        if compare_images(image, restored_minchunk_image):
            return element, new_height, new_width

    return None, None, None

def resize_image_pixel_art(image_path, scale):
    """
    도트 이미지를 가장 근접한 크기로 확대 또는 축소하는 함수

    :param image_path: 이미지 파일 경로
    :param scale: 확대 비율
    :return: 리사이즈된 이미지
    """
    # 이미지 로드 및 전처리
    image = load_image(image_path)
    image_no_bg, mask = preprocess_image(image)
    content, bbox = extract_content(image_no_bg, mask)
    alpha_channel = image[:, :, 3]
    restored_image = create_restored_image(content, bbox, alpha_channel)

    if scale > 1:
        # 확대
        final_width = int(restored_image.shape[1] * scale)
        final_height = int(restored_image.shape[0] * scale)
        resized_image = cv2.resize(restored_image, (final_width, final_height), interpolation=cv2.INTER_NEAREST)
    else:
        # 축소
        chunk_size, minchunk_height, minchunk_width = find_chunk_size(restored_image)
        if chunk_size is None:
            raise ValueError("이미지의 MinChunk 탐색 실패! 이미지 축소 불가능!")
        resize_chunksize = max(1, round(chunk_size * scale))
        final_resize_height = minchunk_height * resize_chunksize
        final_resize_width = minchunk_width * resize_chunksize
        resized_image = cv2.resize(restored_image, (final_resize_width, final_resize_height), interpolation=cv2.INTER_NEAREST)

    return resized_image

def main(image_path, scale):
    """
    메인 함수

    :param image_path: 이미지 파일 경로
    :param scale: 확대 비율
    """
    # 도트 이미지를 리사이즈합니다.
    resized_image = resize_image_pixel_art(image_path, scale)
    # 리사이즈된 이미지를 저장할 경로를 생성합니다.
    output_path = f"resized_{Path(image_path).stem}.png"
    # 리사이즈된 이미지를 저장합니다.
    save_image(resized_image, output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    # 명령줄 인수가 3개 미만일 경우 사용법을 출력하고 종료합니다.
    if len(sys.argv) < 3:
        print("Usage: dot_enlarger <image_path> <scale>")
        sys.exit(1)

    # 첫 번째 명령줄 인수는 이미지 파일 경로입니다.
    image_path = sys.argv[1]
    try:
        # 두 번째 명령줄 인수는 확대 비율입니다.
        scale = float(sys.argv[2])
    except ValueError:
        print("Scale must be a number.")
        sys.exit(1)
    # 메인 함수를 호출하여 이미지 리사이즈를 수행합니다.
    main(image_path, scale)
