import cv2
import sys
from pathlib import Path

def resize_image(image_path, scale, interpolation=cv2.INTER_LINEAR):
    """
    일반 이미지를 주어진 배율로 리사이즈하는 함수

    :param image_path: 이미지 파일 경로
    :param scale: 확대/축소 비율
    :param interpolation: 보간법 (cv2.INTER_LINEAR 또는 cv2.INTER_CUBIC)
    :return: 리사이즈된 이미지
    """
    # 이미지 파일을 읽어옵니다. 알파 채널 포함
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to open image file {image_path}")

    # 새로운 크기를 계산합니다.
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    # 이미지를 새로운 크기로 리사이즈합니다.
    resized_image = cv2.resize(image, new_size, interpolation=interpolation)
    return resized_image

def main(image_path, scale, quality='-bicubic'):
    """
    메인 함수

    :param image_path: 이미지 파일 경로
    :param scale: 확대/축소 비율
    :param quality: 보간법의 품질 ('-bicubic'(품질) 또는 '-bilinear'(성능))
    """
    # 사용자가 선택한 보간법에 따라 OpenCV의 보간법 상수를 설정합니다.
    interpolation = cv2.INTER_CUBIC if quality == '-bicubic' else cv2.INTER_LINEAR
    # 이미지를 주어진 배율과 보간법으로 리사이즈합니다.
    resized_image = resize_image(image_path, scale, interpolation)
    # 리사이즈된 이미지를 저장할 경로를 생성합니다.
    output_path = f"resized_{Path(image_path).stem}.png"
    # 리사이즈된 이미지를 저장합니다.
    cv2.imwrite(output_path, resized_image)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: normal_converter <image_path> <scale> <quality>")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        scale = float(sys.argv[2])
    except ValueError:
        print("배율은 반드시 숫자여야 합니다.")
        sys.exit(1)
    
    quality = sys.argv[3].lower()
    if quality not in ['-bicubic', '-bilinear']:
        print("확대 방식은 '-bicubic' 또는 '-bilinear'이어야 합니다..")
        sys.exit(1)
    
    main(image_path, scale, quality)
