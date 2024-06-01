import cv2
import sys
import numpy as np
from pathlib import Path

def resize_image_pixel_art(image_path, scale):
    """
    도트 이미지를 가장 근접한 크기로 확대하는 함수

    :param image_path: 이미지 파일 경로
    :param scale: 확대 비율
    :return: 리사이즈된 이미지
    """
    # 이미지 파일을 읽어옵니다.
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to open image file {image_path}")

    # 원본 이미지의 높이와 너비를 구합니다.
    orig_height, orig_width = image.shape[:2]
    # 새로운 크기를 계산합니다.
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    # 배율이 1보다 큰 경우, 가장 근접한 2의 거듭제곱으로 올림합니다.
    new_width = 2 ** int(np.ceil(np.log2(new_width)))
    new_height = 2 ** int(np.ceil(np.log2(new_height)))

    # 두 축 중 작은 쪽을 기준으로 동일한 비율을 적용합니다.
    aspect_ratio = min(new_width / orig_width, new_height / orig_height)
    final_width = int(orig_width * aspect_ratio)
    final_height = int(orig_height * aspect_ratio)

    # 새로운 크기로 이미지를 리사이즈합니다.
    resized_image = cv2.resize(image, (final_width, final_height), interpolation=cv2.INTER_NEAREST)
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
    cv2.imwrite(output_path, resized_image)
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
