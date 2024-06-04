import cv2
import sys
import numpy as np
from pathlib import Path

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

def find_chunk_size(image):
    height, width = image.shape[:2]
    min_val = min(height, width)
    common_divisors = []
    for i in range(1, min_val + 1):
        if height % i == 0 and width % i == 0:
            common_divisors.append(i)
    for element in reversed(common_divisors):
        new_height = int(height / element)
        new_width = int(width / element)
        minchunk_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        restored_minchunk_image = cv2.resize(minchunk_image, (width, height), interpolation=cv2.INTER_NEAREST)
        if np.array_equal(image, restored_minchunk_image):
            return element
    return 1  

def resize_image(image, scale):
    height, width = image.shape[:2]
    
    # 확대
    if scale >= 1:
        nearest_scale = int(np.round(scale))
        new_size = (width * nearest_scale, height * nearest_scale)
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
    
    # 축소
    else:
        chunk_size = find_chunk_size(image)
        print(f"Chunk size: {chunk_size}")
        
        # 청크 사이즈 기반 축소
        target_height = int(height * scale)
        target_width = int(width * scale)

        # 청크 크기를 고려한 축소
        new_height = max(chunk_size, (target_height // chunk_size) * chunk_size)
        new_width = max(chunk_size, (target_width // chunk_size) * chunk_size)
        
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    return resized_image

def save_image(image, path):
    cv2.imwrite(path, image)

def main(image_path, scale):
    # 이미지 로드
    image = load_image(image_path)
    if image is None:
        raise ValueError(f"Unable to open image file {image_path}")

    # 이미지 리사이즈
    resized_image = resize_image(image, scale)

    # 리사이즈된 이미지를 저장할 경로를 생성
    output_path = f"resized_{Path(image_path).stem}.png"
    
    # 리사이즈된 이미지를 저장
    save_image(resized_image, output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: dot_resizer <image_path> <scale>")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        scale = float(sys.argv[2])
    except ValueError:
        print("Scale must be a number.")
        sys.exit(1)
    main(image_path, scale)
