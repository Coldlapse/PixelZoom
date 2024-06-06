import sys
import subprocess
import cv2
import numpy as np

def detect_aliased_edges(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, None, None

    # 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel 연산자를 사용하여 x, y 방향의 기울기 계산
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # 기울기의 크기 계산
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 기울기의 방향 계산
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi

    # 계단 현상 감지를 위해 기울기와 방향의 불연속성을 분석
    aliased_edges = (magnitude > 100) & ((angle % 45) < 10)  # 기울기 크기와 방향 기준 설정

    # 계단 현상이 감지된 픽셀의 비율 계산
    aliased_ratio = np.sum(aliased_edges) / magnitude.size

    # 전체 엣지 픽셀 수와 계단 현상 픽셀 수
    total_edges = np.sum(magnitude > 100)
    aliased_edges_count = np.sum(aliased_edges)

    return aliased_ratio, aliased_edges, total_edges, aliased_edges_count

# 결과 시각화 및 픽셀 이미지 판단
def visualize_and_classify_images(image_path):
    # 계단 현상 감지
    aliased_ratio, aliased_edges, total_edges, aliased_edges_count = detect_aliased_edges(image_path)
    if aliased_ratio is None or aliased_edges is None:
        return -1

    # 이미지 읽기
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return -1

    # 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 결과 시각화
    aliased_image = np.zeros_like(gray_image)
    aliased_image[aliased_edges] = 255

    # 픽셀 이미지 여부 판단
    noise_level = (total_edges - aliased_edges_count) / total_edges if total_edges > 0 else 0
    is_pixel_art = noise_level < 0.5

    return is_pixel_art

def main():
    if len(sys.argv) < 3:
        print("사용법 : python PixelZoom.py <image_path> <scale> [-speed | -quality]")
        sys.exit(1)

    image_path = sys.argv[1]
    mode = visualize_and_classify_images(image_path)
    scale = sys.argv[2]

    if mode == 0:
        if len(sys.argv) < 4:
            print("속도 중시 / 품질 중시 모드가 선택되지 않아 기본값인 품질 중시 모드를 선택합니다.")
            quality = '-bicubic'
        else:
            quality = sys.argv[3].lower()
        if quality not in ['-speed', '-quality']:
            print("확대 모드는 반드시 '-speed' 또는 '-quality'이어야 합니다.")
            sys.exit(1)
        subprocess.run(["python", "normal_resizer.py", image_path, scale, quality])
    elif mode == 1:
        subprocess.run(["python", "dot_resizer_v3.py", image_path, scale])
    else:
        print("이미지를 불러오는 것에 실패했습니다. 이미지를 확인해주세요.")
        sys.exit(1)

if __name__ == "__main__":
    main()
