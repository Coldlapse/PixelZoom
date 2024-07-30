import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from math import pi
from pathlib import Path
import time
import psutil
import os


# 1. 엣지 밀도 분석 (이미지 경계선의 밀도를 계산)
def analyze_edge_density(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    edge_density = np.sum(edges) / (gray_image.size)

    return edge_density


def classify_edge_density(image_path):
    edge_density = analyze_edge_density(image_path)
    if edge_density is None:
        return None

    edge_density_threshold = 18  # 임의의 기준값 설정
    is_pixel_art = edge_density > edge_density_threshold

    return is_pixel_art


###


# 2. 경계 및 계단 현상 길이 분석
def detect_jagged_edges(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    image = cv2.resize(image, (512, 512))

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    jagged_count = 0
    total_length = 0

    for contour in contours:
        for i in range(len(contour)):
            pt1 = contour[i][0]
            pt2 = contour[(i + 1) % len(contour)][0]
            total_length += np.linalg.norm(pt1 - pt2)
            if np.linalg.norm(pt1 - pt2) > 10:
                jagged_count += 1

    jagged_ratio = jagged_count / total_length if total_length > 0 else 0
    return jagged_ratio


def classify_jagged_edges(image_path):
    jagged_ratio = detect_jagged_edges(image_path)
    if jagged_ratio is None:
        return None

    jagged_threshold = 0.01  # 임의의 기준값 설정
    is_pixel_art = jagged_ratio > jagged_threshold

    return is_pixel_art


###


# 3. 이미지에서 모든 색상의 종류 개수를 파악
def count_unique_colors(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    b, g, r = cv2.split(image)
    out_in_32U_2D = (
        np.int32(b) << 16 | np.int32(g) << 8 | np.int32(r)
    )  # 비트 시프트를 통해 각 채널을 결합
    out_in_32U_1D = out_in_32U_2D.reshape(-1)  # 1D로 변환
    unique_colors = np.unique(out_in_32U_1D)  # 고유한 색상 값 찾기
    num_unique_colors = len(unique_colors)  # 고유한 색상 개수 계산

    return num_unique_colors


###


# 4. 엣지와 노이즈로 픽셀 여부 판단(현재 채택 중)
def detect_aliased_edges(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, None, None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi

    aliased_edges = (magnitude > 100) & (
        (angle % 45) < 10
    )  # 기울기 크기와 방향 기준 설정
    aliased_ratio = np.sum(aliased_edges) / magnitude.size

    total_edges = np.sum(magnitude > 100)
    aliased_edges_count = np.sum(aliased_edges)

    return aliased_ratio, aliased_edges, total_edges, aliased_edges_count


###


# 결과 시각화 및 픽셀 이미지 판단
def visualize_and_classify_images(image_paths, classify_function):
    process = psutil.Process(os.getpid())
    total_memory_used = 0
    pixel_art_count = 0
    non_pixel_art_count = 0
    start_time = time.time()

    for image_path in image_paths:
        start_memory = process.memory_info().rss

        if classify_function == classify_edge_density:
            is_pixel_art = classify_function(image_path)
        elif classify_function == classify_jagged_edges:
            is_pixel_art = classify_function(image_path)
        elif classify_function == classify_lines:
            is_pixel_art = classify_function(image_path)
        elif classify_function == detect_aliased_edges:
            aliased_ratio, aliased_edges, total_edges, aliased_edges_count = (
                classify_function(image_path)
            )
            if aliased_ratio is None or aliased_edges is None:
                continue
            noise_level = (
                (total_edges - aliased_edges_count) / total_edges
                if total_edges > 0
                else 0
            )
            is_pixel_art = noise_level < 0.5

        if is_pixel_art is None:
            continue

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            continue

        # plt.figure(figsize=(10, 10))
        # plt.subplot(1, 1, 1)
        # plt.title("Original Image")
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.show()

        end_memory = process.memory_info().rss
        memory_used = max(end_memory - start_memory, 0)
        total_memory_used += memory_used

        if is_pixel_art:
            pixel_art_count += 1
        else:
            non_pixel_art_count += 1

        print(f"Image: {image_path}")
        print("Is Pixel Art:", "Yes" if is_pixel_art else "No")
        print("")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Memory used: {total_memory_used / (1024 * 1024):.2f} MB")
    print(f"Total Time taken: {total_time:.2f} seconds")
    print(f"Total Pixel Art Images: {pixel_art_count}")
    print(f"Total Non-Pixel Art Images: {non_pixel_art_count}")


# 메인 함수
def main():
    image_paths = [
        "PixelZoom/image/dot/dot1.png",
        "PixelZoom/image/dot/dot2.png",
        "PixelZoom/image/dot/dot3.png",
        "PixelZoom/image/dot/dot4.png",
        "PixelZoom/image/dot/dot5.png",
        "PixelZoom/image/dot/dot6.png",
        "PixelZoom/image/dot/dot7.png",
        "PixelZoom/image/dot/dot8.png",
        "PixelZoom/image/dot/dot9.png",
        "PixelZoom/image/dot/dot10.png",
        "PixelZoom/image/normal/normal1.png",
        "PixelZoom/image/normal/normal2.png",
        "PixelZoom/image/normal/normal3.png",
        "PixelZoom/image/normal/normal4.png",
        "PixelZoom/image/normal/normal5.png",
        "PixelZoom/image/normal/normal6.png",
        "PixelZoom/image/normal/normal7.png",
        "PixelZoom/image/normal/normal8.png",
        "PixelZoom/image/normal/normal9.png",
        "PixelZoom/image/normal/normal10.png",
    ]

    visualize_and_classify_images(image_paths, classify_edge_density)
    # visualize_and_classify_images(image_paths, classify_jagged_edges)
    # visualize_and_classify_images(image_paths, classify_lines)
    # visualize_and_classify_images(image_paths, detect_aliased_edges)

    # 새로운 함수 사용 예제
    # for image_path in image_paths:
    #    num_colors = count_unique_colors(image_path)
    #    print(f"Image: {image_path} has {num_colors} unique colors")


if __name__ == "__main__":
    main()
