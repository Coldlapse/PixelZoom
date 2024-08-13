import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import os


# 1. 엣지 밀도 분석 (이미지 경계선의 밀도를 계산)
def analyze_edge_density(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    edge_density = np.sum(edges) / (gray_image.size)
    return edge_density


def classify_edge_density(image):
    edge_density = analyze_edge_density(image)
    edge_density_threshold = 18  # 임의의 기준값 설정
    is_pixel_art = edge_density > edge_density_threshold
    return is_pixel_art


# 2. 경계 및 계단 현상 길이 분석
def detect_jagged_edges(image):
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


def classify_jagged_edges(image):
    jagged_ratio = detect_jagged_edges(image)
    jagged_threshold = 0.01  # 임의의 기준값 설정
    is_pixel_art = jagged_ratio > jagged_threshold
    return is_pixel_art


# 3. 이미지에서 모든 색상의 종류 개수를 파악
def count_unique_colors(image):
    b, g, r = cv2.split(image)
    out_in_32U_2D = (
        np.int32(b) << 16 | np.int32(g) << 8 | np.int32(r)
    )  # 비트 시프트를 통해 각 채널을 결합
    out_in_32U_1D = out_in_32U_2D.reshape(-1)  # 1D로 변환
    unique_colors = np.unique(out_in_32U_1D)  # 고유한 색상 값 찾기
    num_unique_colors = len(unique_colors)  # 고유한 색상 개수 계산
    return num_unique_colors


# 4. 엣지와 노이즈로 픽셀 여부 판단
def detect_aliased_edges(image):
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


# 5. 히스토그램 임계값으로 픽셀 아트 구분
def is_pixelated(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection using Canny
    edges = cv2.Canny(gray, 100, 200)

    # Calculate the number of edges
    edge_density = np.sum(edges) / edges.size

    # Calculate color histogram
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

    hist_b = hist_b / hist_b.max()
    hist_g = hist_g / hist_g.max()
    hist_r = hist_r / hist_r.max()

    # Check for spikes in histograms
    hist_spikes = np.sum(hist_b > 0.1) + np.sum(hist_g > 0.1) + np.sum(hist_r > 0.1)

    # Fourier Transform
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))

    # Analyze the magnitude spectrum for high-frequency components
    high_freq_count = np.sum(
        magnitude_spectrum
        > np.mean(magnitude_spectrum) + 2 * np.std(magnitude_spectrum)
    )

    # Heuristic thresholds
    edge_density_threshold = 0.1  # Arbitrary value, adjust based on empirical results
    hist_spike_threshold = 50  # Arbitrary value, adjust based on empirical results
    high_freq_threshold = 500  # Arbitrary value, adjust based on empirical results

    is_pixelated = (
        (edge_density > edge_density_threshold)
        and (hist_spikes < hist_spike_threshold)
        and (high_freq_count > high_freq_threshold)
    )

    return is_pixelated


# 판별 알고리즘에 따른 결과 출력
def visualize_and_classify_images(image_paths, classify_function):
    process = psutil.Process(os.getpid())
    total_memory_used = 0
    pixel_art_count = 0
    non_pixel_art_count = 0
    start_time = time.time()

    for image_path in image_paths:
        start_memory = process.memory_info().rss

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            continue

        analysis_start_time = time.time()

        if classify_function == classify_edge_density:
            is_pixel_art = classify_function(image)
        elif classify_function == classify_jagged_edges:
            is_pixel_art = classify_function(image)
        elif classify_function == detect_aliased_edges:
            aliased_ratio, aliased_edges, total_edges, aliased_edges_count = (
                classify_function(image)
            )
            noise_level = (
                (total_edges - aliased_edges_count) / total_edges
                if total_edges > 0
                else 0
            )
            is_pixel_art = noise_level < 0.5
        elif classify_function == is_pixelated:
            is_pixel_art = classify_function(image_path)

        analysis_end_time = time.time()
        analysis_time = (
            analysis_end_time - analysis_start_time
        ) * 1000  # ms 단위로 변환

        end_memory = process.memory_info().rss
        memory_used = max(end_memory - start_memory, 0)
        total_memory_used += memory_used

        if is_pixel_art:
            pixel_art_count += 1
        else:
            non_pixel_art_count += 1

        print(f"Image: {image_path}")
        print("Analysis Time: {:.2f} ms".format(analysis_time))
        print(f"Memory Used: {memory_used / (1024 * 1024):.2f} MB")
        print("Is Pixel Art:", "Yes" if is_pixel_art else "No")
        print("")

    end_time = time.time()
    total_time = (end_time - start_time) * 1000  # ms 단위로 변환
    print(f"Total Memory used: {total_memory_used / (1024 * 1024):.2f} MB")
    print(f"Total Time taken: {total_time:.2f} ms")
    print(f"Total Pixel Art Images: {pixel_art_count}")
    print(f"Total Non-Pixel Art Images: {non_pixel_art_count}")


# 메인 함수
def main():
    image_paths = [
        "PixelZoom/PixelZoom_src/image/dot/dot1.png",
        "PixelZoom/PixelZoom_src/image/dot/dot2.png",
        "PixelZoom/PixelZoom_src/image/dot/dot3.png",
        "PixelZoom/PixelZoom_src/image/dot/dot4.png",
        "PixelZoom/PixelZoom_src/image/dot/dot5.png",
        "PixelZoom/PixelZoom_src/image/dot/dot6.png",
        "PixelZoom/PixelZoom_src/image/dot/dot7.png",
        "PixelZoom/PixelZoom_src/image/dot/dot8.png",
        "PixelZoom/PixelZoom_src/image/dot/dot9.png",
        "PixelZoom/PixelZoom_src/image/dot/dot10.png",
        "PixelZoom/PixelZoom_src/image/normal/normal1.png",
        "PixelZoom/PixelZoom_src/image/normal/normal2.png",
        "PixelZoom/PixelZoom_src/image/normal/normal3.png",
        "PixelZoom/PixelZoom_src/image/normal/normal4.png",
        "PixelZoom/PixelZoom_src/image/normal/normal5.png",
        "PixelZoom/PixelZoom_src/image/normal/normal6.png",
        "PixelZoom/PixelZoom_src/image/normal/normal7.png",
        "PixelZoom/PixelZoom_src/image/normal/normal8.png",
        "PixelZoom/PixelZoom_src/image/normal/normal9.png",
        "PixelZoom/PixelZoom_src/image/normal/normal10.png",
    ]

    # 엣지 밀도 분석에 따른 판별
    # visualize_and_classify_images(image_paths, classify_edge_density)

    # 경계 및 계단 현상 길이 분석에 따른 판별
    visualize_and_classify_images(image_paths, classify_jagged_edges)

    # 엣지의 기울기와 노이즈 분석에 따른 판별
    # visualize_and_classify_images(image_paths, detect_aliased_edges)

    # 히스토그램 임계값을 이용한 픽셀 아트 판별
    # visualize_and_classify_images(image_paths, is_pixelated)

    # 각 이미지의 고유 색상 개수 출력
    # for image_path in image_paths:
    #    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #    if image is None:
    #        print(f"Error: Could not load image {image_path}")
    #        continue
    #    num_colors = count_unique_colors(image)
    #    print(f"Image: {image_path} has {num_colors} unique colors")


if __name__ == "__main__":
    main()
