import cv2
import numpy as np
import time
import psutil
import os
import glob
from openpyxl import load_workbook

# 엑셀 파일 로드
file_path = "PixelZoom/Research_src/Result.xlsx"  # 작성할 엑셀 파일 경로
workbook = load_workbook(filename=file_path)
sheet = workbook.active  # 첫 번째 시트를 사용합니다.

# 총 메모리와 총 시간을 기록하기 위한 변수
total_memory_sum = 0
total_time_sum = 0


# 각 이미지의 분석 결과를 엑셀에 기록하는 함수
def record_results_to_excel(
    image_index, is_pixel_art, analysis_time, memory_used, cpu_usage, num_colors=None
):
    global total_memory_sum, total_time_sum  # 총 합계에 접근하기 위해 글로벌 변수를 사용

    row = image_index + 2  # 엑셀의 첫 번째 행(헤더)을 건너뛰기 위해 +2

    # 픽셀 아트 여부 기록 (H 열)
    sheet[f"H{row}"] = "p" if is_pixel_art else "n"

    # 분석 시간 기록 (I 열)
    sheet[f"I{row}"] = analysis_time
    total_time_sum += analysis_time  # 총 시간 합계에 추가

    # 메모리 사용량 기록 (J 열)
    sheet[f"J{row}"] = memory_used
    total_memory_sum += memory_used  # 총 메모리 사용량 합계에 추가

    # CPU 점유율 기록 (L 열)
    sheet[f"L{row}"] = cpu_usage

    # 엑셀 파일 저장
    workbook.save(filename=file_path)


# 메모리 사용량 측정 함수 개선
def get_memory_usage(process):
    # 메모리 사용량을 MB 단위로 반환합니다.
    memory_info = process.memory_info()
    rss_memory = memory_info.rss / (1024 * 1024)  # RSS 메모리 사용량(MB)
    return rss_memory


# 판별 알고리즘에 따른 결과 출력 및 엑셀 기록
def visualize_and_classify_images(image_paths, classify_function, start_index=0):
    global total_memory_sum, total_time_sum  # 전역 변수 사용

    process = psutil.Process(os.getpid())
    pixel_art_count = 0
    non_pixel_art_count = 0

    for index, image_path in enumerate(image_paths):
        start_memory = get_memory_usage(process)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            continue

        # CPU 사용률 측정 시작 전
        cpu_usage_start = process.cpu_percent(interval=None)

        analysis_start_time = time.time()

        if classify_function == is_pixelated:
            is_pixel_art = classify_function(image_path)
        else:
            is_pixel_art = classify_function(image)

        analysis_end_time = time.time()
        analysis_time = (
            analysis_end_time - analysis_start_time
        ) * 1000  # ms 단위로 변환

        # CPU 사용률 측정 종료 후
        cpu_usage_end = process.cpu_percent(interval=None)
        cpu_usage = cpu_usage_end - cpu_usage_start  # 작업 중 CPU 사용률 계산

        end_memory = get_memory_usage(process)
        memory_used = max(end_memory - start_memory, 0)

        num_colors = count_unique_colors(image)  # 고유 색상 개수 계산

        # 결과 기록
        record_results_to_excel(
            start_index + index,
            is_pixel_art,
            analysis_time,
            memory_used,
            cpu_usage,
            num_colors,
        )

        if is_pixel_art:
            pixel_art_count += 1
        else:
            non_pixel_art_count += 1

    print(f"Total RSS Memory used: {total_memory_sum:.2f} MB")
    print(f"Total Time taken: {total_time_sum:.2f} ms")
    print(f"Total Pixel Art Images: {pixel_art_count}")
    print(f"Total Non-Pixel Art Images: {non_pixel_art_count}")


# 특정 폴더 내 모든 이미지 경로를 가져오는 함수
def get_image_paths_from_folders(folder_paths, extensions=["*.png", "*.jpg", "*.jpeg"]):
    image_paths = []
    for folder_path in folder_paths:
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    return image_paths


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


def classify_aliased_edges(image):
    aliased_ratio, aliased_edges, total_edges, aliased_edges_count = (
        detect_aliased_edges(image)
    )
    noise_level = (
        (total_edges - aliased_edges_count) / total_edges if total_edges > 0 else 0
    )
    is_pixel_art = noise_level < 0.5
    return is_pixel_art


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


# 특정 폴더 내 모든 이미지 경로를 가져오는 함수
def get_image_paths_from_folders(folder_paths, extensions=["*.png", "*.jpg", "*.jpeg"]):
    image_paths = []
    for folder_path in folder_paths:
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    return image_paths


# 메인 함수
def main():
    folder_paths = [  # 분석할 이미지가 있는 폴더의 경로를 입력
        "PixelZoom/PixelZoom_src/image/dot",
        "PixelZoom/PixelZoom_src/image/normal",
    ]

    image_paths = get_image_paths_from_folders(folder_paths)

    # 엣지 밀도 분석에 따른 판별 - 비추천, 너무 정확도가 떨어짐
    # visualize_and_classify_images(image_paths, classify_edge_density)

    # (1) 경계 및 계단 현상 길이 분석에 따른 판별
    visualize_and_classify_images(image_paths, classify_jagged_edges)

    # (2) 히스토그램 임계값을 이용한 픽셀 아트 판별
    # visualize_and_classify_images(image_paths, is_pixelated)

    # (3) 엣지와 노이즈 분석에 따른 판별
    # visualize_and_classify_images(image_paths, classify_aliased_edges)


if __name__ == "__main__":
    main()
