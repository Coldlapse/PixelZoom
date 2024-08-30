import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from math import pi
from pathlib import Path

#엣지 밀도 분석 (이미지 경계선의 밀도를 계산)
def analyze_edge_density(image_path):
    # 이미지 읽기 및 크기 조정
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    image = cv2.resize(image, (512, 512))

    # 그레이스케일로 변환 후 엣지 검출
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    edge_density = np.sum(edges) / (gray_image.size)

    return edge_density

def classify_edge_density(image_path):
    edge_density = analyze_edge_density(image_path)
    if edge_density is None:
        return

    edge_density_threshold = 18  # 임의의 기준값 설정
    is_pixel_art = edge_density > edge_density_threshold

    print(f"Image: {image_path}")
    print(f"Edge Density: {edge_density}")
    print(f"Is Pixel Art (based on edge density): {'Yes' if is_pixel_art else 'No'}")
    print("")

 #이미지 분류
for image_path in image_paths:
    classify_edge_density(image_path)


#경계 및 계단 현상 분석
def detect_jagged_edges(image_path):
    # 이미지 읽기 및 크기 조정
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    image = cv2.resize(image, (512, 512))

    # 그레이스케일로 변환 후 엣지 검출
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)

    # 엣지 이미지에서 경계 검출
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    jagged_count = 0
    total_length = 0

    for contour in contours:
        for i in range(len(contour)):
            pt1 = contour[i][0]
            pt2 = contour[(i + 1) % len(contour)][0]
            total_length += np.linalg.norm(pt1 - pt2)
            if np.linalg.norm(pt1 - pt2) > 10:  # 경계선이 길다면 계단 현상이 있을 가능성 큼
                jagged_count += 1

    jagged_ratio = jagged_count / total_length if total_length > 0 else 0
    return jagged_ratio

def classify_image(image_path):
    jagged_ratio = detect_jagged_edges(image_path)
    
    if jagged_ratio is None:
        return

    jagged_threshold = 0.01  # 임의의 기준값 설정
    is_pixel_art = jagged_ratio > jagged_threshold

    print(f"Image: {image_path}")
    print(f"Jagged Edge Ratio: {jagged_ratio}")
    print(f"Is Pixel Art: {'Yes' if is_pixel_art else 'No'}")
    print("")

#수평 및 수직 라인의 개수로 분석
def count_lines(img):
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 노이즈 제거를 위한 필터링
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    # Canny 엣지 검출
    edges = cv2.Canny(gray, 50, 150)
    # Hough Line Transform을 사용하여 직선 검출
    lines = cv2.HoughLines(edges, rho=1, theta=pi/180, threshold=50)
    
    if lines is None:
        return 0, 0
    
    # 각도 계산
    angles = lines.squeeze()[:, 1] * 180 / pi
    # 수평 및 수직 라인의 각도 범위 설정
    horizontal_vertical_lines = np.sum(
        np.logical_or.reduce((
            np.abs(angles) < 1,
            np.abs(angles - 90) < 1,
            np.abs(angles - 180) < 1,
            np.abs(angles - 270) < 1
        ))
    )
    
    total_lines = len(angles)
    return horizontal_vertical_lines, total_lines

def classify_images(image_paths):
    results = []

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            continue
        
        hv_lines, total_lines = count_lines(img)
        hv_ratio = hv_lines / total_lines if total_lines > 0 else 0
        
        is_pixel_art = hv_ratio > 0.3  # 임계값 0.5는 수평 및 수직 라인이 전체 라인의 50%를 초과할 때 픽셀 아트로 판단
        results.append((image_path, hv_lines, total_lines, hv_ratio, is_pixel_art))
    
    # 결과 출력
    print("\nImage Classification Results:")
    for path, hv_lines, total_lines, hv_ratio, is_pixel_art in results:
        print(f"Image: {path}")
        print(f"Horizontal/Vertical Lines: {hv_lines}")
        print(f"Total Lines: {total_lines}")
        print(f"HV Ratio: {hv_ratio:.2f}")
        print(f"Is Pixel Art: {'Yes' if is_pixel_art else 'No'}")
        print("")

##임계값으로 하는 알고리즘
def is_dot_image(image_path, edge_threshold=0.1):
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    
    edges = cv2.Canny(image, 100, 200)  # 임계값은 조절하면 됨

   
    total_pixels = image.shape[0] * image.shape[1]
    edge_pixels = cv2.countNonZero(edges)

    
    edge_density = edge_pixels / total_pixels

    #
    if edge_density > edge_threshold:
        return True
    else:
        return False

# 테스트 이미지 경로
#image_path = 'dot_image.jpg' 


#if is_dot_image(image_path):
    print("it is dot")
#else:
    print("it is not dot image")

##K-means 군집화 사용
def is_dot_image(image_path, cluster_threshold=3, pixel_threshold=0.9):
    image = cv2.imread(image_path)

    pixels = image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=cluster_threshold)
    kmeans.fit(pixels)

  
    centers = kmeans.cluster_centers_

   
    unique_colors = len(np.unique(centers, axis=0))

 
    if unique_colors <= cluster_threshold:
      
        counts = np.bincount(kmeans.labels_)
        pixel_ratios = counts / len(pixels)

       
        if np.max(pixel_ratios) >= pixel_threshold:
            return True

    return False

##히스토그램 임계값으로 도트 구분
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
    high_freq_count = np.sum(magnitude_spectrum > np.mean(magnitude_spectrum) + 2 * np.std(magnitude_spectrum))
    
    # Heuristic thresholds
    edge_density_threshold = 0.1  # Arbitrary value, adjust based on empirical results
    hist_spike_threshold = 50    # Arbitrary value, adjust based on empirical results
    high_freq_threshold = 500    # Arbitrary value, adjust based on empirical results
    
    is_pixelated = (edge_density > edge_density_threshold) and (hist_spikes < hist_spike_threshold) and (high_freq_count > high_freq_threshold)
    
    return is_pixelated


#픽셀 블록 structure 분석( 색상의 일관성으로 파악 )
def pixel_block_analysis(image_path, block_size=8):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    height, width, _ = image.shape
    blocks = (height // block_size) * (width // block_size)
    
    variance_sum = 0
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i + block_size, j:j + block_size]
            variance_sum += np.var(block)
    
    avg_variance = variance_sum / blocks
    return avg_variance

#색상 팔레트 제한 분석 ( 제한된 색상을 사용했는지로 파악 )
def classify_pixel_block(image_path, variance_threshold=10):
    avg_variance = pixel_block_analysis(image_path)
    if avg_variance is None:
        return
    is_pixel_art = avg_variance < variance_threshold
    return is_pixel_art


##알고리즘을 결합하여 n개중 m개 이상의 값이 True일때 픽셀이다를 나타내는 함수
def combined_classify_image(image_path):
    edge_density_result = classify_edge_density(image_path)
    jagged_edges_result = classify_jagged_edges(image_path)
    lines_result = classify_lines(image_path)
    pixelated_result = is_pixelated(image_path)

    results = [edge_density_result, jagged_edges_result, lines_result, pixelated_result]
    final_decision = sum(results) > 2  # 네 가지 기준 중 세 가지 이상이 True이면 도트 이미지로 판단

    print(f"Image: {image_path}")
    print(f"Edge Density Result: {edge_density_result}")
    print(f"Jagged Edges Result: {jagged_edges_result}")
    print(f"Lines Result: {lines_result}")
    print(f"Pixelated Result: {pixelated_result}")
    print(f"Is Pixel Art: {'Yes' if final_decision else 'No'}")
    print("")


#######
#실제로 채택할 도트 구분 알고리즘, 이미지의 엣지를 표현했을 때, 발생하는 노이즈의 정도로 픽셀 여부 파악

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
def visualize_and_classify_images(image_paths):
    for image_path in image_paths:
        # 계단 현상 감지
        aliased_ratio, aliased_edges, total_edges, aliased_edges_count = detect_aliased_edges(image_path)
        if aliased_ratio is None or aliased_edges is None:
            continue

        # 이미지 읽기
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            continue

        # 그레이스케일로 변환
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 결과 시각화
        aliased_image = np.zeros_like(gray_image)
        aliased_image[aliased_edges] = 255

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 2, 2)
        plt.title('Aliased Edges')
        plt.imshow(aliased_image, cmap='gray')
        plt.show()

        # 픽셀 이미지 여부 판단
        noise_level = (total_edges - aliased_edges_count) / total_edges if total_edges > 0 else 0
        is_pixel_art = noise_level < 0.5

        print(f"Image: {image_path}")
        print(f"Aliased Ratio: {aliased_ratio:.4f}")
        print(f"Noise Level: {noise_level:.4f}")
        print("Is Pixel Art:", "Yes" if is_pixel_art else "No")
        print("")

# 이미지 경로 목록
image_paths = [
    'image/dot/dot1.png',
    'image/dot/dot2.png',
    'image/dot/dot3.png',
    'image/dot/dot4.png',
    'image/dot/dot5.png',
    'image/dot/dot6.png',
    'image/dot/dot7.png',
    'image/dot/dot8.png',
    'image/dot/dot9.png',
    'image/dot/dot10.png',

    'image/normal/normal1.png',
    'image/normal/normal2.png',
    'image/normal/normal3.png',
    'image/normal/normal4.png',
    'image/normal/normal5.png',
    'image/normal/normal6.png',
    'image/normal/normal7.png',
    'image/normal/normal8.png',
    'image/normal/normal9.png',
    'image/normal/normal10.png',
]

# 이미지 분류 및 시각화
visualize_and_classify_images(image_paths)