import cv2
import time
import psutil
import os

def visualize_and_classify_images(image_paths, classify_function):
    process = psutil.Process(os.getpid())
    total_memory_used = 0
    pixel_art_count = 0
    non_pixel_art_count = 0
    start_time = time.perf_counter()

    for image_path in image_paths:
        start_memory = process.memory_info().rss

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            continue

        analysis_start_time = time.perf_counter()

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

        analysis_end_time = time.perf_counter()
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

    end_time = time.perf_counter()
    total_time = (end_time - start_time) * 1000  # ms 단위로 변환
    print(f"Total Memory used: {total_memory_used / (1024 * 1024):.2f} MB")
    print(f"Total Time taken: {total_time:.2f} ms")
    print(f"Total Pixel Art Images: {pixel_art_count}")
    print(f"Total Non-Pixel Art Images: {non_pixel_art_count}")
