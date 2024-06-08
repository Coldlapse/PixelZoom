import cv2
import sys
import numpy as np
from pathlib import Path

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

def preprocess_image(image):
    if image.shape[2] == 4:
        # 알파 채널을 추출
        alpha_channel = image[:, :, 3]
        # 알파 채널이 완전히 불투명한지 확인
        if np.all(alpha_channel == 255):
            # 알파 채널이 불투명하면 RGB로 처리
            image_no_bg = image[:, :, :3]
            gray = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        else:
            # 알파 채널이 투명한 영역이 있으면 이를 사용
            _, mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
            image_no_bg = image[:, :, :3]
    else:
        # RGB 이미지인 경우 회색조로 변환하여 하얀 배경 검출
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        image_no_bg = image

    return image_no_bg, mask

def extract_content(image, mask):
    # 외곽선을 찾아서 그 외곽선의 경계 상자를 구함
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    content = image[y:y+h, x:x+w]
    return content, (x, y, w, h)

def create_restored_image(content, bbox, original_image_shape, alpha_channel):
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

def resize_image(image, scale):

    # 이미지 전처리
    image_no_bg, mask = preprocess_image(image)

    # 내용물 추출
    content, bbox = extract_content(image_no_bg, mask)

    # 알파 채널 추출 (4채널 이미지인 경우에만)
    if image.shape[2] == 4:
        alpha_channel = image[:, :, 3]
    else:
        # 알파 채널이 없는 경우, 알파 채널을 흰색(완전히 불투명)으로 설정
        alpha_channel = np.ones(image.shape[:2], dtype=np.uint8) * 255

    # 내용물을 포함하는 새 이미지 생성 (투명 배경 유지)
    restored_image = create_restored_image(content, bbox, image.shape, alpha_channel)
    
    oheight, owidth = image.shape[:2]    
    height, width = restored_image.shape[:2]

    min_val = min(height, width)
        
    # 공약수를 저장할 리스트 초기화
    common_divisors = []

    # 1부터 두 정수 중 작은 값까지 반복하여 공약수 찾기
    for i in range(1, min_val + 1):
        if height % i == 0 and width % i == 0:
            common_divisors.append(i)


    minchunksize = -1
    for element in reversed(common_divisors):
        #print("testing for Chunk Size " + str(element))
        new_height = int(height / element)
        new_width = int(width / element)
        minchunk_image = cv2.resize(restored_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        restored_minchunk_image = cv2.resize(minchunk_image, (width, height), interpolation=cv2.INTER_NEAREST)
        if element == 1:
            #print("이미지가 이미 MinChunk이거나 감지에 실패했습니다.")
            break
        if compare_images(restored_image, restored_minchunk_image) == True:
            print("Image의 Chunk Size = " + str(element))
            minchunksize = element
            minchunk_height = new_height
            minchunk_width = new_width
            break

    if minchunksize == -1:
        print("이미지의 MinChunk 탐색에 실패하여, 가장 근접한 정수배로 이미지를 변경합니다. 이미지 축소는 불가능합니다.")
        if scale < 1:
            print("MinChunk가 탐색되지 않은 이미지에 대한 이미지 축소는 불가능합니다.")
            return None
        nearest_scale = int(np.round(scale))
        new_size = (owidth * nearest_scale, oheight * nearest_scale)
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
        return resized_image
        
    else:
        resize_scale = scale # 축소할 배율
        resize_chunksize = round(minchunksize * resize_scale)
        if resize_chunksize == 0:
            resize_chunksize = 1
        final_resize_height = minchunk_height * resize_chunksize
        final_resize_width = minchunk_width * resize_chunksize
        final_resized_image = cv2.resize(minchunk_image, (final_resize_width, final_resize_height), interpolation=cv2.INTER_NEAREST)
        print(f"{resize_scale}배로 변경하라고 입력받은 결과, 가장 근접한 변경 배율은 {resize_chunksize / minchunksize}임으로 계산되었습니다.")
        print(f"원본 {width}*{height} 이미지를 {final_resize_width}*{final_resize_height} 이미지로 변경하였습니다.")
        return final_resized_image


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
    if type(resized_image) is type(None):
        print("이미지 사이즈 변경에 실패하였습니다. 축소가 불가능한 이미지에 대해 1보다 낮은 배율을 입력하였을 수 있습니다.")
    else:
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
