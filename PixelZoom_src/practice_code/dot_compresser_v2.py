import cv2
import numpy as np
import sys

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

# 이미지 경로 설정
image_path = 'ghost1.png'
output_path = 'ghostoutput_image.png'

# 이미지 로드
image = load_image(image_path)

if image is None:
    print(f"Error: Unable to load image at {image_path}")
    sys.exit(1)

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
    print("testing for Chunk Size " + str(element))
    new_height = int(height / element)
    new_width = int(width / element)
    minchunk_image = cv2.resize(restored_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    restored_minchunk_image = cv2.resize(minchunk_image, (width, height), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"Test/{element}restored.png", restored_image)
    cv2.imwrite(f"Test/{element}minchunk.png", minchunk_image)
    cv2.imwrite(f"Test/{element}restoredminchunk.png", restored_minchunk_image)
    if element == 1:
        print("이미지가 이미 MinChunk이거나 감지에 실패했습니다.")
        break
    if compare_images(restored_image, restored_minchunk_image):
        print("Image의 Chunk Size = " + str(element))
        minchunksize = element
        minchunk_height = new_height
        minchunk_width = new_width
        break

if minchunksize == -1:
    print("이미지의 MinChunk 탐색 실패! 이미지 축소 불가능!")
else:
    resize_scale = 0.2  # 축소할 배율
    resize_chunksize = round(minchunksize * resize_scale)
    if resize_chunksize == 0:
        resize_chunksize = 1
    final_resize_height = minchunk_height * resize_chunksize
    final_resize_width = minchunk_width * resize_chunksize
    final_resized_image = cv2.resize(minchunk_image, (final_resize_width, final_resize_height), interpolation=cv2.INTER_NEAREST)
    print(f"{resize_scale}배로 축소하라고 입력받은 결과, 가장 근접한 축소 배율은 {resize_chunksize / minchunksize}임으로 계산되었습니다.")
    print(f"원본 {width}*{height} 이미지를 {final_resize_width}*{final_resize_height} 이미지로 축소하였습니다.")
    cv2.imwrite("Test/Final_Resized_Image.png", final_resized_image)

# 복원된 이미지 저장
save_image(restored_image, output_path)

print(f"복원된 이미지가 {output_path}에 저장되었습니다.")
