import sys
import subprocess

def main():
    if len(sys.argv) < 4:
        print("일반 이미지 확대 입력 예시: python main.py normal <image_path> <scale> <quality>")
        print("도트 이미지 확대 입력 예시: python main.py pixel <image_path> <scale>")
        sys.exit(1)

    mode = sys.argv[1].lower()
    image_path = sys.argv[2]
    scale = sys.argv[3]

    if mode == 'normal':
        if len(sys.argv) < 5:
            print("일반 이미지 확대 입력 예시: python main.py normal <image_path> <scale> <quality>")
            sys.exit(1)
        quality = sys.argv[4].lower()
        if quality not in ['-bicubic', '-bilinear']:
            print("확대 방법은 반드시 '-bicubic' 또는 '-bilinear'이어야 합니다.")
            sys.exit(1)
        subprocess.run(["python", "normal_resizer.py", image_path, scale, quality])
    elif mode == 'pixel':
        subprocess.run(["python", "dot_resizer.py", image_path, scale])
    else:
        print("유효하지 않은 모드입니다. 'normal'혹은 'pixel'모드를 사용해주세요. ")
        sys.exit(1)

if __name__ == "__main__":
    main()
