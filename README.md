"# PixelZoom"

# 07/31

- all_pixel_distinguish_code 추가

## 픽셀 이미지 판별 알고리즘 정립

# 사용 알고리즘

- 1. 엣지 밀도 분석
- 2. 경계 및 계단현상 길이 분석
- 3. 고유 색상 개수 분석 - 현재 샘플로는 변별력이 너무 없기에, 각 이미지 별로 고유 색상을 구해서 출력하는 것까지만 구현
- 4. 엣지 중 수평 및 수직 기울기 비율 분석(현재 채택 중)

# 비교 기준

- 주어진 이미지들을 모두 계산하는 데에 걸리는 총 시간(메모리)을 기준으로 한다

1. 연산 시간
2. 메모리 사용량
3. 정확도? (픽셀과 논픽셀 이미지를 나눌 정확한 기준이 없기에 아직 보류)
4. CPU 및 GPU 사용량? (계속 예상을 벗어나는 이상한 결과가 나와서 좋은 방식을 찾는 중, 추후 수정)
