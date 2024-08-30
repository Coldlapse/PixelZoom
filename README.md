# PixelZoom

2024-1학기 동국대학교 인간컴퓨터상호작용 팀 설계과제 (5조)

## 무슨 프로젝트인가요?
- 일반적으로 픽셀 아트(도트 이미지)라고 불리우는 장르의 이미지는, 그 이미지의 확대나 축소를 할 때 "Nearest Neighbor" 보간법을 적용하여야 그 의미를 잃지 않고 확대 축소됨이 널리 알려져 있습니다.
- 이 프로시저는 후술할 세 모듈을 통해 구동되는 이미지 확대-축소기로, 구체적으로는 **입력받은 이미지가 픽셀 아트인지 아닌지 구분하고, 그 형태에 알맞는 리사이징을 실행** 하는 것을 목적으로 합니다.

## 실행 방법
<pre><code> python PixelZoom.py \<image_path> \<scale> [-speed | -quality] </code></pre>

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

# 08/14

- Research_src 폴더의 all_pixel_distinguish_code 수정
  - 이제 일일히 이미지 경로를 입력할 필요 없이, 이미지가 담긴 폴더 경로를 입력하는 것으로 판별 가능
- 기존 픽셀 10개, 일반 이미지 10개를 통해 실험하여 그 결과를 엑셀에 작성하는 것까지 확인. 엑셀에 작성할 땐 행을 지정해줘야 함(H는 IP1, I는 IP1의 시간, ...)
- 문제점 : 일부 이미지의 판별 과정에서 메모리가 0으로 표현되는 문제가 있음. 또한, CPU 점유율을 확인하기 위해 Psutil 라이브러리를 사용했는데, 결과가 음수가 나오는 등 정상적이지 않다. CPU 점유율을 더 자세히 확인할 수 있는 추가 수단 모색 필요

# 사용 알고리즘
- 3개의 알고리즘을 골라 진행한다면 2, 3, 4로 진행할 듯 함
- 1. 엣지 밀도 분석(비추천)
- 2. 경계 및 계단현상 길이 분석
- 3. 히스토그램을 통한 색상 변화 추적
- 4. 엣지 중 수평 및 수직 기울기 비율 분석(현재 채택 중)

 

  
