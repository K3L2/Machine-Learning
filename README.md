# PCB_QM
PCB 품질 관리 시스템

[제품 분류 모델 h5 다운로드 링크](https://drive.google.com/file/d/1-0Y10fCKNyUmCp7HhN1mYojuOUkIjSTb/view?usp=sharing)

[제품 분류 클래스 json 다운로드 링크](https://drive.google.com/file/d/1-2ncMF7uDW_1nawRT_jCvwagBPpJ2YMR/view?usp=sharing)


## 모델 제작 기록

### 240718 이미지 분할 YOLO 모델

1. PSNR 방식으로 캐글 PCB 데이터에서 포토샵된 부분 추출, 최대 (256,256) 사이즈, 01 제품 대상
2. LabelIMG 로 YOLO 이미지 라벨링, (open_circuit, short) 대상
3. 결과
   1. 01 제품이 아닌 다른 사진의 결함도 추출 가능함을 확인
   2. 색상으로 학습했지만 흑백사진도 객체 인식 됨을 확인.
   3. 이미지가 (256,256) 사이즈 일 때만 정확도가 있었음. (적용 이미지에 전처리 과정이 필요함.) 

### 250719 원본 데이터 증강 YOLO 모델

1. 원본 라벨링 데이터에 데이터 증강 적용
   albumentations 라이브러리 사용 
   1. 랜덤 크롭 (Random Cropping)
   2. 스케일 변환 (Scaling)
   3. 좌우/상하 반전 (Flip)
   4. 회전 (Rotation)
   5. 밝기 및 대비 조절 (Brightness and Contrast Adjustment)
2. YOLO로 학습, epochs 30, 6개 라벨
3. 결과
   1. 모든 테스트 이미지에서 객체인식 불가
      1. 문제 원인 
         - 학습 배치를 살펴보니 이미지 증강하면서 라벨링이 제대로 되지 않았음.
      2. 개선방안
         - 증강 후 라벨링을 새로하는 코드 수정이 필요해보임

