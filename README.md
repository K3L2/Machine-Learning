# PCB_QM
PCB 품질 관리 시스템

[제품 분류 모델 h5 다운로드 링크](https://drive.google.com/file/d/1-0Y10fCKNyUmCp7HhN1mYojuOUkIjSTb/view?usp=sharing)

[제품 분류 클래스 json 다운로드 링크](https://drive.google.com/file/d/1-2ncMF7uDW_1nawRT_jCvwagBPpJ2YMR/view?usp=sharing)


## 모델 제작 기록

### 240718 이미지 분할 YOLO 모델

1. PSNR 방식으로 캐글 PCB 데이터에서 포토샵된 부분 추출, 최대 (256,256) 사이즈, 01 제품 대상
2. LabelIMG 로 YOLO 이미지 라벨리, (open_circuit, short) 대상
3. 결과
   1. 01 제품이 아닌 다른 사진의 결함도 추출 가능함을 확인
   2. 색상으로 학습했지만 흑백사진도 객체 인식 됨을 확인.
   3. 이미지가 (256,256) 사이즈 일 때만 정확도가 있었음. (적용 이미지에 전처리 과정이 필요함.) 