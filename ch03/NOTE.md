# CH03. 고급 컴퓨터 비전: 이미지에서 특징 감지하기

> 모델을 한 단계 더 발전시키려면?

- 이미지에 있는 특징을 감지하고 이해하는 능력을 갖추어야 한다.
- 이를 위해 컨볼루션 신경망(Convolutional Neural Network, CNN)을 사용한다.

## 3.1 합성곱 (Convolution)

- 합성곱은 단순한 가중치의 필터로서 픽셀에 가중치를 곱해서 새로운 픽셀 값을 만듭니다.

## 3.2 풀링 (Pooling)

- 풀링은 이미지 안에 있는 콘텐츠의 의미를 보존하면서 이미지의 픽셀을 줄이는 과정 (다운 샘플링)
- 최대 풀링(Max Pooling)과 평균 풀링(Average Pooling)이 있다.