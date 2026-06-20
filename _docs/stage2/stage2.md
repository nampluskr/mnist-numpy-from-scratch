---
tags: [docs, stage2, overview]
created: 2026-06-19
updated: 2026-06-19
---

# Stage 2 MNIST DataLoader

## 1. 개요

Stage 2는 로컬에 저장된 MNIST gz 파일을 읽어 학습에 사용할 수 있는 DataLoader를 구성하는 단계이다.
Stage 1에서 확정한 task 규약(`src/task.py`)을 Dataset 내부에서 호출하여 과제별 target 배열 변환을 처리하고, DataLoader는 `__len__`과 `__getitem__`을 구현한 Dataset이면 종류에 무관하게 수용하도록 설계한다.
이 Stage가 완료되면 Trainer와 Evaluator가 DataLoader를 통해 batch를 소비할 수 있는 데이터 파이프라인이 완성된다.

## 2. Phase 구성

### 2.1. Phase 2.1 MNIST raw data loading

`src/data/mnist.py`에 `load_mnist(split)` 함수를 구현한다.
로컬 경로의 4개 gz 파일(`train-images`, `train-labels`, `t10k-images`, `t10k-labels`)을 읽어 `(images, labels)` tuple을 반환한다.
`images`는 `(N, 28, 28)` uint8, `labels`는 `(N,)` uint8 원본 배열이다.

- [[phase2.1_mnist|Phase 2.1 MNIST raw data loading]]

### 2.2. Phase 2.2 Dataset 클래스 구현

`src/data/mnist.py`에 `MnistDataset` 클래스를 추가한다.
`load_mnist()`를 내부에서 호출한 뒤 이미지를 `(N, 784)` float32로 reshape 및 정규화하고, `transform_targets(labels, task)`로 과제별 target 배열을 준비한다.
`__getitem__(idx)`는 `(image, target)` 단일 샘플 tuple을 반환한다.

- [[phase2.2_dataset|Phase 2.2 Dataset 클래스 구현]]

### 2.3. Phase 2.3 DataLoader 구현

`src/data/dataloader.py`에 범용 `DataLoader` 클래스를 구현한다.
`__len__`과 `__getitem__`을 구현한 Dataset이면 모두 수용하며, `__iter__`는 `(images_batch, targets_batch)` tuple을 yield한다.
shuffle 옵션은 epoch 시작 시 인덱스를 무작위로 섞는 방식으로 처리한다.

- [[phase2.3_dataloader|Phase 2.3 DataLoader 구현]]

## 3. 주요 산출물

| 산출물 | 내용 |
|---|---|
| `src/data/mnist.py` | `load_mnist()` 함수 + `MnistDataset` 클래스 |
| `src/data/dataloader.py` | 범용 `DataLoader` 클래스 |
| `tests/stage2/` | mnist, dataset, dataloader 테스트 파일 3개 (54개 테스트 통과) |
