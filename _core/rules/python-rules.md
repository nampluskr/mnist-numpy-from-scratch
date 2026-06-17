---
tags: [project, rules]
created: 2026-06-15
updated: 2026-06-15
---

# python-rules.md

이 프로젝트에서 Python 코드를 작성할 때 적용되는 통합 규칙 및 형식 지침이다.

## 1. 네이밍 규칙

### 1.1. 식별자 네이밍

PEP8 기반이며, 멤버 변수에 접두사를 붙이지 않고 private 변수에는 단일 언더스코어(`_`)만 허용한다.

| 항목 | 규칙 | 예시 |
|---|---|---|
| 변수명 | `snake_case` | `backbone_name`, `num_epochs` |
| 함수명 | `snake_case` | `get_trainer`, `load_backbone_weights` |
| 클래스명 | `PascalCase` | `QuadDataset`, `BaseTrainer` |
| 상수 | `UPPER_SNAKE_CASE` | `MAX_EPOCHS`, `DEFAULT_IMG_SIZE` |
| 멤버 변수 접두사 | 없음 | `self.name` (`m_name`, `name_` 금지) |
| private 접두사 | `_` (접근 제어 목적만) | `self._cache` |

### 1.2. 도메인 용어 통일

split 명칭 혼용과 모델명·백본명 혼용을 방지하기 위해 아래 용어로 통일한다.

| 잘못된 용어 | 올바른 용어 | 이유 |
|---|---|---|
| `val` | `valid` | split 명칭 통일 |
| `model_name` | `backbone_name` | 모델명과 백본명 구분 |

## 2. 코드 스타일

### 2.1. 경로 표기

Python 코드 내 경로는 `os.path` 방식을 사용한다. `pathlib.Path`는 사용하지 않는다.

```python
# Correct
import os
path = os.path.join("data", "train", "images")
base = os.path.dirname(os.path.abspath(__file__))

# Incorrect
from pathlib import Path
path = Path("data") / "train" / "images"
```

### 2.2. 세로 정렬 금지

등호(`=`), 콜론(`:`)의 세로 정렬(vertical alignment)은 유지보수 비용을 높이므로 금지한다.

```python
# Bad
backbone_name = "resnet50"
img_size      = 224
num_epochs    = 10

# Good
backbone_name = "resnet50"
img_size = 224
num_epochs = 10
```

dict 리터럴도 동일하게 적용한다.

```python
# Bad
config = {
    "backbone" : "resnet50",
    "img_size" : 224,
    "epochs"   : 10,
}

# Good
config = {
    "backbone": "resnet50",
    "img_size": 224,
    "epochs": 10,
}
```

### 2.3. 함수 인자 전달

`**kwargs` 언팩킹 대신 명시적 인자 전달을 사용한다. IDE의 타입 추론과 오류 감지를 위해서이다.

```python
# Bad
trainer_kwargs = {"backbone_dir": backbone_dir, "img_size": img_size}
trainer = TrainerClass(**trainer_kwargs)

# Good
trainer = TrainerClass(backbone_dir=backbone_dir, img_size=img_size)
```

## 3. 코드 주석

### 3.1. 주석 언어

코드 내 주석(comment), docstring, type hint는 영어로만 작성한다. AI CLI는 어떤 경우에도 한국어 주석을 작성하지 않는다.

파일 첫 줄에는 파일명과 한 줄 영어 설명을 작성한다.

```python
# trainer.py: Wrapper class for training loop with early stopping support.
mean = x.mean(dim=0)
```

### 3.2. 주석 작성 원칙

주석은 핵심 코드 블록의 경계를 구분하거나 비직관적인 동작의 이유를 설명하는 용도에 한해 작성한다.

```python
# Build output path using stage and backbone name
output_path = os.path.join(output_dir, f"{stage}-{backbone_name}.pth")

# Load pretrained weights before replacing head to avoid shape mismatch
load_backbone_weights(model, backbone_dir)
replace_head(model, num_outputs=8)
```

## 4. 코드 블록

### 4.1. 언어 명시

코드 블록은 항상 언어를 명시한다. 언어를 특정할 수 없는 경우 `text`를 사용한다.

````markdown
```python
```bash
```powershell
```cpp
```json
```yaml
```markdown
```text
````

### 4.2. 실행 가능한 명령어

셸 명령어는 실제 실행 가능한 형태로 작성하며, 필요한 경우 예상 출력을 함께 제시한다.

```bash
conda activate pytorch_env
python train.py --config config.yaml
```

```text
[Expected output]
Epoch 1/50: loss=0.4231, acc=0.8120
```

## 5. 타입 힌트 및 독스트링

타입 힌트와 독스트링은 요청 시에만 추가한다. 기본적으로 생략하며, 명시적 요청이 있을 때만 아래 형식으로 작성한다.

```python
# Default - no type hints, no docstring
def get_config(model_type, dataset, category, num_epochs):
    config = {
        "model_type": model_type,
        "dataset": dataset,
        "category": category,
        "num_epochs": num_epochs,
    }
    return config

# Only when explicitly requested
def get_config(model_type: str, dataset: str, category: str, num_epochs: int) -> dict:
    """Build training config.

    Args:
        model_type: Registered model key (e.g., "padim", "stfpm").
        dataset: Dataset name (e.g., "mvtec").
        category: Category name within dataset.
        num_epochs: Number of training epochs.

    Returns:
        Config dict.
    """
    ...
```

## 6. 출력 형식

상태 출력 시 이모지를 사용하지 않으며 ASCII 텍스트로만 상태를 표시한다.

```python
# Bad
print(f"  {'완료 OK' if ok else '실패 FAIL'}")

# Good
print(f"  {'[OK]' if ok else '[FAIL]'} {lec_id}")
print(f"  {'[OK]' if exists else '[--]'} {fname}")
```
