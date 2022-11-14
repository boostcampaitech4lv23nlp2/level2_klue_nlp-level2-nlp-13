# KLUE-Relation Extraction

### 가상환경
```bash
# 가상환경 생성
python -3.8 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 필요 라이브러리 설치
pip install -r requirements.txt
```

### 학습
```bash
python train.py --config=base_config
```

### inference
```bash
# 실행 시 prediction 폴더에 submission.csv가 생성됨
python inference.py
```