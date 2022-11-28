# KLUE-Relation Extraction

문장의 단어(Entity)에 대한 속성과 관계를 예측하는 Task

```
Example)
sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
subject_entity: 썬 마이크로시스템즈
object_entity: 오라클

relation: 단체:별칭 (org:alternate_names)
```

<br/>

---

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
python main.py -c "custom_config" # ./config/custom_config.yaml 을 이용할 시 
python main.py -m t -c "custom_config"
python main.py --mode train --config "custom_config"
```

### 추가 학습
추가 학습을 하려면 기존 모델의 체크포인트를 path.resume_path에 추가하시면 됩니다 (기타 위와 동일).
```bash
python main.py -m t -c "custom_config"
```

### 추론
```bash
# 실행 시 prediction 폴더에 submission.csv가 생성됨
python main.py -m i -s "saved_models/klue/bert-base.ckpt"
python main.py -m i -s "saved_models/klue/bert-base.ckpt" -c "custom_config"
```
### base_config.yaml
- tokenizer - syllable: True 설정하면 음절 단위 토크나이저 적용 가능


### TODO
- [x] auprc warning 확인
- [ ] focal loss
- [ ] confusion matrix
