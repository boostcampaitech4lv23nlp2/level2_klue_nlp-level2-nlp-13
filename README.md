# KLUE-Relation Extraction

## 1️⃣ What's new
이 repo는 huggingface API 기반의 베이스라인 코드 수정판(dev_hf)과 Pytorch Lightning API 기반 템플렛(main)을 제공합니다. 템플렛은 Lightning trainer를 바탕으로 
- k-fold CV
- entity marker
- syllable tokenizer
- TAPT(Task-Adaptive Pre-Training)용 MLM(Masked Language Modeling)
- WandB logger
- 앙상블(logit/probability ensembling)
- confusion matrix 

등을 지원합니다.   

## 2️⃣ Introduction
문장의 단어(Entity)에 대한 속성과 관계를 예측하는 Task
- **input:** sentence, subject_entity, object_entity
- **output:** relation 30개 중 하나를 예측한 pred_label, 그리고 30개 클래스 각각에 대해 예측한 확률 probs
- **평가지표**
  - no_relation class를 제외한 micro F1 score
  - 모든 class에 대한 area under the precision-recall curve (AUPRC)
  - 2가지 metric으로 평가하며, **micro F1 score가 우선**시 된다.

## 3️⃣ 팀원 소개

김별희|이원재|이정아|임성근|정준녕|
:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/42535803?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/61496071?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/65378914?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/14817039?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/51015187?v=4' height=80 width=80px></img>
[Github](https://github.com/kimbyeolhee)|[Github](https://github.com/wjlee-ling)|[Github](https://github.com/jjeongah)|[Github](https://github.com/lim4349)|[Github](https://github.com/ezez-refer)

## 4️⃣ 데이터
```
Example)
sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
subject_entity: 썬 마이크로시스템즈
object_entity: 오라클

relation: 단체:별칭 (org:alternate_names)
```
- train.csv: 32470개 <br>
- test_data.csv: 총 7765개 (정답 라벨 blind = 100으로 임의 표현) <br>

## 5️⃣ 모델 설명
<details>
    <summary><b><font size="10">Project Tree</font></b></summary>
<div markdown="1">

```
.
```
</div>
</details>

## 6️⃣ How to Run
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
python main.py -c custom_config # ./config/custom_config.yaml 을 이용할 시 
python main.py -m t -c custom_config
python main.py --mode train --config custom_config
```

### 추가 학습
추가 학습을 하려면 기존 모델의 체크포인트를 config.path.resume_path에 추가하시면 됩니다 (기타 위와 동일).
```bash
python main.py -m t -c custom_config
```

### 추론
```bash
# 실행 시 prediction 폴더에 submission.csv가 생성됨
python main.py -m i -s "saved_models/klue/bert-base.ckpt"
python main.py -m i -s "saved_models/klue/bert-base.ckpt" -c custom_config
```

### (추가) 학습 + 추론
학습과 추론을 한 번에 실행할 수 있습니다. 추가 학습할 모델의 체크포인트를 config.path.resume_path에 입력하시고 다음을 실행하시면 추가로 학습 후 추론까지 진행합니다.
```bash
python main.py --mode all --config custom_config 
python main.py -m a -c custom_config
```

### 앙상블
```bash
# config.yaml 내 ensemble 항목에서 ckpt_paths (-> logit ensembling) 나 csv_paths (-> probability ensembling) 를 채운 후 다음 실행
python main.py --mode ensemble 
```

### base_config.yaml
- tokenizer - syllable: True 설정하면 음절 단위 토크나이저 적용 가능

## 7️⃣ etc
dict_label_to_num.pkl: 문자 label과 숫자 label로 표현된 dictionary, 총 30개 classes (class는 아래와 같이 정의 되어 있며, 평가를 위해 일치 시켜주시길 바랍니다.) pickle로 load하게 되면, 딕셔너리 형태의 정보를 얻을 수 있습니다.
```
with open('./dict_label_to_num.pkl', 'rb') as f:
    label_type = pickle.load(f)

{'no_relation': 0, 'org:top_members/employees': 1, 'org:members': 2, 'org:product': 3, 'per:title': 4, 'org:alternate_names': 5, 'per:employee_of': 6, 'org:place_of_headquarters': 7, 'per:product': 8, 'org:number_of_employees/members': 9, 'per:children': 10, 'per:place_of_residence': 11, 'per:alternate_names': 12, 'per:other_family': 13, 'per:colleagues': 14, 'per:origin': 15, 'per:siblings': 16, 'per:spouse': 17, 'org:founded': 18, 'org:political/religious_affiliation': 19, 'org:member_of': 20, 'per:parents': 21, 'org:dissolved': 22, 'per:schools_attended': 23, 'per:date_of_death': 24, 'per:date_of_birth': 25, 'per:place_of_birth': 26, 'per:place_of_death': 27, 'org:founded_by': 28, 'per:religion': 29}
```

30개의 class 정보
![class](https://user-images.githubusercontent.com/65378914/217735779-266b91ec-b41f-4c47-addd-8a9174531aac.png)

