# [NLP] 문장 간 유사도 측정
### 의미 유사도 판별(Semantic Text Similarity, STS)이란 두 문장이 의미적으로 얼마나 유사한지를 수치화하는 자연어처리 태스크입니다.
----

관계를 예측한다는 점에서 Textual Entailment (TE)와 헷갈릴 수 있습니다. 두 문제의 가장 큰 차이점은 ‘방향성’입니다. STS는 두 문장이 서로 동등한 양방향성을 가정하고 진행되지만, TE의 경우 방향성이 존재합니다. 예를 들어 자동차는 운송수단이지만, 운송수단 집합에 반드시 자동차만 있는 것은 아닙니다. 또한 출력 형태에 대해서도 차이가 있습니다. TE, STS 모두 관계 유사도에 대해 참/거짓으로 판단할 수 있지만, STS는 수치화된 점수를 출력할 수도 있습니다.

이처럼 STS의 수치화 가능한 양방향성은 정보 추출, 질문-답변 및 요약과 같은 NLP 작업에 널리 활용되고 있습니다. 실제 어플리케이션으로는 데이터 증강, 챗봇의 질문 제안, 혹은 중복 문장 탐지 등에 응용되고 있습니다.

우리는 STS 데이터셋을 활용해 두 문장의 유사도를 측정하는 AI모델을 구축할 것입니다. 유사도 점수와 함께 두 문장의 유사함을 참과 거짓으로 판단하는 참고 정보도 같이 제공하지만, 최종적으로 0과 5사이의 유사도 점수를 예측하는 것을 목적으로 합니다!

-----

## 데이터
### 평가 데이터의 50%는 Public 점수 계산에 활용되어 실시간 리더보드에 표기가 되고, 남은 50%는 Private 결과 계산에 활용되어 대회 종료 후 평가됩니다.
- 기본 데이터
  - 학습 데이터셋 9,324개
  - 검증 데이터셋 550개
  - 평가 데이터는 1,100개
- 시행착오 데이터 (try)
    - [데이터 증강] 1-1, 2번 적용
    - 학습 데이터셋 15,277개
    - 검증 데이터셋 1,689개
    - 평가 데이터셋 1,100개
- 최종 데이터 (final)
    - [데이터 증강] 1-2, 2번 적용
    - 학습 데이터셋 14,369개
    - 검증 데이터셋 1,500개
    - 평가 데이터셋 1,100개
---
## 모델
- 모델1
  - klue/roberta-small
  - batch size : ?
  - epoch : ?
- 모델2
  - monologg/koelectra-base-v3-discriminator
  - batch size : ?
  - epoch : ?
---
## 실험 결과

|모델|pearson score|설명(특이사항)|
|------|---|---|
|모델1|0.xx|blahblah|
|모델2|0.xx|blahblah|
---
## 명령어 예제
### Train
```
python main.py -m t -c base_config
```
### Continue Train
```
python main.py -m ct -s 'save_models/xlm-roberta-large_maxEpoch1_batchSize32_still-mountain-1/epoch=0-step=4203-val_pearson=0.9-val_loss=0.4.ckpt' -c base_config
```
### Inference
```
python main.py -m i -s 'save_models/xlm-roberta-large_maxEpoch1_batchSize32_still-mountain-1/epoch=0-step=4203-val_pearson=0.9-val_loss=0.4.ckpt' -c base_config
```
### WandB Sweep
```
python main.py -m e -c base_config
```
- 실행 후 반복 횟수 입력


---
