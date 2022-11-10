import torch

import model.model as module_arch
from data_loader.data_loaders import Dataloader, KfoldDataloader


def new_instance(conf, config=None):  # sweep 부분 때문에 두번째 인자 추가

    if config is None:
        learning_rate = conf.train.learning_rate
    else:
        learning_rate = config.learning_rate

    dataloader = Dataloader(
        conf.model.model_name,
        conf.train.batch_size,
        conf.data.train_ratio,
        conf.data.shuffle,
        conf.path.train_path,
        conf.path.test_path,
        conf.path.predict_path,
        conf.data.swap,
    )

    # custom 모델 인지 확인
    model = module_arch.Model(
        conf.model.model_name,
        learning_rate,
        conf.train.loss,
        dataloader.new_vocab_size(),
        conf.train.use_frozen,
    )  # 새롭게 추가한 토큰 사이즈 반영

    return dataloader, model


def load_model(args, conf, dataloader: Dataloader, model):  # continue_train과 inference시에 모델을 불러오는 기능은 같기 때문에 메서드로 구현함
    # 불러온 모델이 저장되어 있는 디렉터리를 parsing함
    # ex) 'save_models/klue/roberta-small_maxEpoch1_batchSize32_blooming-wind-57'
    save_path = "/".join(args.saved_model.split("/")[:-1])

    # huggingface에 저장된 모델명을 parsing함
    # ex) 'klue/roberta-small'
    model_name = "/".join(args.saved_model.split("/")[1:-1]).split("_")[0]

    if args.saved_model.split(".")[-1] == "ckpt":
        model = model.load_from_checkpoint(args.saved_model)
    elif args.saved_model.split(".")[-1] == "pt" and args.mode != "continue train" and args.mode != "ct":
        model = torch.load(args.saved_model)
    else:
        exit("saved_model 파일 오류")

    conf.path.save_path = save_path + "/"
    conf.model.model_name = "/".join(model_name.split("/")[1:])
    return model, args, conf
