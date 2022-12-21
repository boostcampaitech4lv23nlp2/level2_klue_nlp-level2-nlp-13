import argparse
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import inference
import train
from omegaconf import OmegaConf


if __name__ == "__main__":
    """
    하이퍼 파라미터 등 각종 설정값을 입력받습니다
    터미널 실행 예시 : python3 run.py --batch_size=64 ...
    실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="custom_config")
    parser.add_argument("--mode", "-m", default="train")
    parser.add_argument(
        "--saved_model",
        "-s",
        default=None,
        help="저장된 모델의 파일 경로를 입력해주세요. 예시: saved_models/klue/roberta-small/epoch=?-step=?.ckpt 또는 save_models/model.pt",
    )
    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./config/{args.config}.yaml")
    if args.saved_model:
        config.path.saved_model = args.saved_model

    SEED = config.utils.seed
    pl.seed_everything(SEED, workers=True) # covers torch, numpy, random
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if args.mode == "train" or args.mode == "t":
        if config.k_fold.use_k_fold is True:
            assert config.k_fold.use_k_fold == isinstance(dataloader, kfoldDataloader), "Check your config again: Make sure `config.k_fold.use_k_fold` is compatible with `config.datalaoder.architecture`"
            if config.utils.on_step is False:
                assert config.utils.patience >= config.k_fold.num_folds, "The given value for `config.utils.patience` should be higher than the number of folds"
            train.train_cv(config)
        else:
            train.train(config)

    elif args.mode == "exp" or args.mode == "e":
        exp_count = int(input("실험할 횟수를 입력해주세요 "))
        train.sweep(config, exp_count)

    elif args.mode == "inference" or args.mode == "i":
        if args.saved_model is None:
            print("경로를 입력해주세요")
        else:
            inference.inference(args, config)

    elif args.mode == "ensemble":
        import ensemble
        ensemble.inference(args, config)
    
    elif args.mode == "all" or args.mode == "a":
        if args.saved_model is not None:
            print("Cannot input 'saved_model' for 'all' mode")

        train.train(config)
        inference.inference(args, config)

    else:
        print("모드를 다시 설정해주세요 ")
        print("train     : t,\ttrain")
        print("exp       : e,\texp")
        print("inference : i,\tinference")
        print("continue train : ct,\tcontinue train")
