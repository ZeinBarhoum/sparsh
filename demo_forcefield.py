import os
import hydra
import numpy as np
import torch
import torch.utils.data as data
from omegaconf import OmegaConf, DictConfig
from pprint import pprint
import json


def demo(cfg: DictConfig):
    _GLOBAL_SEED = cfg.seed
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Instantiating model <{cfg.task._target_}>")
    task_name = cfg.experiment_name
    path_checkpoints = cfg.paths.output_dir + "/checkpoints/"
    eval_ckpts = sorted(os.listdir(path_checkpoints))
    eval_ckpts = [ckpt for ckpt in eval_ckpts if ckpt[-4:] == ".pth"]
    last_ckpt = eval_ckpts[-1]  # -3

    cfg.task.checkpoint_task = f"{path_checkpoints}/{last_ckpt}"
    model = hydra.utils.instantiate(cfg.task)
    print(f"Testing {task_name}  - {last_ckpt}")
    demo_partial = hydra.utils.instantiate(cfg.test.demo)
    demo = demo_partial(device=device, module=model)

    demo.set_test_params(
        task=task_name,
        sensor=cfg.sensor,
        ckpt=last_ckpt,
        dataset_name=None,
        path_outputs=cfg.test.path_outputs,
        config=cfg,
    )

    demo.init()
    demo.run_model()
    print("*** Demo finished ***")


@hydra.main(version_base="1.3", config_path="config")
def main(cfg: DictConfig):
    print(cfg)
    exp_name = f"{cfg.sensor}_{cfg.task_name}_{cfg.ssl_name}_vit{cfg.ssl_model_size}_{cfg.train_data_budget}"
    path_outputs = cfg.paths.output_dir
    path_ckpt_encoders = cfg.task.checkpoint_encoder

    for exp in os.listdir(path_outputs):
        if exp_name in exp and exp[0:4] != "2024":
            path_outputs = f"{path_outputs}/{exp}"
            break

    exp_config = f"{path_outputs}/config.yaml"

    test_cfg = cfg.test.copy()
    data = cfg.data.copy()
    cfg = OmegaConf.load(exp_config)
    cfg.data = data
    cfg.test = test_cfg
    cfg.paths.output_dir = path_outputs
    cfg.task.checkpoint_encoder = path_ckpt_encoders

    demo(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

