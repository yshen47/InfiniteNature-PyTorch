import copy
import shutil
from data.utils.utils import *
import os
from pytorch_lightning.loggers import WandbLogger
import signal
import datetime
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', help="config path",
                        default="configs/clevr-infinite.yaml")
    parser.add_argument('--experiment_name_suffix', help="",
                        default="debug")
    args = parser.parse_args()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    config = OmegaConf.load(args.config_path)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "image_logger": {
            "target": "data.utils.utils.ImageLogger",
            "params": {
                "batch_frequency": 500,
                "max_images": 4,
                "clamp": True
            }
        },
    }
    callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    # prepare log name
    log_name = args.experiment_name_suffix
    if config.log_keywords is not None:
        for keyword in config.log_keywords.split(','):
            keyword = keyword.strip()
            value = None
            curr_config = copy.deepcopy(config)
            for k in keyword.split('.'):
                curr_config = curr_config[k]
            value = curr_config
            log_name += f"_{k}_{value}"
    log_name += f"_{str(now)}"

    logdir = os.path.join("logs", log_name)
    os.makedirs(logdir, exist_ok=True)

    cfgdir = os.path.join(logdir, "configs")
    os.makedirs(cfgdir, exist_ok=True)
    shutil.copy(args.config_path, str(cfgdir) + "/config.yaml")

    data = instantiate_from_config(config.data)

    # config.model.params.data_config = config.data.params
    model = instantiate_from_config(config.model)
    model.logdir = logdir

    wandb_logger = WandbLogger(
        entity="generating_sfm",
        project='InfiniteNature',
        save_dir=logdir,
        name=logdir.split('/')[-1],
    )

    gpu_ids = []
    for gpu_id in config.model.gpu_ids.split(','):
        if gpu_id != '':
            gpu_ids.append(int(gpu_id))

    trainer_kwargs = dict()
    trainer_opt = argparse.Namespace(**{
        "gpus": 0 if -1 in gpu_ids else len(gpu_ids),
        "strategy": "ddp",
        "logger": wandb_logger,
        "devices": gpu_ids,
        "accelerator": 'gpu' if len(gpu_ids) > 0 else 'cpu'
    })
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    trainer_kwargs["callbacks"].append(CheckpointEveryNSteps(2500, os.path.join(logdir, "checkpoints", "last.ckpt")))
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

    trainer.num_sanity_val_steps = 2
    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb
            pudb.set_trace()

    signal.signal(signal.SIGUSR2, divein)

    trainer.fit(model, data)
