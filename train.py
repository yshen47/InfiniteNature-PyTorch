import copy
import shutil
from data.utils.utils import *
import os
from pytorch_lightning.loggers import WandbLogger
import signal


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', help="config path",
                        default="configs/google_earth.yaml")
    parser.add_argument('--experiment_name_suffix', help="",
                        default="debug")
    args = parser.parse_args()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    config = OmegaConf.load(args.config_path)

    # prepare log name
    log_name = args.experiment_name_suffix
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
    seed_everything(config.seed)

    data = DataLoader(instantiate_from_config(config.data),
                       batch_size=1,
                       num_workers=config.data.num_worker,
                       drop_last=False)

    config.model.params.data_config = config.data.params
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

    trainer = Trainer(gpus=0 if -1 in gpu_ids else len(gpu_ids),
                      strategy='ddp',
                      logger=wandb_logger,
                      devices=gpu_ids,
                      accelerator='gpu' if len(gpu_ids) > 0 else 'cpu'
                      )

    # def melk(*args, **kwargs):
    #     # run all checkpoint hooks
    #     if trainer.global_rank == 0:
    #         print("Summoning checkpoint.")
    #         ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    #         trainer.save_checkpoint(ckpt_path)


    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb
            pudb.set_trace()

    # signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    trainer.fit(model, data)
