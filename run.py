import os
import wandb
from src.sac import SACDiscrete
from src.utils import train

def run(cfg):
    wandb_api = cfg['wandb']['api_key']
    wandb_name = cfg['wandb']['proj_name'] if cfg['wandb']['proj_name'] is not None else 'SAC_Discrete'

    if wandb_api is None:
        os.environ['WANDB_MODE'] = 'dryrun'
    else:
        os.environ['WANDB_API_KEY'] = api_wandb

    wandb.init(project=wandb_name, config=cfg)
    cfg = wandb.config

    print(cfg.model)
    sac = SACDiscrete(cfg.model)
    model, mean, std, x = train(sac, cfg.env)

    final_mean, final_std = test(sac, epochs=25)
    print(f"Agent mean score is {final_mean} with std {final_std}")

    plot(x, mean, std)

    wandb.finish()

