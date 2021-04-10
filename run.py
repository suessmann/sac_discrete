import os
import wandb
from src.sac import SACDiscrete
from src.utils import train, test, plot


def run_training(cfg):
    wandb_api = cfg['wandb']['api_key']
    wandb_name = cfg['wandb']['proj_name'] if cfg['wandb']['proj_name'] is not None else 'SAC_Discrete'

    if wandb_api is None:
        os.environ['WANDB_MODE'] = 'dryrun'
    else:
        os.environ['WANDB_API_KEY'] = wandb_api

    wandb.init(project=wandb_name, config=cfg)
    cfg = wandb.config

    print(cfg.model)
    sac = SACDiscrete(cfg.model)
    model, mean, std, x = train(sac, cfg.env)

    final_mean, final_std = test(sac, epochs=25, env=cfg.env['env'])
    print(f"Agent mean score is {final_mean} with std {final_std}")

    plot(x, mean, std)

    wandb.finish()


def run_eval(cfg):
    os.environ['WANDB_MODE'] = 'dryrun'

    wandb.init(project='', config=cfg)
    cfg = wandb.config

    print(cfg.model)
    sac = SACDiscrete(cfg.model)
    sac.load_model(cfg.eval['path_to_actor'], *cfg.eval['path_to_critics'])

    test(sac, cfg.env['env'], epochs=1, render=True)

