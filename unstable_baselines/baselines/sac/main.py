import os
import sys
# sys.path.append(os.path.join(os.getcwd(), './'))
# sys.path.append(os.path.join(os.getcwd(), '../../'))
import gym
import click
from unstable_baselines.common.logger import Logger
from unstable_baselines.baselines.sac.trainer import SACTrainer
from unstable_baselines.baselines.sac.agent import SACAgent
from unstable_baselines.common.util import set_device_and_logger, load_config, set_global_seed
from unstable_baselines.common.buffer import ReplayBuffer
from unstable_baselines.common.env_wrapper import get_env
from tqdm import tqdm
from functools import partialmethod

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument("config-path",type=str)
@click.option("--log-dir", default=os.path.join("logs", "sac"))
@click.option("--print-log", type=bool, default=True)
@click.option("--enable-pbar", type=bool, default=True)
@click.option("--info", type=str, default="")
@click.option("--load-path", type=str, default="")


@click.option("--gpu", type=int, default=-1)
@click.option("--seed", type=int, default=35)
@click.option("--env", type=str, default="HalfCheetah-v2") # HalfCheetah-v2, Hopper-v2, Walker2d-v2
@click.option("--agent_save_frequency", type=int, default=0)
@click.option("--agent_save_path", type=str, default="/home/data/qzj/data/unstable_baselines/unstable_baselines/model_based_rl/results/sac/HalfCheetah/agents/")
@click.option("--return_save_path", type=str, default="/home/data/qzj/data/unstable_baselines/unstable_baselines/model_based_rl/results/sac/HalfCheetah/return/half100.txt") #具体到文件，10000步保存一次
@click.option("--reset_frequency", type=int, default=0) #reset的频率，默认不reset
@click.option("--utd", type=int, default=1) 

@click.argument('args', nargs=-1)
def main(config_path, log_dir, gpu, print_log, enable_pbar, seed, info, load_path, env, agent_save_frequency, agent_save_path, return_save_path, reset_frequency, utd, args):
    #todo: add load and update parameters function
    args = load_config(config_path, args)

    #silence tqdm progress bar output
    if not enable_pbar:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    #set global seed
    set_global_seed(seed)

    #initialize logger
    #env_name = args['env_name']
    env_name = env
    logger = Logger(log_dir, env_name, seed, info_str=info, print_to_terminal=print_log)

    #set device and logger
    set_device_and_logger(gpu, logger)

    #save args
    logger.log_str_object("parameters", log_dict = args)

    #initialize environment
    logger.log_str("Initializing Environment")
    train_env = get_env(env_name, seed=seed)
    eval_env = get_env(env_name, seed=seed)
    observation_space = train_env.observation_space
    action_space = train_env.action_space

    #initialize buffer
    logger.log_str("Initializing Buffer")
    buffer = ReplayBuffer(observation_space, action_space, **args['buffer'])

    #initialize agent
    logger.log_str("Initializing Agent")
    agent = SACAgent(observation_space, action_space, **args['agent'])

    #initialize trainer
    logger.log_str("Initializing Trainer")
    trainer  = SACTrainer(
        seed,
        agent,
        train_env,
        eval_env,
        buffer,
        load_path=load_path,
        agent_save_frequency=agent_save_frequency,
        agent_save_path=agent_save_path,
        return_save_path=return_save_path,
        reset_frequency = reset_frequency,
        utd = utd,
        **args['trainer']
    )

    
    logger.log_str("Started training")
    trainer.train()


if __name__ == "__main__":
    main()