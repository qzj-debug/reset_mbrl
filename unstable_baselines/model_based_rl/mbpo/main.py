import os
import sys
# sys.path.append(os.path.join(os.getcwd(), '..'))
import gym
import click
from gym.core import Env
from unstable_baselines.common.logger import Logger
from unstable_baselines.model_based_rl.mbpo.trainer import MBPOTrainer
from unstable_baselines.model_based_rl.mbpo.agent import MBPOAgent
from unstable_baselines.common.util import set_device_and_logger, load_config, set_global_seed
from unstable_baselines.common.buffer import ReplayBuffer
from unstable_baselines.common.env_wrapper import get_env
from unstable_baselines.model_based_rl.mbpo.transition_model import TransitionModel
from unstable_baselines.common.scheduler import Scheduler
from unstable_baselines.common import util
from tqdm import tqdm
from functools import partialmethod

def parse_reset_layers(ctx, param, value):
    if value is not None:
        # 将逗号分隔的字符串转换为整数列表
        reset_layers = [int(x) for x in value.split(',')]
        return reset_layers

@click.command(context_settings=dict( #把main函数变成命令行commmad
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument("config-path",type=str, required=True)
@click.option("--log-dir", default=os.path.join("logs","mbpo"))
@click.option("--gpu", type=int, default= 5) #设置可用的gpu
@click.option("--print-log", type=bool, default=True)
@click.option("--enable-pbar", type=bool, default=True)
@click.option("--seed", type=int, default=300)  #设置随机种子
@click.option("--info", type=str, default="")
@click.option("--load-path", type=str, default="")


@click.option("--reset_frequency", type=int, default = 0) #模型的reset频率，每5000步重置一下模型,默认不reset 参数
@click.option("--save_frequency", type=int, default = 0) #每一万步保存一次模型,默认不保存model
@click.option("--model_save_path", type=str, default="/home/data/qzj/data/unstable_baselines/unstable_baselines/model_based_rl/models/reset_mbpo/")#模型保存的路径
@click.option("--return_save_path", type=str, default = '/home/data/qzj/data/unstable_baselines/unstable_baselines/model_based_rl/results/mbpo/HalfCheetah/half300.txt' )#return保存的路径,要具体到文件名
@click.option("--reset_layers", type=str, default= '2,4,6,8', callback=parse_reset_layers) #设置reset的layers,解析为列表

@click.option("--reset_sac_frequency", type=int, default=20000) # reset agent的频率，默认不reset参数, 设置为2*10^4,所有网络都重置


@click.argument('args', nargs=-1)

def main(config_path, log_dir, gpu, print_log, enable_pbar, seed, info, load_path, reset_frequency, save_frequency, model_save_path, return_save_path, reset_layers, reset_sac_frequency, args):
    print(os.getcwd())
    #todo: add load and update parameters function
    args = load_config(config_path, args)

    #silence tqdm progress bar output
    if not enable_pbar:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        
    #set global seed
    set_global_seed(seed)

    #initialize logger
    env_name = args['env_name']
    logger = Logger(log_dir, env_name,seed=seed, info_str=info, print_to_terminal=print_log)
    logger.log_str("logging to {}".format(logger.log_path))

    #set device and logger
    set_device_and_logger(gpu, logger)

    #save args
    logger.log_str_object("parameters", log_dict = args)

    #initialize environment
    logger.log_str("Initializing Environment")
    env = get_env(env_name, seed=seed)
    eval_env = get_env(env_name, seed=seed)
    obs_space = env.observation_space
    action_space = env.action_space

    #initialize buffer
    logger.log_str("Initializing Buffer")
    env_buffer = ReplayBuffer(obs_space, action_space, **args['env_buffer'])
    model_buffer = ReplayBuffer(obs_space, action_space, **args['model_buffer'])

    #initialize agent
    logger.log_str("Initializing Agent")
    agent = MBPOAgent(obs_space, action_space, env_name=env_name, **args['agent'])
    print("initializing mdoel")
    #initialize env model predictor
    transition_model = TransitionModel(obs_space, action_space, env_name = env_name, **args['transition_model'])

    print("initializing generator")
    #initialize rollout step generator
    rollout_step_generator = Scheduler(**args['rollout_step_scheduler'])


    #initialize trainer
    logger.log_str("Initializing Trainer")
    trainer  = MBPOTrainer(
        agent,
        env,
        eval_env,
        transition_model,
        env_buffer,
        model_buffer,
        rollout_step_generator,
        load_path,
        reset_frequency,
        save_frequency,
        model_save_path,
        return_save_path,
        reset_layers,
        reset_sac_frequency,
        **args['trainer']
    )
    
    logger.log_str("Started training")
    trainer.train()


if __name__ == "__main__":
    main()