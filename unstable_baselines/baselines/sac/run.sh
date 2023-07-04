#!/bin/bash

newTmuxSession(){ #新建tmux session
    session=$1
    tmux has-session -t $session 2>/dev/null
    if [ $? == 0 ]; then
        echo "Session $session already exists"
        tmux kill-session -t $session
        tmux new-session -d -s $session
        echo "Session $session created done"
    else
        tmux new-session -d -s $session
        echo "Session $session created done"
    fi
}



tasks=("HalfCheetah" "Hopper" "Walker2d")
seed="300"
gpu="5"
agent_save_frequency="0"
reset_frequency="0"
utd="5"

for task in "${tasks[@]}"
do
    return_save_path="/home/data/qzj/data/unstable_baselines/unstable_baselines/model_based_rl/results/sac/$task/return/utd32_$task$seed.txt"

    agent_save_path="/home/data/qzj/data/unstable_baselines/unstable_baselines/model_based_rl/results/sac/$task/agents/"

    #newTmuxSession "$task-new"
    newTmuxSession "reset_sac_$task$seed"
    tmux send -t "reset_sac_$task$seed" "cd /home/data/qzj/data/unstable_baselines/unstable_baselines/sac" C-m
    tmux send -t "reset_sac_$task$seed" "conda activate usb" C-m
    tmux send -t "reset_sac_$task$seed" "rm $return_save_path" C-m
    tmux send -t "reset_sac_$task$seed" "python main.py configs/$task-v2.py --gpu $gpu --seed $seed --env $task-v2 --agent_save_frequency $agent_save_frequency --return_save_path $return_save_path --agent_save_path $agent_save_path --reset_frequency $reset_frequency --utd $utd" C-m
done


# python main_mbpo.py --env_name 'Hopper-v2' --num_epoch 300 --model_type 'pytorch'