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
gpu="2"
reset_frequency="0"
reset_sac_frequency="0"
reset_layers="6,8"
model_utds=("10")
agent_utds=("10")

for task in "${tasks[@]}"
do
    for agent_utd in "${agent_utds[@]}"
    do
        return_save_path="/home/data/qzj/data/unstable_baselines/unstable_baselines/model_based_rl/results/mbpo/$task/agent_utd${agent_utd}_$task$seed.txt"
        #newTmuxSession "$task-new"
        newTmuxSession "agent_utd${agent_utd}_$task$seed"
        tmux send -t "agent_utd${agent_utd}_$task$seed" "cd /home/data/qzj/data/unstable_baselines/unstable_baselines/model_based_rl/mbpo" C-m
        tmux send -t "agent_utd${agent_utd}_$task$seed" "conda activate usb" C-m
        tmux send -t "agent_utd${agent_utd}_$task$seed" "python main.py configs/$task-v2.py --gpu $gpu --seed $seed --reset_frequency $reset_frequency --return_save_path $return_save_path --reset_layers $reset_layers --reset_sac_frequency $reset_sac_frequency --agent_utd $agent_utd" C-m
    done
done

# python main_mbpo.py --env_name 'Hopper-v2' --num_epoch 300 --model_type 'pytorch'
