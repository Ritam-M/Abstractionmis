# Scaling Marginalized Importance Sampling to High-Dimensional State-Spaces via State Abstraction 

This repository contains code to run single-run evaluations of experiments in "Scaling Marginalized Importance Sampling to High-Dimensional State-Spaces via State Abstraction" 
Dependencies:
1. Mujoco 2.1.0
2. pytorch 1.11.0
3. python 3.8.5

Tabular run:
```
python3 run_single_dis.py --env_name ToyMDP --traj_len 100 --oracle_batch_size 500 --batch_size $BATCH --seed 0 --gamma 0.999  --mdp_num $MDP_NUM --pi_set $PI_SET_NUM --eps 1e-10 --exp_name $EXP_NAME
```
```
Assumptions satisifed: $MDP_NUM = 0, $PI_SET_NUM = 0
Not bisim and pie act: $MDP_NUM = 1, $PI_SET_NUM = 0
Bisim and not pie act: $MDP_NUM = 2, $PI_SET_NUM = 1
not both: $MDP_NUM = 3, $PI_SET_NUM = 1
$EXP_NAME = {ope, densities} (densities is for Figure 3b)
```

If condor is available for many trials:
```
python3 run_condor_tab.py results --gamma 0.999 --env_name ToyMDP --mdp_num $MDP_NUM --pi_set $PI_SET_NUM --mix_ratio ($MDP_NUM + $PI_SET_NUM) --num_trials 15 --condor --exp_name $EXP_NAME
```

Function approximation run:
```
python3 run_single_cont.py --env_name $ENV_NAME --traj_len $TRAJ --oracle_batch_size 200 --batch_size $BATCH --gamma 0.995 --Q_lr $QLR --W_lr $WLR --lam_lr 1e-3  --exp_name gan --seed 0 --epochs 100000
```
where
``` 
$ENV_NAME = {Reacher, Walker, Pusher, AntUMaze}
$TRAJ = {200, 500, 300, 500}
$BATCH = {5, 10, 50, 75, 100, 300, 500, 1000}
$QLR/$WLR = {5e-5, 1e-4, 3e-4, 7e-4, 1e-3}
```

If condor is available for many trials/across all data/hyperparamter searching:
```
python3 run_condor.py results --gamma 0.995 --env_name $ENV_NAME --epochs 100000 --num_trials 15 --condor --exp_name gan
```

To graph results, 
1. move *.npy files into appropriate folder (see gen_graphs.sh. for e.g.: results/data/antumaze)
2. execute ./gen_graphs.sh

