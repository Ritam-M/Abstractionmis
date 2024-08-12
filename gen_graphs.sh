#!/bin/bash
mkdir results/graphs

# reacher
mkdir results/graphs/reacher
python3 plot.py results/data/reacher/ --domain reacher --plot_type data  --vs batch --other_fixed_val 200;
python3 plot.py results/data/reacher/ --domain reacher --batch 10 --traj 200 --plot_type training --tr_metric rew;
python3 plot.py results/data/reacher/ --domain reacher --batch 500 --traj 200 --plot_type training --tr_metric rew;
python3 plot.py results/data/reacher/ --domain reacher --batch 5 --traj 200 --plot_type hp_sensitivity;
python3 plot.py results/data/reacher/ --domain reacher --batch 10 --traj 200 --plot_type hp_sensitivity;
python3 plot.py results/data/reacher/ --domain reacher --batch 50 --traj 200 --plot_type hp_sensitivity;
mv reacher_*.pdf results/graphs/reacher

# walker2d
mkdir results/graphs/walker
python3 plot.py results/data/walker/ --domain walker --plot_type data  --vs batch --other_fixed_val 500;
python3 plot.py results/data/walker/ --domain walker  --batch 10 --traj 500 --plot_type training --tr_metric rew;
python3 plot.py results/data/walker/ --domain walker  --batch 500 --traj 500 --plot_type training --tr_metric rew;
python3 plot.py results/data/walker/ --domain walker --batch 5 --traj 500 --plot_type hp_sensitivity;
python3 plot.py results/data/walker/ --domain walker --batch 10 --traj 500 --plot_type hp_sensitivity;
python3 plot.py results/data/walker/ --domain walker --batch 50 --traj 500 --plot_type hp_sensitivity;
mv walker_*.pdf results/graphs/walker

# pusher
mkdir results/graphs/pusher
python3 plot.py results/data/pusher/ --domain pusher --plot_type data  --vs batch --other_fixed_val 300;
python3 plot.py results/data/pusher/ --domain pusher --batch 10 --traj 300 --plot_type training --tr_metric rew;
python3 plot.py results/data/pusher/ --domain pusher --batch 500 --traj 300 --plot_type training --tr_metric rew;
python3 plot.py results/data/pusher/ --domain pusher --batch 5 --traj 300 --plot_type hp_sensitivity;
python3 plot.py results/data/pusher/ --domain pusher --batch 10 --traj 300 --plot_type hp_sensitivity;
python3 plot.py results/data/pusher/ --domain pusher --batch 50 --traj 300 --plot_type hp_sensitivity;
mv pusher_*.pdf results/graphs/pusher

# antumaze
mkdir results/graphs/antumaze
python3 plot.py results/data/antumaze/ --domain antumaze --plot_type data  --vs batch --other_fixed_val 500;
python3 plot.py results/data/antumaze/ --domain antumaze  --batch 10 --traj 500 --plot_type training --tr_metric rew;
python3 plot.py results/data/antumaze/ --domain antumaze  --batch 500 --traj 500 --plot_type training --tr_metric rew;
python3 plot.py results/data/antumaze/ --domain antumaze  --batch 5 --traj 500 --plot_type hp_sensitivity;
python3 plot.py results/data/antumaze/ --domain antumaze  --batch 10 --traj 500 --plot_type hp_sensitivity;
python3 plot.py results/data/antumaze/ --domain antumaze  --batch 50 --traj 500 --plot_type hp_sensitivity;
mv antumaze_*.pdf results/graphs/antumaze

# toymdp
mkdir results/graphs/toymdp
python3 plot.py results/data/toymdp/ --domain toymdp --plot_type data --vs batch --other_fixed_val 100;
python3 plot.py results/data/toymdp_densities  --domain toymdp --plot_type dens_scatter --batch 300 --traj 100
python3 plot.py results/data/toymdp10_densities  --domain toymdp --plot_type dens_scatter --batch 300 --traj 100
python3 plot.py results/data/toymdp21_densities  --domain toymdp --plot_type dens_scatter --batch 300 --traj 100
python3 plot.py results/data/toymdp31_densities  --domain toymdp --plot_type dens_scatter --batch 300 --traj 100
