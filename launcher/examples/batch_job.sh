#!/bin/sh
#PBS -N idql_waymax
#PBS -P ail841.ee3231024.course
#PBS -q high
#PBS -m bea
#PBS -M me1222006@iitd.ac.in
#PBS -l select=1:ncpus=1:ngpus=1:centos=icelake
#PBS -l walltime=24:00:00
# $PBS_O_WORKDIR is the directory from where the job is fired.

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $HOME

./proxy.sh &
export http_proxy="http://proxy22.iitd.ac.in:3128"
export https_proxy="http://proxy22.iitd.ac.in:3128"
export WANDB_MODE=offline


# conda init
source /home/mech/btech/me1222006/anaconda3/bin/activate
conda activate waymax
cd /home/mech/btech/me1222006/scratch/projects/IDQL


python /home/mech/btech/me1222006/scratch/projects/IDQL/launcher/examples/train_ddpm_idql_waymo.py --dataset_path /home/mech/btech/me1222006/scratch/data/waymax_expert_50k/waymo_train_50k_ckpt40000.npz --eval_data_dir /home/mech/btech/me1222006/scratch/data/raw/validation --eval_num_scenarios 500 --eval_start_scenario 0 --eval_episodes 500 --max_steps 1500000 --eval_interval 250000 --batch_size 512 --eval_gif_dir /home/mech/btech/me1222006/scratch/projects/IDQL/launcher/examples/gifs --seed 2 --experiment_name idql_waymo_seed_2 --eval_gif_dir /home/mech/btech/me1222006/scratch/projects/IDQL/launcher/examples/gifs_3 
# python launcher/examples/extract_waymax_expert_data.py \
#         --data_dir /home/mech/btech/me1222006/scratch/data/raw/training \
#         --max_scenarios 10 \
#         --checkpoint_every 5 \
#         --output_path ~/scratch/data/try/abc.npz