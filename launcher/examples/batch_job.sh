#!/bin/sh
#PBS -N idql_waymax
#PBS -P ail722.me1222006.course
#PBS -q scai_q
#PBS -m bea
#PBS -M me1222006@iitd.ac.in
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=9:00:00
# $PBS_O_WORKDIR is the directory from where the job is fired.

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $HOME

./proxy.sh &
export http_proxy="http://proxy22.iitd.ac.in:3128"
export https_proxy="http://proxy22.iitd.ac.in:3128"

# conda init
source /home/mech/btech/me1222006/anaconda3/bin/activate
conda activate waymax
cd /home/mech/btech/me1222006/scratch/projects/IDQL


python /home/mech/btech/me1222006/scratch/projects/IDQL/launcher/examples/train_ddpm_idql_waymo.py --dataset_path /home/mech/btech/me1222006/scratch/data/waymax_expert/waymo_train_10k.npz --eval_data_dir /home/mech/btech/me1222006/scratch/data/raw/validation --eval_num_scenarios 5 --eval_start_scenario 0 --eval_episodes 5 --max_steps 1000000 --eval_interval 100000 --batch_size 512 --eval_gif_dir /home/mech/btech/me1222006/scratch/projects/IDQL/launcher/examples/gifs
