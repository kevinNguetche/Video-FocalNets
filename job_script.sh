#!/bin/bash
#SBATCH --job-name=videofocalnet8444_train  # Nom du job
# On cedar rrg-mpederso
# On beluga def-mpederso
#SBATCH --account=def-mpederso          # Compte Compute Canada
#SBATCH --nodes=1
# on cedar v100l:4                
# on beluga tesla_v100-sxm2-16gb:4
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:4   # Nombre de GPUs par nœud
#SBATCH --tasks-per-node=4              # Nombre de processus (tasks) par nœud
#SBATCH --cpus-per-task=4               # Nombre de CPUs par processus
# On cedar 128G
# On beluga 64G
#SBATCH --mem=64G                       # Mémoire totale allouée par nœud
#SBATCH --time=0-02:00                  # Temps maximum d'exécution (4 heures)
#SBATCH --mail-user=kevin.nguetche.1@ens.etsmtl.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --open-mode=append
#SBATCH -o /home/dilan/projects/def-mpederso/dilan/job_result/videofocalnet8444_train-%j.out
#SBATCH -e /home/dilan/projects/def-mpederso/dilan/job_result/videofocalnet8444_train-%j.err

cd $SLURM_TMPDIR
# Python env
cp -r /home/dilan/projects/def-mpederso/dilan/focal.zip .
unzip -qq focal.zip
module load StdEnv/2023 cuda/12.2 opencv/4.9.0 python/3.10

source focal/bin/activate

# Datasets
/home/dilan/projects/def-mpederso/dilan/k400_resize/train/
mkdir k400_resize && cd k400_resize
cp -r /projects/def-mpederso/dilan/k400_resize/train_resize_origine.tar.zst .
cp -r /projects/def-mpederso/dilan/k400_resize/val_resize_origine.tar.zst .
tar --use-compress-program=unzstd -xvf train_resize_origine.tar.zst
tar --use-compress-program=unzstd -xvf val_resize_origine.tar.zst

cd $SLURM_TMPDIR

# My code
# If not ssh connexion
# git clone https://github.com/kevinNguetche/Video-FocalNets.git 
git clone git@github.com:kevinNguetche/Video-FocalNets.git
cd Video-FocalNets
git checkout custom_videofocalnet_8
cp -r /home/dilan/projects/def-mpederso/dilan/focalnet_base_srf.pth .

# Cmd launch
timeout 119m python -u -m torch.distributed.launch --nproc_per_node 4 main.py  \
--cfg configs/kinetics400/video-focalnet_base.yaml \
--output /projects/def-mpederso/dilan/checkpointsK400-tiny8444AdamwVideofocalnet120EpochsBatch512 \
--accumulation-steps 16 --opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3 \
-- prefix ${SLURM_TMPDIR}/k400_resize/
DATA.TRAIN_FILE datasets/train_tiny_s.csv \
DATA.VAL_FILE datasets/val_k400_s.csv

# Timeout verification
if [ $? -eq 124 ]; then
  echo "The script timed out after 119 minutes. Requeuing job..."
  # Relancer le même script
  sbatch job_script.sh
else
  echo "The script finished before timing out."

fi

