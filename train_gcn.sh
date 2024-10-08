#!/bin/bash
#SBATCH --job-name="neg5_810"
#SBATCH --partition=smart_health_00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --time 720:00:00
#SBATCH --output=gcn_neg6_turn_final.log

set -xe


echo "START TIME: $(date)"


export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR: " $MASTER_ADDR
export MASTER_PORT=$(expr 29776 + $(echo -n $SLURM_JOBID | tail -c 2))
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo "MASTER_PORT: " $MASTER_PORT

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

echo "NNODES: " $NNODES


# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

export http_proxy=http://172.16.1.135:3128/ ; export https_proxy=http://172.16.1.135:3128/ ; export HTTP_PROXY=http://172.16.1.135:3128/ ; export HTTPS_PROXY=http://172.16.1.135:3128/


CMD=" \
    train_GCN.py \
    --q_model_name_or_path=uer/roberta-base-wwm-chinese-cluecorpussmall \
    --q_model_type=gcn \
    --d_model_name_or_path=uer/roberta-base-wwm-chinese-cluecorpussmall \
    --q_max_seq_length=768 \
    --d_max_seq_length=768 \
    --score_temperature=0.015625 \
    --train_file=/mnt/petrelfs/lizelin/code/data/improved_train.jsonl\
    --validation_file=/mnt/petrelfs/lizelin/code/data/improved_validation.jsonl\
    --test_file=/mnt/petrelfs/lizelin/code/data/improved_test.jsonl\
    --preprocessing_num_workers=8\
    --do_train \
    --do_eval \
    --do_test\
    --overwrite_output_dir \
    --output_dir=./checkpoints89 \
    --remove_unused_columns=False \
    --log_level=info \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --gradient_accumulation_steps=2 \
    --eval_accumulation_steps=2 \
    --num_train_epochs=20 \
    --learning_rate=1e-6 \
    --overwrite_output_dir=True \
    --ddp_find_unused_parameters=True\
    --logging_steps=4 \
    --report_to=tensorboard \
    --save_total_limit=5 \
    --evaluation_strategy=epoch \
    --save_strategy=epoch \
    --metric_for_best_model=eval_ndcg \
    --load_best_model_at_end=True \
    --gradient_checkpointing=True \
    --ddp_find_unused_parameters=False \
    --bf16_full_eval=True \
    --bf16=True\
    --early_stopping_patience=3 \
    --early_stopping_threshold=0.002 \
    --early_stopping=True
"

LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
"


SRUN_ARGS=" \
    --wait=30 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD"

echo "END TIME: $(date)"


