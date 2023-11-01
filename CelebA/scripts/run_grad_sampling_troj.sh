echo "gpu_ids: $1"
echo "e_seed: $2"
echo "setting: $3"

echo "index: $4"
echo "split: $5"

echo "ckpt: $6"

echo "f: $7"
echo "t_strategy: $8"
echo "K: $9"
echo "Z: ${10}"

export HF_HOME="~/codes/.cache/huggingface"

CUDA_VISIBLE_DEVICES=$1 python grad_sampling_troj_unconditional.py \
    --dataset_name="celeba" \
    --dataloader_num_workers=8 \
    --model_config_name_or_path="config.json" \
    --resolution=64 --center_crop \
    --train_batch_size=16 \
    --e_seed=$2 \
    --index_path=./data/indices/$3/$4 \
    --split=$5 \
    --output_dir=./saved/$3/$6 \
    --gen_path=./saved/$3/gen \
    --f=$7 \
    --t_strategy=$8 \
    --K=$9 \
    --Z=${10} \
    --seed=42
    