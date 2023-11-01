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

export MODEL_NAME="lambdalabs/miniSD-diffusers"
export DATASET_NAME="data/artbench-10-imagefolder/**"
export HF_HOME="~/codes/.cache/huggingface"

CUDA_VISIBLE_DEVICES=$1 python grad_text_to_image_lora.py \
    --dataset_name=$DATASET_NAME --caption_column="label" \
    --dataloader_num_workers=8 \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --resolution=256 --center_crop \
    --train_batch_size=32 \
    --e_seed=$2 \
    --index_path=./data/indices/$3/$4 \
    --gen_path=./saved/$3/gen \
    --split=$5 \
    --output_dir=./saved/$3/$6 \
    --f=$7 \
    --t_strategy=$8 \
    --K=$9 \
    --Z=${10} \
    --seed=42
