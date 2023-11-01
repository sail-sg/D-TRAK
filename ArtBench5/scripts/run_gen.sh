echo "gpu_ids: $1"
echo "gen_seed: $2"

echo "setting: $3"

export MODEL_NAME="lambdalabs/miniSD-diffusers"
export DATASET_NAME="data/artbench-10-imagefolder/**"
export HF_HOME="~/codes/.cache/huggingface"

CUDA_VISIBLE_DEVICES=$1 python gen_text_to_image_lora.py \
    --dataset_name=$DATASET_NAME --caption_column="label" \
    --dataloader_num_workers=8 \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --resolution=256 --center_crop \
    --train_batch_size=256 \
    --model_path=./saved/$3/sd-lora \
    --gen_seed=$2 \
    --output_dir=./saved/$3/gen
