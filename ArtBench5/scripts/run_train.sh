echo "gpu_ids: $1"
echo "main_process_port: $2"
echo "setting: $3"

export MODEL_NAME="lambdalabs/miniSD-diffusers"
export DATASET_NAME="data/artbench-10-imagefolder/**"
export HF_HOME="~/codes/.cache/huggingface"

accelerate launch --gpu_ids $1 --main_process_port=$2 --mixed_precision="fp16" train_text_to_image_lora.py \
    --dataset_name=$DATASET_NAME --caption_column="label" \
    --dataloader_num_workers=8 \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --resolution=256 --center_crop --random_flip \
    --train_batch_size=64 \
    --num_train_epochs=100 \
    --checkpointing_steps=-1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=3e-04 \
    --adam_weight_decay=1e-06 \
    --lr_scheduler="cosine" \
    --validation_prompt="a ukiyo e painting" \
    --report_to="wandb" --validation_epochs=10 \
    --index_path=./data/indices/$3/idx-train.pkl \
    --output_dir=./saved/$3/sd-lora \
    --seed=42 \
    --wandb_name="Artbench5-$3-train"