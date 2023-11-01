echo "gpu_ids: $1"
echo "main_process_port: $2"
echo "setting: $3"

export HF_HOME="~/codes/.cache/huggingface"

accelerate launch --gpu_ids $1 --main_process_port=$2 train_unconditional.py \
    --dataset_name="cifar10" \
    --dataloader_num_workers=8 \
    --model_config_name_or_path="config.json" \
    --resolution=32 --center_crop --random_flip \
    --train_batch_size=128 \
    --num_epochs=200 \
    --checkpointing_steps=-1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --adam_weight_decay=1e-6 \
    --logger="wandb" --save_images_epochs=20 \
    --index_path=./data/indices/$3/idx-train.pkl \
    --output_dir=./saved/$3/ddpm \
    --seed=42 \
    --wandb_name="CIFAR10-$3-train"