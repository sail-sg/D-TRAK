echo "gpu_ids: $1"
echo "main_process_port: $2"
echo "setting: $3"

echo "start: $4"
echo "end: $5"


export HF_HOME="~/codes/.cache/huggingface"

for seed in `seq 0 0`
do
echo ${seed}
    for index in `seq $4 $5`
    do
    echo ${index}
    accelerate launch --gpu_ids $1 --main_process_port=$2 train_unconditional.py \
    --dataset_name="celeba" \
    --dataloader_num_workers=8 \
    --model_config_name_or_path="config.json" \
    --resolution=64 --center_crop --random_flip \
    --train_batch_size=128 \
    --num_epochs=200 \
    --checkpointing_steps=100000 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --adam_weight_decay=1e-6 \
    --logger="wandb" --save_images_epochs=20 \
    --index_path=./data/indices/$3/retrain/sub-idx-${index}.pkl \
    --output_dir=./saved/$3/retrain/ddpm-sub-${index}-${seed} \
    --seed=${seed} \
    --wandb_name="Celeba-$3-retrain"
    done
done
