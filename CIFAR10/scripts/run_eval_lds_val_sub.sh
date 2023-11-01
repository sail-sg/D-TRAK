echo "gpu_ids: $1"
echo "e_seed: $2"
echo "setting: $3"

echo "index: $4"

echo "start: $5"
echo "end: $6"

export HF_HOME="~/codes/.cache/huggingface"

for seed in `seq 1 1`
do
echo ${seed}
CUDA_VISIBLE_DEVICES=$1 python eval_unconditional.py \
    --dataset_name="cifar10" \
    --dataloader_num_workers=8 \
    --model_config_name_or_path="config.json" \
    --resolution=32 --center_crop \
    --train_batch_size=256 \
    --index_path=./data/indices/$3/$4 \
    --gen_path=./saved/$3/gen \
    --output_dir=./saved/$3/lds-val \
    --e_seed=$2 \
    --start $5 --end $6 \
    --seed=${seed}
done




