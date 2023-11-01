echo "gpu_ids: $1"
echo "e_seed: $2"

echo "setting: $3"

echo "index: $4"

echo "start: $5"
echo "end: $6"


export MODEL_NAME="lambdalabs/miniSD-diffusers"
export DATASET_NAME="data/artbench-10-imagefolder/**"
export HF_HOME="~/codes/.cache/huggingface"

for seed in `seq 0 2`
do
echo ${seed}
CUDA_VISIBLE_DEVICES=$1 python eval_text_to_image_lora.py \
    --dataset_name=$DATASET_NAME --caption_column="label" \
    --dataloader_num_workers=8 \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --resolution=256 --center_crop \
    --train_batch_size=256 \
    --index_path=./data/indices/$3/$4 \
    --gen_path=./saved/$3/gen \
    --output_dir=./saved/$3/lds-val \
    --e_seed=$2 \
    --start $5 --end $6 \
    --seed=${seed}
done
