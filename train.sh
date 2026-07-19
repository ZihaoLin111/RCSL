set -eu

cd "$(dirname "$0")"

init_txt=glove
bs=256
gpu=0
python_cmd=${PYTHON:-python3}
data_path=./data
vocab_path=./vocab
glove_cache_path=./vocab/vector_cache
mine_epoch=25
memory_update_interval=5
mnn_topk_start=10
mnn_topk_end=1
mnn_topk_decay_rounds=3

tau=0.03

data=f30k #coco 
gpu=0
data_name=${data}_precomp
for paired_length in 500 1000 2000 5000
do
tag=bidir_topk${mnn_topk_start}-${mnn_topk_end}r${mnn_topk_decay_rounds}_${init_txt}_pl${paired_length}_bs${bs}_tau${tau}
logger_path=./runsx/${data}_${tag}/log
model_path=./runsx/${data}_${tag}/checkpoint

CUDA_VISIBLE_DEVICES=$gpu "$python_cmd" train.py --gpu "$gpu" --paired_length "$paired_length" \
  --data_name "$data_name" --logger_path "$logger_path" --model_path "$model_path" --init_txt "$init_txt" \
  --log_step 200 --embed_size 1024 --tau "$tau" --batch_size "$bs" --data_path "$data_path" \
  --vocab_path "$vocab_path" --glove_cache_path "$glove_cache_path" --MineEpoch "$mine_epoch" \
  --memory_update_interval "$memory_update_interval" --mnn_topk_start "$mnn_topk_start" \
  --mnn_topk_end "$mnn_topk_end" --mnn_topk_decay_rounds "$mnn_topk_decay_rounds"
done
