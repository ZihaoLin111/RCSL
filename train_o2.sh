set -eu

cd "$(dirname "$0")"

init_txt=${INIT_TXT:-glove}
bs=${BATCH_SIZE:-256}
gpu=${GPU:-0}
python_cmd=${PYTHON:-python3}
data_path=${DATA_PATH:-./data}
vocab_path=${VOCAB_PATH:-./vocab}
glove_cache_path=${GLOVE_CACHE_PATH:-./vocab/vector_cache}
paired_length=${PAIRED_LENGTH:-1000}
mine_epoch=${MINE_EPOCH:-25}
memory_update_interval=${MEMORY_UPDATE_INTERVAL:-5}
seed=${SEED:-42}
tau=${TAU:-0.03}
candidate_k=${OT_CANDIDATE_K:-32}
epsilon=${OT_EPSILON:-0.05}
rho=${OT_RHO:-1.0}
max_iter=${OT_MAX_ITER:-200}
tol=${OT_TOL:-1e-3}
block_size=${OT_BLOCK_SIZE:-1024}
weight_floor=${OT_WEIGHT_FLOOR:-0.0}
confidence=${OT_CONFIDENCE:-mass_concentration}

data=${DATASET:-f30k}
data_name=${data}_precomp
tag=o2_uot_k${candidate_k}_eps${epsilon}_rho${rho}_wf${weight_floor}_seed${seed}_${init_txt}_pl${paired_length}_bs${bs}_tau${tau}
logger_path=./runsx/${data}_${tag}/log
model_path=./runsx/${data}_${tag}/checkpoint

PYTHONHASHSEED=$seed CUDA_VISIBLE_DEVICES=$gpu "$python_cmd" train.py --gpu "$gpu" --seed "$seed" \
  --mining_method ot --paired_length "$paired_length" --data_name "$data_name" \
  --logger_path "$logger_path" --model_path "$model_path" --init_txt "$init_txt" \
  --log_step 200 --embed_size 1024 --tau "$tau" --batch_size "$bs" --data_path "$data_path" \
  --vocab_path "$vocab_path" --glove_cache_path "$glove_cache_path" --MineEpoch "$mine_epoch" \
  --memory_update_interval "$memory_update_interval" --ot_candidate_k "$candidate_k" \
  --ot_epsilon "$epsilon" --ot_rho "$rho" --ot_weight_floor "$weight_floor" \
  --ot_max_iter "$max_iter" --ot_tol "$tol" --ot_block_size "$block_size" \
  --ot_confidence "$confidence"
