# case info
# chip vendor: nvidia, kunlunxin,  iluvatar, cambricon etc. key vendor is required.
vendor: str = "nvidia"
# model name
name: str = "Transformer"
data_dir: str = "/home/datasets"

do_train = True
fp16 = False
# =========================================================
# data
# =========================================================

init_checkpoint: str = ""
resume: str = ""

# =========================================================
# Model
# =========================================================


# =========================================================
# loss scale
# =========================================================

log_freq: int = 5

seed: int = 41

gradient_accumulation_steps = 1
dist_backend: str = 'nccl'

# =========================================================
# train && evaluate
# =========================================================
device: str = None
n_device: int = 1

adam_betas = [0.9, 0.997]
adam_eps = 1e-09
amp = False
arch = "transformer_wmt_en_de_big_t2t"
attention_dropout = 0.1
beam = 4
bpe_codes = None
buffer_size = 64
clip_norm = 0.0
cpu = False
criterion = "label_smoothed_cross_entropy"
data = "data/wmt14_en_de_joined_dict"
decoder_attention_heads = 16
decoder_embed_dim = 1024
decoder_embed_path = None
decoder_ffn_embed_dim = 4096
decoder_layers = 6
decoder_learned_pos = False
decoder_normalize_before = True
distributed_rank = 0
distributed_world_size = 1
do_sanity_check = False
dropout = 0.1
encoder_attention_heads = 16
encoder_embed_dim = 1024
encoder_embed_path = None
encoder_ffn_embed_dim = 4096
encoder_layers = 6
encoder_learned_pos = False
encoder_normalize_before = True
file = None
fp16 = False
fuse_dropout_add = False
fuse_layer_norm = False
fuse_relu_dropout = False
gen_subset = "test"
label_smoothing = 0.1
left_pad_source = True
left_pad_target = False
lenpen = 1
local_rank = 0
log_interval = 5
lr = [0.000846]
lr_scheduler = "inverse_sqrt"
lr_shrink = 0.1
max_epoch = 2
max_len_a = 0
max_len_b = 200
max_positions = (1024, 1024)
max_sentences = None
max_sentences_valid = None
max_source_positions = 1024
max_target_positions = 1024
max_tokens = 5120
max_update = 100
min_len = 1
min_lr = 0.0
momentum = 0.99
nbest = 1
no_beamable_mm = False
no_early_stop = False
no_epoch_checkpoints = False
no_save = False
no_token_positional_embeddings = False
num_shards = 1
online_eval = True
optimizer = "adam"
pad_sequence = 1
path = None
prefix_size = 0
print_alignment = False
quiet = False
raw_text = False
relu_dropout = 0.1
remove_bpe = "@@"
replace_unk = None
restore_file = "checkpoint_last.pt"
sampling = False
sampling_temperature = 1
sampling_topk = -1
save_dir = "results"
save_interval = 1
save_predictions = False
seed = 1
shard_id = 0
share_all_embeddings = True
share_decoder_input_output_embed = False
source_lang = None
stat_file = "run_log.json"
target_bleu = 0.0
target_lang = None
test_cased_bleu = False
train_subset = "train"
unkpen = 0
unnormalized = False
update_freq = [1]
valid_subset = "valid"
validate_interval = 1
warmup_init_lr = 0.0
warmup_updates = 4000
weight_decay = 0.0

epochs: int = max_epoch
