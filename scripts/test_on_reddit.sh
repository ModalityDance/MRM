
# Reddit with 150 samples per seen user
python inference.py \
--embed_pt  data/emb/reddit/V2.pt \
--meta_json data/emb/reddit/V2.json \
--ckpt output/reddit/<RUN_DIR>/epoch_<EPOCH>.pt \
--dataset REDDIT \
--seen_train_limit 150 \
--unseen_train_limit 50 \
--hidden_layers 2 \
--inner_lr 5e-3 \
--eval_inner_epochs 1 \
--val_ratio 0.9 \
--score_threshold -1 \
--seed 42 \
--device cuda:0

# Reddit with 100 samples per seen user:
python inference.py \
--embed_pt  data/emb/reddit/V2.pt \
--meta_json data/emb/reddit/V2.json \
--ckpt output/reddit/<RUN_DIR>/epoch_<EPOCH>.pt \
--dataset REDDIT \
--seen_train_limit 100 \
--unseen_train_limit 50 \
--hidden_layers 2 \
--inner_lr 5e-3 \
--eval_inner_epochs 1 \
--val_ratio 0.9 \
--score_threshold -1 \
--seed 42 \
--device cuda:0