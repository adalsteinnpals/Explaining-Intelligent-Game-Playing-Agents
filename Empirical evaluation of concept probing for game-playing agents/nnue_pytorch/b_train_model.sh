train_setting=$1


python train.py --smart-fen-skipping --random-fen-skipping 4 --train-setting $train_setting \
--batch-size 16384 --threads 2 --num-workers 32 --gpus 1 \
--network-save-period 100 --validation-size 100 \
training_data.binpack training_data.binpack
