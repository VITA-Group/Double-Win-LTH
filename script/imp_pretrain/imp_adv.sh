python -u main_adv.py \
    -a resnet50 \
    --dist-url 'tcp://127.0.0.1:4356' \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --wd 1e-4 \
    --seed 1 \
    --lr 0.0005 \
    --rate 0.2 \
    --train_step 3 \
    --init $1 \
    --save_dir $2 \
    $3 \
    -b 2048 \
    --prune_epoch 30 \
    --epochs 570 