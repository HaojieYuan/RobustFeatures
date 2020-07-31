CUDA_VISIBLE_DEVICES=0 python cifar10_train.py --batch-size 64 --frequency low --out cifar_low > cifar_low.txt &
CUDA_VISIBLE_DEVICES=1 python cifar10_train.py --batch-size 64 --frequency high --out cifar_high > cifar_high.txt &
CUDA_VISIBLE_DEVICES=2 python cifar10_train.py --batch-size 64 --frequency normal --out cifar_normal > cifar_normal.txt &
wait
echo "Done."