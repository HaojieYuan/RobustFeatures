CUDA_VISIBLE_DEVICES=3 python cifar10_train.py --batch-size 64 --frequency low --out cifar_low_disentangle > cifar_low_disentangle.txt &
CUDA_VISIBLE_DEVICES=4 python cifar10_train.py --batch-size 64 --frequency high --out cifar_high_disentangle > cifar_high_disentangle.txt &
wait
echo "Done."