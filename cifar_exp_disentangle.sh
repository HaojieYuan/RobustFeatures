CUDA_VISIBLE_DEVICES=0 python cifar10_train.py --batch-size 64 --frequency low --out cifar_low_disentangle_adam > cifar_low_disentangle_adam.txt &
CUDA_VISIBLE_DEVICES=1 python cifar10_train.py --batch-size 64 --frequency high --out cifar_high_disentangle_adam > cifar_high_disentangle_adam.txt &
wait
echo "Done."