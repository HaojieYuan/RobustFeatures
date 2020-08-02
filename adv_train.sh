CUDA_VISIBLE_DEVICES=8 python cifar10_adv_train.py --batch-size 32 --frequency normal --out cifar_adv_normal_adam > cifar_adv_normal_adam.txt &
wait
echo "Done."