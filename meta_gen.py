

folder2id = {}
f = open('/home/haojieyuan/Data/ImageNet/ILSVRC_2012/folder_name2idx.txt')

for line in f:
    split = line.strip.split()
    folder_name = split[0]
    idx = split[1]
    folder2id{folder_name} = idx

f.close()