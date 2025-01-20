import argparse
import os
import shutil
from pygame.examples.cursors import image
from PIL import Image
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random

def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    lr = random.random()*0.0005 + 0.0000005
    augmentation_prob= random.random()*0.7
    epoch = config.num_epochs
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    print(config)
    image_path=r"/home/iair/Downloads/dataset/train/image"
    train_image=[]
    for image_name in os.listdir(image_path):
        train_image.append(os.path.join(image_path,image_name))
    val_image=[]
    test_image=[]
    count=0
    image_path=r"/home/iair/Downloads/dataset/test/image"
    for image_name in os.listdir(image_path):
        if count<570:
            val_image.append(os.path.join(image_path,image_name))
        else:
            test_image.append(os.path.join(image_path,image_name))
        count += 1
    # save_path_image="/home/iair/Downloads/dataset/test/test_image/image"
    # save_path_GT = "/home/iair/Downloads/dataset/test/test_image/GT"
    # for i in range(10):
    #     file_name = os.path.basename(test_image[i])
    #     file_name = os.path.splitext(file_name)[0]
    #     file_name = file_name + '.png'
    #     GT_path="/home/iair/Downloads/dataset/test/GT/"+file_name
    #     with Image.open(test_image[i]) as img:
    #         img.verify()
    #         shutil.copy2(test_image[i],save_path_image)
    #     with Image.open(GT_path) as img:
    #         img.verify()
    #         shutil.copy2(GT_path,save_path_GT)

    # print(f"test_image_path-------->{test_image[-1]}")
    train_loader = get_loader(image_path=train_image,
                              path="/home/iair/Downloads/dataset/train/GT/",
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=val_image,
                              path="/home/iair/Downloads/dataset/test/GT/",
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    test_loader = get_loader(image_path=test_image,
                             path="/home/iair/Downloads/dataset/test/GT/",
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='./dataset/train/')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    parser.add_argument('--test_path', type=str, default='./dataset/test/')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
