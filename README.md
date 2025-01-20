# Camouflage_Target_Segmentation

## Install
install the requireements

```shell
# use anaconda 
conda create -n CFS python=3.10
pip3 install torch torchvision torchaudio
pip install scikit-learn
pip install -U cython
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```
conplete the commands, have instelled envionment needed for CRF Unet Adaboost
## train
```shell
cd ./Unet
```
change the mode in main.py
```shell
# mode
parser.add_argument('--mode', type=str, default='test') # train
# epoch
parser.add_argument('--num_epochs', type=int, default=200) # epoch
# image path
image_path=r"/home/iair/Downloads/dataset/test/image"
train_loader = get_loader(image_path=train_image,
                              path="/home/iair/Downloads/dataset/train/GT/",
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)

```
change the par first to start training  Myself use epoch 100
after this, addd dense CRF model to fine the segmentation images




