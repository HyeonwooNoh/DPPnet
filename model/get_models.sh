## script downloading pre-trained model

## VGG16 model
wget http://cvlab.postech.ac.kr/research/imageqa/model/VGG16_torch.tar.gz
tar -zxvf VGG16_torch.tar.gz
rm -rf VGG16_torch.tar.gz

cd DPPnet
wget http://cvlab.postech.ac.kr/research/dppnet/data/model/DPPnet_vqa.t7
wget http://cvlab.postech.ac.kr/research/dppnet/data/model/CNN_FIXED_vqa.t7
cd ..

