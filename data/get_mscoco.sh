## get mscoco dataset (images)

# http://mscoco.org/dataset/#download
mkdir -p MSCOCO/images
cd MSCOCO/images
# trainset images
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
# valset images
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
# testset images
wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip
# unzip
unzip train2014.zip
unzip val2014.zip
unzip test2015.zip
# rm compressed files
rm -rf train2014.zip
rm -rf val2014.zip
rm -rf test2015.zip
cd ../..
