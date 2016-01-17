## get data and models..

## get data
cd data
./get_vqa.sh
./get_skipthoughts_params.sh
./get_mscoco.sh
cd ..

## get model
cd model
./get_models.sh
cd ..

## get cache for vqa training data
cd cache
./get_cache.sh
cd ..

## compile HashedNets
cd model/HashedNets/libhashnn
./compile.sh
cd ../../..
