python get_data.py --net=pnet
python train.py --net=pnet
python get_data.py --net=rnet --pnet_path=./model_store/pnet_epoch_20.pt
python train.py --net=rnet
python get_data.py --net=onet --pnet_path=./model_store/pnet_epoch_20.pt --rnet_path=./model_store/rnet_epoch_20.pt
python train.py --net=onet
echo "Training finished!"