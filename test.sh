python test.py --net=pnet --min_face_size=1 --pnet_path=./model_store/pnet_epoch_20.pt --rnet_path=./model_store/rnet_epoch_20.pt --onet_path=./model_store/onet_epoch_20.pt --save_name=pnet
python test.py --net=rnet --min_face_size=1 --pnet_path=./model_store/pnet_epoch_20.pt --rnet_path=./model_store/rnet_epoch_20.pt --onet_path=./model_store/onet_epoch_20.pt --save_name=rnet
python test.py --net=onet --min_face_size=1 --pnet_path=./model_store/pnet_epoch_20.pt --rnet_path=./model_store/rnet_epoch_20.pt --onet_path=./model_store/onet_epoch_20.pt --save_name=onet
echo "Testing finished!"