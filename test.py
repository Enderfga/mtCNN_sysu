import cv2
from utils.detect import create_mtcnn_net, MtcnnDetector
from utils.vision import vis_face
import argparse


MIN_FACE_SIZE = 24

def parse_args():
    parser = argparse.ArgumentParser(description='Test MTCNN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--net', dest='net', help='which net to show', type=str)
    parser.add_argument('--pnet_path', default="./model_store/pnet_epoch_20.pt",help='path to pnet model', type=str)
    parser.add_argument('--rnet_path', default="./model_store/rnet_epoch_20.pt",help='path to rnet model', type=str)
    parser.add_argument('--onet_path', default="./model_store/onet_epoch_20.pt",help='path to onet model', type=str)
    parser.add_argument('--image_path', default="./mid.png",help='path to image', type=str)
    parser.add_argument('--min_face_size', default=MIN_FACE_SIZE,help='min face size', type=int)
    parser.add_argument('--use_cuda', default=False,help='use cuda', type=bool)
    parser.add_argument('--thresh', default=[0.6, 0.7, 0.7],help='thresh', type=list)
    parser.add_argument('--save_name', default="result.jpg",help='save name', type=str)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    pnet, rnet, onet = create_mtcnn_net(p_model_path=args.pnet_path, r_model_path=args.rnet_path,o_model_path=args.onet_path, use_cuda=args.use_cuda)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=args.min_face_size,threshold=args.thresh)
    img = cv2.imread(args.image_path)
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    p_bboxs, r_bboxs, bboxs, landmarks = mtcnn_detector.detect_face(img)
    # print box_align
    save_name = args.save_name
    if args.net == 'pnet':
        vis_face(img_bg, p_bboxs, landmarks, MIN_FACE_SIZE, save_name)
    elif args.net == 'rnet':
        vis_face(img_bg, r_bboxs, landmarks, MIN_FACE_SIZE, save_name)
    elif args.net == 'onet':
        vis_face(img_bg, bboxs, landmarks, MIN_FACE_SIZE, save_name)
