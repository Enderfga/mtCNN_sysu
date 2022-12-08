import cv2
from utils.detect import create_mtcnn_net, MtcnnDetector
from utils.vision import vis_face
import argparse


MIN_FACE_SIZE = 3

def parse_args():
    parser = argparse.ArgumentParser(description='Test MTCNN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--net', default='onet', help='which net to show', type=str)
    parser.add_argument('--pnet_path', default="./model_store/pnet_epoch_20.pt",help='path to pnet model', type=str)
    parser.add_argument('--rnet_path', default="./model_store/rnet_epoch_20.pt",help='path to rnet model', type=str)
    parser.add_argument('--onet_path', default="./model_store/onet_epoch_20.pt",help='path to onet model', type=str)
    parser.add_argument('--path', default="./img/mid.png",help='path to image', type=str)
    parser.add_argument('--min_face_size', default=MIN_FACE_SIZE,help='min face size', type=int)
    parser.add_argument('--use_cuda', default=False,help='use cuda', type=bool)
    parser.add_argument('--thresh', default='[0.1, 0.1, 0.1]',help='thresh', type=str)
    parser.add_argument('--save_name', default="result.jpg",help='save name', type=str)
    parser.add_argument('--input_mode', default=1,help='image or video', type=int)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    thresh = [float(i) for i in (args.thresh).split('[')[1].split(']')[0].split(',')]
    pnet, rnet, onet = create_mtcnn_net(p_model_path=args.pnet_path, r_model_path=args.rnet_path,o_model_path=args.onet_path, use_cuda=args.use_cuda)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=args.min_face_size,threshold=thresh)
    if args.input_mode == 1:
        img = cv2.imread(args.path)
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
    elif args.input_mode == 0:
        cap=cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('out.mp4' ,fourcc,10,(640,480))
        while True:
                t1=cv2.getTickCount()
                ret,frame = cap.read()
                if ret == True:
                    boxes_c,landmarks = mtcnn_detector.detect_face(frame)
                    t2=cv2.getTickCount()
                    t=(t2-t1)/cv2.getTickFrequency()
                    fps=1.0/t
                    for i in range(boxes_c.shape[0]):
                        bbox = boxes_c[i, :4]
                        score = boxes_c[i, 4]
                        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                    
                        #画人脸框
                        cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                            (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                        #画置信度
                        cv2.putText(frame, '{:.2f}'.format(score), 
                                    (corpbbox[0], corpbbox[1] - 2), 
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,(0, 0, 255), 2)
                        #画fps值
                    cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    #画关键点
                    for i in range(landmarks.shape[0]):
                        for j in range(len(landmarks[i])//2):
                            cv2.circle(frame, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255))  
                    a = out.write(frame)
                    cv2.imshow("result", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    

