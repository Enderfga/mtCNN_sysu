<<<<<<< HEAD
import sys
import numpy as np
import cv2
import os
from utils.tool import IoU,convert_to_square
import numpy.random as npr
import argparse
from utils.detect import MtcnnDetector, create_mtcnn_net
from utils.dataloader import ImageDB,TestImageLoader
import time
from six.moves import cPickle
import utils.config as config
import utils.vision as vision
sys.path.append(os.getcwd())


txt_from_path = './data_set/wider_face_train_bbx_gt.txt'
anno_file = os.path.join(config.ANNO_STORE_DIR, 'anno_train.txt')
# anno_file = './anno_store/anno_train.txt'

prefix = ''
use_cuda = True
im_dir = "./data_set/face_detection/WIDER_train/images/"
traindata_store = './data_set/train/'
prefix_path = "./data_set/face_detection/WIDER_train/images/"
annotation_file = './anno_store/anno_train.txt'
prefix_path_lm = ''
annotation_file_lm = "./data_set/face_landmark/CNN_FacePoint/train/trainImageList.txt"
# ----------------------------------------------------other----------------------------------------------
pos_save_dir = "./data_set/train/12/positive"
part_save_dir = "./data_set/train/12/part"
neg_save_dir = './data_set/train/12/negative'
pnet_postive_file =  os.path.join(config.ANNO_STORE_DIR, 'pos_12.txt')
pnet_part_file = os.path.join(config.ANNO_STORE_DIR, 'part_12.txt')
pnet_neg_file = os.path.join(config.ANNO_STORE_DIR, 'neg_12.txt')
imglist_filename_pnet = os.path.join(config.ANNO_STORE_DIR, 'imglist_anno_12.txt')
# ----------------------------------------------------PNet----------------------------------------------
rnet_postive_file =  os.path.join(config.ANNO_STORE_DIR, 'pos_24.txt')
rnet_part_file = os.path.join(config.ANNO_STORE_DIR, 'part_24.txt')
rnet_neg_file = os.path.join(config.ANNO_STORE_DIR, 'neg_24.txt')
rnet_landmark_file = os.path.join(config.ANNO_STORE_DIR, 'landmark_24.txt')
imglist_filename_rnet = os.path.join(config.ANNO_STORE_DIR, 'imglist_anno_24.txt')
# ----------------------------------------------------RNet----------------------------------------------
onet_postive_file =  os.path.join(config.ANNO_STORE_DIR, 'pos_48.txt')
onet_part_file = os.path.join(config.ANNO_STORE_DIR, 'part_48.txt')
onet_neg_file = os.path.join(config.ANNO_STORE_DIR, 'neg_48.txt')
onet_landmark_file = os.path.join(config.ANNO_STORE_DIR, 'landmark_48.txt')
imglist_filename_onet = os.path.join(config.ANNO_STORE_DIR, 'imglist_anno_48.txt')
# ----------------------------------------------------ONet----------------------------------------------



def assemble_data(output_file, anno_file_list=[]):

    #assemble the pos, neg, part annotations to one file
    size = 12

    if len(anno_file_list)==0:
        return 0

    if os.path.exists(output_file):
        os.remove(output_file)

    for anno_file in anno_file_list:
        with open(anno_file, 'r') as f:
            print(anno_file)
            anno_lines = f.readlines()

        base_num = 250000

        if len(anno_lines) > base_num * 3:
            idx_keep = npr.choice(len(anno_lines), size=base_num * 3, replace=True)
        elif len(anno_lines) > 100000:
            idx_keep = npr.choice(len(anno_lines), size=len(anno_lines), replace=True)
        else:
            idx_keep = np.arange(len(anno_lines))
            np.random.shuffle(idx_keep)
        chose_count = 0
        with open(output_file, 'a+') as f:
            for idx in idx_keep:
                # write lables of pos, neg, part images
                f.write(anno_lines[idx])
                chose_count+=1

    return chose_count
def wider_face(txt_from_path, txt_to_path):
    line_from_count = 0
    with open(txt_from_path, 'r') as f:
        annotations = f.readlines()
    with open(txt_to_path, 'w+') as f:
        while line_from_count < len(annotations):
            if annotations[line_from_count][2]=='-':
                img_name = annotations[line_from_count][:-1]
                line_from_count += 1                                                    # change line to read the number
                bbox_count = int(annotations[line_from_count])                          # num of bboxes
                line_from_count += 1                                                    # change line to read the posession
                for _ in range(bbox_count):
                    bbox = list(map(int,annotations[line_from_count].split()[:4]))      # give a loop to append all the boxes
                    bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]         # make x1, y1, w, h  -->  x1, y1, x2, y2
                    bbox = list(map(str,bbox))
                    img_name += (' '+' '.join(bbox))
                    line_from_count+=1
                f.write(img_name +'\n')
            else:                                                                       # dectect the file name
                line_from_count+=1                                                      

# ----------------------------------------------------origin----------------------------------------------
def get_Pnet_data():
    if not os.path.exists(pos_save_dir):
        os.makedirs(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.makedirs(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.makedirs(neg_save_dir)
    f1 = open(os.path.join('./anno_store', 'pos_12.txt'), 'w')
    f2 = open(os.path.join('./anno_store', 'neg_12.txt'), 'w')
    f3 = open(os.path.join('./anno_store', 'part_12.txt'), 'w')
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print("%d pics in total" % num)
    p_idx = 0 # positive
    n_idx = 0 # negative
    d_idx = 0 # dont care
    idx = 0
    box_idx = 0
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        # annotation[0]文件名
        im_path = os.path.join(im_dir, annotation[0])
        # print(im_path)
        # print(os.path.exists(im_path))
        bbox = list(map(float, annotation[1:]))
        # annotation[1:]人脸坐标，一张脸4个值，对应两个点的坐标
        boxes = np.array(bbox, dtype=np.int32).reshape(-1, 4)
        # -1处的值为人脸数目
        if boxes.shape[0]==0:
            continue
        # 若无人脸则跳过本次循环
        img = cv2.imread(im_path)
        # print(img.shape)
        # exit()
        # 计数
        idx += 1
        if idx % 100 == 0:
            print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

        # 图片三通道
        height, width, channel = img.shape

        neg_num = 0

        # 取50次不同的框
        while neg_num < 50:
            size = np.random.randint(12, min(width, height) / 2)
            nx = np.random.randint(0, width - size)
            ny = np.random.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])

            Iou = IoU(crop_box, boxes) # IoU为 重合部分 / 两框之和 ，越大越好

            cropped_im = img[ny: ny + size, nx: nx + size, :]  # 裁去多余部分并resize成 12*12
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1

        for box in boxes:
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            # w = x2 - x1 + 1
            # h = y2 - y1 + 1
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # ignore small faces
            # in case the ground truth boxes of small faces are not accurate
            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue
            if w < 12 or h < 12:
                continue

            # generate negative examples that have overlap with gt
            for i in range(5):
                size = np.random.randint(12, min(width, height) / 2)

                # delta_x and delta_y are offsets of (x1, y1)
                delta_x = np.random.randint(max(-size, -x1), w)
                delta_y = np.random.randint(max(-size, -y1), h)
                nx1 = max(0, x1 + delta_x)
                ny1 = max(0, y1 + delta_y)

                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IoU(crop_box, boxes)

                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    f2.write(save_file + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1

            # generate positive examples and part faces
            for i in range(20):
                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)

                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
                ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                cropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif IoU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            box_idx += 1
            #print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

    f1.close()
    f2.close()
    f3.close()


def assembel_Pnet_data():
    anno_list = []

    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    # anno_list.append(pnet_landmark_file)
    chose_count = assemble_data(imglist_filename_pnet ,anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename_pnet)

# -----------------------------------------------------------------------------------------------------------------------------------------------#

def gen_rnet_data(data_dir, anno_file, pnet_model_file, prefix_path='', use_cuda=True, vis=False):

    """
    :param data_dir: train data
    :param anno_file:
    :param pnet_model_file:
    :param prefix_path:
    :param use_cuda:
    :param vis:
    :return:
    """

    # load trained pnet model
    
    pnet, _, _ = create_mtcnn_net(p_model_path = pnet_model_file, use_cuda = use_cuda)
    mtcnn_detector = MtcnnDetector(pnet = pnet, min_face_size = 12)

    # load original_anno_file, length = 12880
    imagedb = ImageDB(anno_file, mode = "test", prefix_path = prefix_path)
    imdb = imagedb.load_imdb()
    image_reader = TestImageLoader(imdb, 1, False)
    
    all_boxes = list()
    batch_idx = 0

    print('size:%d' %image_reader.size)
    for databatch in image_reader:
        if batch_idx % 100 == 0:
            print ("%d images done" % batch_idx)
        im = databatch
        t = time.time()

        # obtain boxes and aligned boxes
        boxes, boxes_align = mtcnn_detector.detect_pnet(im=im)
        if boxes_align is None:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue
        if vis:
            rgb_im = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
            vision.vis_two(rgb_im, boxes, boxes_align)

        t1 = time.time() - t
        print('cost time ',t1)
        t = time.time()
        all_boxes.append(boxes_align)
        batch_idx += 1
        # if batch_idx == 100:
            # break
        # print("shape of all boxes {0}".format(all_boxes))
        # time.sleep(5)

    # save_path = model_store_path()
    # './model_store'
    save_path = './model_store'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, "detections_%d.pkl" % int(time.time()))
    with open(save_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    # save_file = './model_store/detections_1588751332.pkl'
    gen_rnet_sample_data(data_dir, anno_file, save_file, prefix_path)



def gen_rnet_sample_data(data_dir, anno_file, det_boxs_file, prefix_path):

    """
    :param data_dir:
    :param anno_file: original annotations file of wider face data
    :param det_boxs_file: detection boxes file
    :param prefix_path:
    :return:
    """

    neg_save_dir = os.path.join(data_dir, "24/negative")
    pos_save_dir = os.path.join(data_dir, "24/positive")
    part_save_dir = os.path.join(data_dir, "24/part")


    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        # print(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image

    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    image_size = 24
    net = "rnet"

    im_idx_list = list()
    gt_boxes_list = list()
    num_of_images = len(annotations)
    print ("processing %d images in total" % num_of_images)

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_idx = os.path.join(prefix_path, annotation[0])
        # im_idx = annotation[0]

        boxes = list(map(float, annotation[1:]))
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)


    # './anno_store'
    save_path = './anno_store'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f1 = open(os.path.join(save_path, 'pos_%d.txt' % image_size), 'w')
    f2 = open(os.path.join(save_path, 'neg_%d.txt' % image_size), 'w')
    f3 = open(os.path.join(save_path, 'part_%d.txt' % image_size), 'w')

    # print(det_boxs_file)
    det_handle = open(det_boxs_file, 'rb')

    det_boxes = cPickle.load(det_handle)

    # an image contain many boxes stored in an array
    print(len(det_boxes), num_of_images)
    # assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):

        # if (im_idx+1) == 100:
            # break

        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if gts.shape[0]==0:
            continue
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        # change to square
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            # Iou with all gts must below 0.3
            if np.max(Iou) < 0.3 and neg_num < 60:
                # save the examples
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                # print(save_file)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    f1.close()
    f2.close()
    f3.close()

def model_store_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/model_store"

def get_Rnet_data(pnet_model):
    gen_rnet_data(traindata_store, annotation_file, pnet_model_file = pnet_model, prefix_path = prefix_path, use_cuda = True)


def assembel_Rnet_data():
    anno_list = []

    anno_list.append(rnet_postive_file)
    anno_list.append(rnet_part_file)
    anno_list.append(rnet_neg_file)
    # anno_list.append(pnet_landmark_file)

    chose_count = assemble_data(imglist_filename_rnet ,anno_list)
    print("RNet train annotation result file path:%s" % imglist_filename_rnet)
#-----------------------------------------------------------------------------------------------------------------------------------------------#
def gen_onet_data(data_dir, anno_file, pnet_model_file, rnet_model_file, prefix_path='', use_cuda=True, vis=False):


    pnet, rnet, _ = create_mtcnn_net(p_model_path=pnet_model_file, r_model_path=rnet_model_file, use_cuda=use_cuda)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, min_face_size=12)

    imagedb = ImageDB(anno_file,mode="test",prefix_path=prefix_path)
    imdb = imagedb.load_imdb()
    image_reader = TestImageLoader(imdb,1,False)

    all_boxes = list()
    batch_idx = 0

    print('size:%d' % image_reader.size)
    for databatch in image_reader:
        if batch_idx % 50 == 0:
            print("%d images done" % batch_idx)

        im = databatch

        t = time.time()

        # pnet detection = [x1, y1, x2, y2, score, reg]
        p_boxes, p_boxes_align = mtcnn_detector.detect_pnet(im=im)

        t0 = time.time() - t
        t = time.time()
        # rnet detection
        boxes, boxes_align = mtcnn_detector.detect_rnet(im=im, dets=p_boxes_align)

        t1 = time.time() - t
        print('cost time pnet--',t0,'  rnet--',t1)
        t = time.time()

        if boxes_align is None:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue
        if vis:
            rgb_im = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
            vision.vis_two(rgb_im, boxes, boxes_align)

        
        all_boxes.append(boxes_align)
        batch_idx += 1

    save_path = './model_store'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, "detections_%d.pkl" % int(time.time()))
    with open(save_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)


    gen_onet_sample_data(data_dir,anno_file,save_file,prefix_path)



def gen_onet_sample_data(data_dir,anno_file,det_boxs_file,prefix):

    neg_save_dir = os.path.join(data_dir, "48/negative")
    pos_save_dir = os.path.join(data_dir, "48/positive")
    part_save_dir = os.path.join(data_dir, "48/part")

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image

    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    image_size = 48
    net = "onet"

    im_idx_list = list()
    gt_boxes_list = list()
    num_of_images = len(annotations)
    print("processing %d images in total" % num_of_images)

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_idx = os.path.join(prefix,annotation[0])

        boxes = list(map(float, annotation[1:]))
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)

    save_path = './anno_store'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f1 = open(os.path.join(save_path, 'pos_%d.txt' % image_size), 'w')
    f2 = open(os.path.join(save_path, 'neg_%d.txt' % image_size), 'w')
    f3 = open(os.path.join(save_path, 'part_%d.txt' % image_size), 'w')

    det_handle = open(det_boxs_file, 'rb')

    det_boxes = cPickle.load(det_handle)
    print(len(det_boxes), num_of_images)
    # assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1
        if gts.shape[0]==0:
            continue
        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        for box in dets:
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    f1.close()
    f2.close()
    f3.close()



def model_store_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/model_store"


def get_Onet_data(pnet_model, rnet_model):
    gen_onet_data(traindata_store, annotation_file, pnet_model_file = pnet_model, rnet_model_file = rnet_model,prefix_path=prefix_path,use_cuda = True, vis = False)


def assembel_Onet_data():
    anno_list = []

    anno_list.append(onet_postive_file)
    anno_list.append(onet_part_file)
    anno_list.append(onet_neg_file)
    anno_list.append(onet_landmark_file)

    chose_count = assemble_data(imglist_filename_onet ,anno_list)
    print("ONet train annotation result file path:%s" % imglist_filename_onet)


def gen_landmark_48(anno_file, data_dir, prefix = ''):


    size = 48
    image_id = 0

    landmark_imgs_save_dir = os.path.join(data_dir,"48/landmark")
    if not os.path.exists(landmark_imgs_save_dir):
        os.makedirs(landmark_imgs_save_dir)

    anno_dir = './anno_store'
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)

    landmark_anno_filename = "landmark_48.txt"
    save_landmark_anno = os.path.join(anno_dir,landmark_anno_filename)

    # print(save_landmark_anno)
    # time.sleep(5)
    f = open(save_landmark_anno, 'w')
    # dstdir = "train_landmark_few"

    with open(anno_file, 'r') as f2:
        annotations = f2.readlines()

    num = len(annotations)
    print("%d total images" % num)

    l_idx =0
    idx = 0
    # image_path bbox landmark(5*2)
    for annotation in annotations:
        # print imgPath

        annotation = annotation.strip().split(' ')

        assert len(annotation)==15,"each line should have 15 element"

        im_path = os.path.join('./data_set/face_landmark/CNN_FacePoint/train/',annotation[0].replace("\\", "/"))

        gt_box = list(map(float, annotation[1:5]))
        # gt_box = [gt_box[0], gt_box[2], gt_box[1], gt_box[3]]


        gt_box = np.array(gt_box, dtype=np.int32)

        landmark = list(map(float, annotation[5:]))
        landmark = np.array(landmark, dtype=np.float)

        img = cv2.imread(im_path)
        # print(im_path)
        assert (img is not None)

        height, width, channel = img.shape
        # crop_face = img[gt_box[1]:gt_box[3]+1, gt_box[0]:gt_box[2]+1]
        # crop_face = cv2.resize(crop_face,(size,size))

        idx = idx + 1
        if idx % 100 == 0:
            print("%d images done, landmark images: %d"%(idx,l_idx))
        # print(im_path)
        # print(gt_box)
        x1, x2, y1, y2 = gt_box
        gt_box[1] = y1
        gt_box[2] = x2
        # time.sleep(5)

        # gt's width
        w = x2 - x1 + 1
        # gt's height
        h = y2 - y1 + 1
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        # random shift
        for i in range(10):
            bbox_size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            delta_x = np.random.randint(-w * 0.2, w * 0.2)
            delta_y = np.random.randint(-h * 0.2, h * 0.2)
            nx1 = max(x1 + w / 2 - bbox_size / 2 + delta_x, 0)
            ny1 = max(y1 + h / 2 - bbox_size / 2 + delta_y, 0)

            nx2 = nx1 + bbox_size
            ny2 = ny1 + bbox_size
            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            cropped_im = img[int(ny1):int(ny2) + 1, int(nx1):int(nx2) + 1, :]
            resized_im = cv2.resize(cropped_im, (size, size),interpolation=cv2.INTER_LINEAR)

            offset_x1 = (x1 - nx1) / float(bbox_size)
            offset_y1 = (y1 - ny1) / float(bbox_size)
            offset_x2 = (x2 - nx2) / float(bbox_size)
            offset_y2 = (y2 - ny2) / float(bbox_size)

            offset_left_eye_x = (landmark[0] - nx1) / float(bbox_size)
            offset_left_eye_y = (landmark[1] - ny1) / float(bbox_size)

            offset_right_eye_x = (landmark[2] - nx1) / float(bbox_size)
            offset_right_eye_y = (landmark[3] - ny1) / float(bbox_size)

            offset_nose_x = (landmark[4] - nx1) / float(bbox_size)
            offset_nose_y = (landmark[5] - ny1) / float(bbox_size)

            offset_left_mouth_x = (landmark[6] - nx1) / float(bbox_size)
            offset_left_mouth_y = (landmark[7] - ny1) / float(bbox_size)

            offset_right_mouth_x = (landmark[8] - nx1) / float(bbox_size)
            offset_right_mouth_y = (landmark[9] - ny1) / float(bbox_size)


            # cal iou
            iou = IoU(crop_box.astype(np.float), np.expand_dims(gt_box.astype(np.float), 0))
            # print(iou)
            if iou > 0.65:
                save_file = os.path.join(landmark_imgs_save_dir, "%s.jpg" % l_idx)
                cv2.imwrite(save_file, resized_im)

                f.write(save_file + ' -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n' % \
                (offset_x1, offset_y1, offset_x2, offset_y2, \
                offset_left_eye_x,offset_left_eye_y,offset_right_eye_x,offset_right_eye_y,offset_nose_x,offset_nose_y,offset_left_mouth_x,offset_left_mouth_y,offset_right_mouth_x,offset_right_mouth_y))
                # print(save_file)
                # print(save_landmark_anno)
                l_idx += 1

    f.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Get data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--net', dest='net', help='which net to show', type=str)
    parser.add_argument('--pnet_path', default="./original_model/pnet_epoch.pt",help='path to pnet model', type=str)
    parser.add_argument('--rnet_path', default="./original_model/rnet_epoch.pt",help='path to rnet model', type=str)
    parser.add_argument('--use_cuda', default=True,help='use cuda', type=bool)

    args = parser.parse_args()
    return args

#-----------------------------------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    args = parse_args()
    dir = 'anno_store'
    if not os.path.exists(dir):
        os.makedirs(dir)
    if args.net == "pnet":
        wider_face(txt_from_path, anno_file)
        get_Pnet_data()
        assembel_Pnet_data()
    elif args.net == "rnet":
        get_Rnet_data(args.pnet_path)
        assembel_Rnet_data()
    elif args.net == "onet":
        get_Onet_data(args.pnet_path, args.rnet_path)
        gen_landmark_48(annotation_file_lm, traindata_store, prefix_path_lm)
=======
import sys
import numpy as np
import cv2
import os
from utils.tool import IoU,convert_to_square
import numpy.random as npr
import argparse
from utils.detect import MtcnnDetector, create_mtcnn_net
from utils.dataloader import ImageDB,TestImageLoader
import time
from six.moves import cPickle
import utils.config as config
import utils.vision as vision
sys.path.append(os.getcwd())


txt_from_path = './data_set/wider_face_train_bbx_gt.txt'
anno_file = os.path.join(config.ANNO_STORE_DIR, 'anno_train.txt')
# anno_file = './anno_store/anno_train.txt'

prefix = ''
use_cuda = True
im_dir = "./data_set/face_detection/WIDER_train/images/"
traindata_store = './data_set/train/'
prefix_path = "./data_set/face_detection/WIDER_train/images/"
annotation_file = './anno_store/anno_train.txt'
prefix_path_lm = ''
annotation_file_lm = "./data_set/face_landmark/CNN_FacePoint/train/trainImageList.txt"
# ----------------------------------------------------other----------------------------------------------
pos_save_dir = "./data_set/train/12/positive"
part_save_dir = "./data_set/train/12/part"
neg_save_dir = './data_set/train/12/negative'
pnet_postive_file =  os.path.join(config.ANNO_STORE_DIR, 'pos_12.txt')
pnet_part_file = os.path.join(config.ANNO_STORE_DIR, 'part_12.txt')
pnet_neg_file = os.path.join(config.ANNO_STORE_DIR, 'neg_12.txt')
imglist_filename_pnet = os.path.join(config.ANNO_STORE_DIR, 'imglist_anno_12.txt')
# ----------------------------------------------------PNet----------------------------------------------
rnet_postive_file =  os.path.join(config.ANNO_STORE_DIR, 'pos_24.txt')
rnet_part_file = os.path.join(config.ANNO_STORE_DIR, 'part_24.txt')
rnet_neg_file = os.path.join(config.ANNO_STORE_DIR, 'neg_24.txt')
rnet_landmark_file = os.path.join(config.ANNO_STORE_DIR, 'landmark_24.txt')
imglist_filename_rnet = os.path.join(config.ANNO_STORE_DIR, 'imglist_anno_24.txt')
# ----------------------------------------------------RNet----------------------------------------------
onet_postive_file =  os.path.join(config.ANNO_STORE_DIR, 'pos_48.txt')
onet_part_file = os.path.join(config.ANNO_STORE_DIR, 'part_48.txt')
onet_neg_file = os.path.join(config.ANNO_STORE_DIR, 'neg_48.txt')
onet_landmark_file = os.path.join(config.ANNO_STORE_DIR, 'landmark_48.txt')
imglist_filename_onet = os.path.join(config.ANNO_STORE_DIR, 'imglist_anno_48.txt')
# ----------------------------------------------------ONet----------------------------------------------



def assemble_data(output_file, anno_file_list=[]):

    #assemble the pos, neg, part annotations to one file
    size = 12

    if len(anno_file_list)==0:
        return 0

    if os.path.exists(output_file):
        os.remove(output_file)

    for anno_file in anno_file_list:
        with open(anno_file, 'r') as f:
            print(anno_file)
            anno_lines = f.readlines()

        base_num = 250000

        if len(anno_lines) > base_num * 3:
            idx_keep = npr.choice(len(anno_lines), size=base_num * 3, replace=True)
        elif len(anno_lines) > 100000:
            idx_keep = npr.choice(len(anno_lines), size=len(anno_lines), replace=True)
        else:
            idx_keep = np.arange(len(anno_lines))
            np.random.shuffle(idx_keep)
        chose_count = 0
        with open(output_file, 'a+') as f:
            for idx in idx_keep:
                # write lables of pos, neg, part images
                f.write(anno_lines[idx])
                chose_count+=1

    return chose_count
def wider_face(txt_from_path, txt_to_path):
    line_from_count = 0
    with open(txt_from_path, 'r') as f:
        annotations = f.readlines()
    with open(txt_to_path, 'w+') as f:
        while line_from_count < len(annotations):
            if annotations[line_from_count][2]=='-':
                img_name = annotations[line_from_count][:-1]
                line_from_count += 1                                                    # change line to read the number
                bbox_count = int(annotations[line_from_count])                          # num of bboxes
                line_from_count += 1                                                    # change line to read the posession
                for _ in range(bbox_count):
                    bbox = list(map(int,annotations[line_from_count].split()[:4]))      # give a loop to append all the boxes
                    bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]         # make x1, y1, w, h  -->  x1, y1, x2, y2
                    bbox = list(map(str,bbox))
                    img_name += (' '+' '.join(bbox))
                    line_from_count+=1
                f.write(img_name +'\n')
            else:                                                                       # dectect the file name
                line_from_count+=1                                                      

# ----------------------------------------------------origin----------------------------------------------
def get_Pnet_data():
    if not os.path.exists(pos_save_dir):
        os.makedirs(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.makedirs(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.makedirs(neg_save_dir)
    f1 = open(os.path.join('./anno_store', 'pos_12.txt'), 'w')
    f2 = open(os.path.join('./anno_store', 'neg_12.txt'), 'w')
    f3 = open(os.path.join('./anno_store', 'part_12.txt'), 'w')
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print("%d pics in total" % num)
    p_idx = 0 # positive
    n_idx = 0 # negative
    d_idx = 0 # dont care
    idx = 0
    box_idx = 0
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        # annotation[0]文件名
        im_path = os.path.join(im_dir, annotation[0])
        # print(im_path)
        # print(os.path.exists(im_path))
        bbox = list(map(float, annotation[1:]))
        # annotation[1:]人脸坐标，一张脸4个值，对应两个点的坐标
        boxes = np.array(bbox, dtype=np.int32).reshape(-1, 4)
        # -1处的值为人脸数目
        if boxes.shape[0]==0:
            continue
        # 若无人脸则跳过本次循环
        img = cv2.imread(im_path)
        # print(img.shape)
        # exit()
        # 计数
        idx += 1
        if idx % 100 == 0:
            print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

        # 图片三通道
        height, width, channel = img.shape

        neg_num = 0

        # 取50次不同的框
        while neg_num < 50:
            size = np.random.randint(12, min(width, height) / 2)
            nx = np.random.randint(0, width - size)
            ny = np.random.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])

            Iou = IoU(crop_box, boxes) # IoU为 重合部分 / 两框之和 ，越大越好

            cropped_im = img[ny: ny + size, nx: nx + size, :]  # 裁去多余部分并resize成 12*12
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1

        for box in boxes:
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            # w = x2 - x1 + 1
            # h = y2 - y1 + 1
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # ignore small faces
            # in case the ground truth boxes of small faces are not accurate
            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue
            if w < 12 or h < 12:
                continue

            # generate negative examples that have overlap with gt
            for i in range(5):
                size = np.random.randint(12, min(width, height) / 2)

                # delta_x and delta_y are offsets of (x1, y1)
                delta_x = np.random.randint(max(-size, -x1), w)
                delta_y = np.random.randint(max(-size, -y1), h)
                nx1 = max(0, x1 + delta_x)
                ny1 = max(0, y1 + delta_y)

                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IoU(crop_box, boxes)

                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    f2.write(save_file + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1

            # generate positive examples and part faces
            for i in range(20):
                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)

                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
                ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                cropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif IoU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            box_idx += 1
            #print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

    f1.close()
    f2.close()
    f3.close()


def assembel_Pnet_data():
    anno_list = []

    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    # anno_list.append(pnet_landmark_file)
    chose_count = assemble_data(imglist_filename_pnet ,anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename_pnet)

# -----------------------------------------------------------------------------------------------------------------------------------------------#

def gen_rnet_data(data_dir, anno_file, pnet_model_file, prefix_path='', use_cuda=True, vis=False):

    """
    :param data_dir: train data
    :param anno_file:
    :param pnet_model_file:
    :param prefix_path:
    :param use_cuda:
    :param vis:
    :return:
    """

    # load trained pnet model
    
    pnet, _, _ = create_mtcnn_net(p_model_path = pnet_model_file, use_cuda = use_cuda)
    mtcnn_detector = MtcnnDetector(pnet = pnet, min_face_size = 12)

    # load original_anno_file, length = 12880
    imagedb = ImageDB(anno_file, mode = "test", prefix_path = prefix_path)
    imdb = imagedb.load_imdb()
    image_reader = TestImageLoader(imdb, 1, False)
    
    all_boxes = list()
    batch_idx = 0

    print('size:%d' %image_reader.size)
    for databatch in image_reader:
        if batch_idx % 100 == 0:
            print ("%d images done" % batch_idx)
        im = databatch
        t = time.time()

        # obtain boxes and aligned boxes
        boxes, boxes_align = mtcnn_detector.detect_pnet(im=im)
        if boxes_align is None:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue
        if vis:
            rgb_im = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
            vision.vis_two(rgb_im, boxes, boxes_align)

        t1 = time.time() - t
        print('cost time ',t1)
        t = time.time()
        all_boxes.append(boxes_align)
        batch_idx += 1
        # if batch_idx == 100:
            # break
        # print("shape of all boxes {0}".format(all_boxes))
        # time.sleep(5)

    # save_path = model_store_path()
    # './model_store'
    save_path = './model_store'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, "detections_%d.pkl" % int(time.time()))
    with open(save_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    # save_file = './model_store/detections_1588751332.pkl'
    gen_rnet_sample_data(data_dir, anno_file, save_file, prefix_path)



def gen_rnet_sample_data(data_dir, anno_file, det_boxs_file, prefix_path):

    """
    :param data_dir:
    :param anno_file: original annotations file of wider face data
    :param det_boxs_file: detection boxes file
    :param prefix_path:
    :return:
    """

    neg_save_dir = os.path.join(data_dir, "24/negative")
    pos_save_dir = os.path.join(data_dir, "24/positive")
    part_save_dir = os.path.join(data_dir, "24/part")


    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        # print(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image

    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    image_size = 24
    net = "rnet"

    im_idx_list = list()
    gt_boxes_list = list()
    num_of_images = len(annotations)
    print ("processing %d images in total" % num_of_images)

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_idx = os.path.join(prefix_path, annotation[0])
        # im_idx = annotation[0]

        boxes = list(map(float, annotation[1:]))
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)


    # './anno_store'
    save_path = './anno_store'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f1 = open(os.path.join(save_path, 'pos_%d.txt' % image_size), 'w')
    f2 = open(os.path.join(save_path, 'neg_%d.txt' % image_size), 'w')
    f3 = open(os.path.join(save_path, 'part_%d.txt' % image_size), 'w')

    # print(det_boxs_file)
    det_handle = open(det_boxs_file, 'rb')

    det_boxes = cPickle.load(det_handle)

    # an image contain many boxes stored in an array
    print(len(det_boxes), num_of_images)
    # assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):

        # if (im_idx+1) == 100:
            # break

        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if gts.shape[0]==0:
            continue
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        # change to square
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            # Iou with all gts must below 0.3
            if np.max(Iou) < 0.3 and neg_num < 60:
                # save the examples
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                # print(save_file)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    f1.close()
    f2.close()
    f3.close()

def model_store_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/model_store"

def get_Rnet_data(pnet_model):
    gen_rnet_data(traindata_store, annotation_file, pnet_model_file = pnet_model, prefix_path = prefix_path, use_cuda = True)


def assembel_Rnet_data():
    anno_list = []

    anno_list.append(rnet_postive_file)
    anno_list.append(rnet_part_file)
    anno_list.append(rnet_neg_file)
    # anno_list.append(pnet_landmark_file)

    chose_count = assemble_data(imglist_filename_rnet ,anno_list)
    print("RNet train annotation result file path:%s" % imglist_filename_rnet)
#-----------------------------------------------------------------------------------------------------------------------------------------------#
def gen_onet_data(data_dir, anno_file, pnet_model_file, rnet_model_file, prefix_path='', use_cuda=True, vis=False):


    pnet, rnet, _ = create_mtcnn_net(p_model_path=pnet_model_file, r_model_path=rnet_model_file, use_cuda=use_cuda)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, min_face_size=12)

    imagedb = ImageDB(anno_file,mode="test",prefix_path=prefix_path)
    imdb = imagedb.load_imdb()
    image_reader = TestImageLoader(imdb,1,False)

    all_boxes = list()
    batch_idx = 0

    print('size:%d' % image_reader.size)
    for databatch in image_reader:
        if batch_idx % 50 == 0:
            print("%d images done" % batch_idx)

        im = databatch

        t = time.time()

        # pnet detection = [x1, y1, x2, y2, score, reg]
        p_boxes, p_boxes_align = mtcnn_detector.detect_pnet(im=im)

        t0 = time.time() - t
        t = time.time()
        # rnet detection
        boxes, boxes_align = mtcnn_detector.detect_rnet(im=im, dets=p_boxes_align)

        t1 = time.time() - t
        print('cost time pnet--',t0,'  rnet--',t1)
        t = time.time()

        if boxes_align is None:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue
        if vis:
            rgb_im = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
            vision.vis_two(rgb_im, boxes, boxes_align)

        
        all_boxes.append(boxes_align)
        batch_idx += 1

    save_path = './model_store'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, "detections_%d.pkl" % int(time.time()))
    with open(save_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)


    gen_onet_sample_data(data_dir,anno_file,save_file,prefix_path)



def gen_onet_sample_data(data_dir,anno_file,det_boxs_file,prefix):

    neg_save_dir = os.path.join(data_dir, "48/negative")
    pos_save_dir = os.path.join(data_dir, "48/positive")
    part_save_dir = os.path.join(data_dir, "48/part")

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image

    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    image_size = 48
    net = "onet"

    im_idx_list = list()
    gt_boxes_list = list()
    num_of_images = len(annotations)
    print("processing %d images in total" % num_of_images)

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_idx = os.path.join(prefix,annotation[0])

        boxes = list(map(float, annotation[1:]))
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)

    save_path = './anno_store'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f1 = open(os.path.join(save_path, 'pos_%d.txt' % image_size), 'w')
    f2 = open(os.path.join(save_path, 'neg_%d.txt' % image_size), 'w')
    f3 = open(os.path.join(save_path, 'part_%d.txt' % image_size), 'w')

    det_handle = open(det_boxs_file, 'rb')

    det_boxes = cPickle.load(det_handle)
    print(len(det_boxes), num_of_images)
    # assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1
        if gts.shape[0]==0:
            continue
        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        for box in dets:
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    f1.close()
    f2.close()
    f3.close()



def model_store_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/model_store"


def get_Onet_data(pnet_model, rnet_model):
    gen_onet_data(traindata_store, annotation_file, pnet_model_file = pnet_model, rnet_model_file = rnet_model,prefix_path=prefix_path,use_cuda = True, vis = False)


def assembel_Onet_data():
    anno_list = []

    anno_list.append(onet_postive_file)
    anno_list.append(onet_part_file)
    anno_list.append(onet_neg_file)
    anno_list.append(onet_landmark_file)

    chose_count = assemble_data(imglist_filename_onet ,anno_list)
    print("ONet train annotation result file path:%s" % imglist_filename_onet)


def gen_landmark_48(anno_file, data_dir, prefix = ''):


    size = 48
    image_id = 0

    landmark_imgs_save_dir = os.path.join(data_dir,"48/landmark")
    if not os.path.exists(landmark_imgs_save_dir):
        os.makedirs(landmark_imgs_save_dir)

    anno_dir = './anno_store'
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)

    landmark_anno_filename = "landmark_48.txt"
    save_landmark_anno = os.path.join(anno_dir,landmark_anno_filename)

    # print(save_landmark_anno)
    # time.sleep(5)
    f = open(save_landmark_anno, 'w')
    # dstdir = "train_landmark_few"

    with open(anno_file, 'r') as f2:
        annotations = f2.readlines()

    num = len(annotations)
    print("%d total images" % num)

    l_idx =0
    idx = 0
    # image_path bbox landmark(5*2)
    for annotation in annotations:
        # print imgPath

        annotation = annotation.strip().split(' ')

        assert len(annotation)==15,"each line should have 15 element"

        im_path = os.path.join('./data_set/face_landmark/CNN_FacePoint/train/',annotation[0].replace("\\", "/"))

        gt_box = list(map(float, annotation[1:5]))
        # gt_box = [gt_box[0], gt_box[2], gt_box[1], gt_box[3]]


        gt_box = np.array(gt_box, dtype=np.int32)

        landmark = list(map(float, annotation[5:]))
        landmark = np.array(landmark, dtype=np.float)

        img = cv2.imread(im_path)
        # print(im_path)
        assert (img is not None)

        height, width, channel = img.shape
        # crop_face = img[gt_box[1]:gt_box[3]+1, gt_box[0]:gt_box[2]+1]
        # crop_face = cv2.resize(crop_face,(size,size))

        idx = idx + 1
        if idx % 100 == 0:
            print("%d images done, landmark images: %d"%(idx,l_idx))
        # print(im_path)
        # print(gt_box)
        x1, x2, y1, y2 = gt_box
        gt_box[1] = y1
        gt_box[2] = x2
        # time.sleep(5)

        # gt's width
        w = x2 - x1 + 1
        # gt's height
        h = y2 - y1 + 1
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        # random shift
        for i in range(10):
            bbox_size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            delta_x = np.random.randint(-w * 0.2, w * 0.2)
            delta_y = np.random.randint(-h * 0.2, h * 0.2)
            nx1 = max(x1 + w / 2 - bbox_size / 2 + delta_x, 0)
            ny1 = max(y1 + h / 2 - bbox_size / 2 + delta_y, 0)

            nx2 = nx1 + bbox_size
            ny2 = ny1 + bbox_size
            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            cropped_im = img[int(ny1):int(ny2) + 1, int(nx1):int(nx2) + 1, :]
            resized_im = cv2.resize(cropped_im, (size, size),interpolation=cv2.INTER_LINEAR)

            offset_x1 = (x1 - nx1) / float(bbox_size)
            offset_y1 = (y1 - ny1) / float(bbox_size)
            offset_x2 = (x2 - nx2) / float(bbox_size)
            offset_y2 = (y2 - ny2) / float(bbox_size)

            offset_left_eye_x = (landmark[0] - nx1) / float(bbox_size)
            offset_left_eye_y = (landmark[1] - ny1) / float(bbox_size)

            offset_right_eye_x = (landmark[2] - nx1) / float(bbox_size)
            offset_right_eye_y = (landmark[3] - ny1) / float(bbox_size)

            offset_nose_x = (landmark[4] - nx1) / float(bbox_size)
            offset_nose_y = (landmark[5] - ny1) / float(bbox_size)

            offset_left_mouth_x = (landmark[6] - nx1) / float(bbox_size)
            offset_left_mouth_y = (landmark[7] - ny1) / float(bbox_size)

            offset_right_mouth_x = (landmark[8] - nx1) / float(bbox_size)
            offset_right_mouth_y = (landmark[9] - ny1) / float(bbox_size)


            # cal iou
            iou = IoU(crop_box.astype(np.float), np.expand_dims(gt_box.astype(np.float), 0))
            # print(iou)
            if iou > 0.65:
                save_file = os.path.join(landmark_imgs_save_dir, "%s.jpg" % l_idx)
                cv2.imwrite(save_file, resized_im)

                f.write(save_file + ' -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n' % \
                (offset_x1, offset_y1, offset_x2, offset_y2, \
                offset_left_eye_x,offset_left_eye_y,offset_right_eye_x,offset_right_eye_y,offset_nose_x,offset_nose_y,offset_left_mouth_x,offset_left_mouth_y,offset_right_mouth_x,offset_right_mouth_y))
                # print(save_file)
                # print(save_landmark_anno)
                l_idx += 1

    f.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Get data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--net', dest='net', help='which net to show', type=str)
    parser.add_argument('--pnet_path', default="./original_model/pnet_epoch.pt",help='path to pnet model', type=str)
    parser.add_argument('--rnet_path', default="./original_model/rnet_epoch.pt",help='path to rnet model', type=str)
    parser.add_argument('--use_cuda', default=True,help='use cuda', type=bool)

    args = parser.parse_args()
    return args

#-----------------------------------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    args = parse_args()
    if args.net == "pnet":
        wider_face(txt_from_path, anno_file)
        get_Pnet_data()
        assembel_Pnet_data()
    elif args.net == "rnet":
        get_Rnet_data(args.pnet_path)
        assembel_Rnet_data()
    elif args.net == "onet":
        get_Onet_data(args.pnet_path, args.rnet_path)
        gen_landmark_48(annotation_file_lm, traindata_store, prefix_path_lm)
>>>>>>> 16975964b37895a84e3935a9ee4a3a5c8f57ef49
        assembel_Onet_data()