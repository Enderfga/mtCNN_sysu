import cv2
import time
import numpy as np
import torch
from utils.models import PNet,RNet,ONet
import utils.tool as utils
from utils.dataloader import *


def create_mtcnn_net(p_model_path=None, r_model_path=None, o_model_path=None, use_cuda=True):

    pnet, rnet, onet = None, None, None

    if p_model_path is not None:
        pnet = PNet(use_cuda=use_cuda)
        if(use_cuda):
            print('p_model_path:{0}'.format(p_model_path))
            pnet.load_state_dict(torch.load(p_model_path))
            pnet.cuda()
        else:
            # forcing all GPU tensors to be in CPU while loading
            pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage))
        pnet.eval()

    if r_model_path is not None:
        rnet = RNet(use_cuda=use_cuda)
        if (use_cuda):
            print('r_model_path:{0}'.format(r_model_path))
            rnet.load_state_dict(torch.load(r_model_path))
            rnet.cuda()
        else:
            rnet.load_state_dict(torch.load(r_model_path, map_location=lambda storage, loc: storage))
        rnet.eval()

    if o_model_path is not None:
        onet = ONet(use_cuda=use_cuda)
        if (use_cuda):
            print('o_model_path:{0}'.format(o_model_path))
            onet.load_state_dict(torch.load(o_model_path))
            onet.cuda()
        else:
            onet.load_state_dict(torch.load(o_model_path, map_location=lambda storage, loc: storage))
        onet.eval()

    return pnet,rnet,onet




class MtcnnDetector(object):
    """
        P,R,O net face detection and landmarks align
    """
    def  __init__(self,
                 pnet = None,
                 rnet = None,
                 onet = None,
                 min_face_size=12,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.709,
                 ):

        self.pnet_detector = pnet
        self.rnet_detector = rnet
        self.onet_detector = onet
        self.min_face_size = min_face_size
        self.stride=stride
        self.thresh = threshold
        self.scale_factor = scale_factor


    def unique_image_format(self,im):
        if not isinstance(im,np.ndarray):
            if im.mode == 'I':
                im = np.array(im, np.int32, copy=False)
            elif im.mode == 'I;16':
                im = np.array(im, np.int16, copy=False)
            else:
                im = np.asarray(im)
        return im

    def square_bbox(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x m
                input bbox
        Returns:
        -------
            a square bbox
        """
        square_bbox = bbox.copy()

        # x2 - x1
        # y2 - y1
        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        l = np.maximum(h,w)
        # x1 = x1 + w*0.5 - l*0.5
        # y1 = y1 + h*0.5 - l*0.5
        square_bbox[:, 0] = bbox[:, 0] + w*0.5 - l*0.5
        square_bbox[:, 1] = bbox[:, 1] + h*0.5 - l*0.5

        # x2 = x1 + l - 1
        # y2 = y1 + l - 1
        square_bbox[:, 2] = square_bbox[:, 0] + l - 1
        square_bbox[:, 3] = square_bbox[:, 1] + l - 1
        return square_bbox


    def generate_bounding_box(self, map, reg, scale, threshold):
        """
            generate bbox from feature map
        Parameters:
        ----------
            map: numpy array , n x m x 1
                detect score for each position
            reg: numpy array , n x m x 4
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
        stride = 2
        cellsize = 12 # receptive field

        t_index = np.where(map > threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])

        # reg = (1, n, m, 4)
        # choose bounding box whose socre are larger than threshold
        dx1, dy1, dx2, dy2 = [reg[0, t_index[0], t_index[1], i] for i in range(4)]
        reg = np.array([dx1, dy1, dx2, dy2])
        score = map[t_index[0], t_index[1], 0]

        # hence t_index[1] means column, t_index[1] is the value of x
        # hence t_index[0] means row, t_index[0] is the value of y
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),            # x1 of prediction box in original image
                                 np.round((stride * t_index[0]) / scale),            # y1 of prediction box in original image
                                 np.round((stride * t_index[1] + cellsize) / scale), # x2 of prediction box in original image
                                 np.round((stride * t_index[0] + cellsize) / scale), # y2 of prediction box in original image
                                                                                     # reconstruct the box in original image
                                 score,
                                 reg,
                                 # landmarks
                                 ])

        return boundingbox.T


    def resize_image(self, img, scale):
        """
            resize image and transform dimention to [batchsize, channel, height, width]
        Parameters:
        ----------
            img: numpy array , height x width x channel
                input image, channels in BGR order here
            scale: float number
                scale factor of resize operation
        Returns:
        -------
            transformed image tensor , 1 x channel x height x width
        """
        height, width, channels = img.shape
        new_height = int(height * scale)     # resized new height
        new_width = int(width * scale)       # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
        return img_resized


    def pad(self, bboxes, w, h):
        """
            pad the the boxes
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
        # width and height
        tmpw = (bboxes[:, 2] - bboxes[:, 0] + 1).astype(np.int32)
        tmph = (bboxes[:, 3] - bboxes[:, 1] + 1).astype(np.int32)
        numbox = bboxes.shape[0]

        dx = np.zeros((numbox, ))
        dy = np.zeros((numbox, ))
        edx, edy  = tmpw.copy()-1, tmph.copy()-1
        # x, y: start point of the bbox in original image
        # ex, ey: end point of the bbox in original image
        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w-1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h-1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list


    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array
            input image array
            one batch

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """

        # original wider face data

        net_size = 12

        current_scale = float(net_size) / self.min_face_size    # find initial scale
        im_resized = self.resize_image(im, current_scale) # scale = 1.0
        current_height, current_width, _ = im_resized.shape

        # fcn
        all_boxes = list()
        i = 0
        while min(current_height, current_width) > net_size:
            # print(i)
            feed_imgs = []
            image_tensor = convert_image_to_tensor(im_resized)
            feed_imgs.append(image_tensor)
            feed_imgs = torch.stack(feed_imgs)
            feed_imgs.requires_grad = True

            if self.pnet_detector.use_cuda:
                feed_imgs = feed_imgs.cuda()

            # receptive field is 12×12
            # 12×12 --> score
            # 12×12 --> bounding box
            cls_map, reg = self.pnet_detector(feed_imgs)

            cls_map_np = convert_chwTensor_to_hwcNumpy(cls_map.cpu())
            reg_np = convert_chwTensor_to_hwcNumpy(reg.cpu())

            boxes = self.generate_bounding_box(cls_map_np[ 0, :, :], reg_np, current_scale, self.thresh[0])

            # generate pyramid images
            current_scale *= self.scale_factor # self.scale_factor = 0.709
            im_resized = self.resize_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue

            # non-maximum suppresion
            keep = utils.nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)
            # i+=1

        if len(all_boxes) == 0:
            return None, None

        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = utils.nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]

        # x2 - x1
        # y2 - y1
        bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        boxes = np.vstack([all_boxes[:,0],
                   all_boxes[:,1],
                   all_boxes[:,2],
                   all_boxes[:,3],
                   all_boxes[:,4],
                  ])

        boxes = boxes.T

        align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
        align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
        align_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
        align_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh

        # refine the boxes
        boxes_align = np.vstack([ align_topx,
                              align_topy,
                              align_bottomx,
                              align_bottomy,
                              all_boxes[:, 4],
                              ])
        boxes_align = boxes_align.T

        return boxes, boxes_align

    def detect_rnet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """
        # im: an input image
        h, w, c = im.shape

        if dets is None:
            return None,None

        dets = self.square_bbox(dets)
        # rounds
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        '''
        # helper for setting RNet batch size
        batch_size = self.rnet_detector.batch_size
        ratio = float(num_boxes) / batch_size
        if ratio > 3 or ratio < 0.3:
            print "You may need to reset RNet batch size if this info appears frequently, \
        face candidates:%d, current batch_size:%d"%(num_boxes, batch_size)
        '''

        cropped_ims_tensors = []
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = im[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            crop_im = cv2.resize(tmp, (24, 24))
            crop_im_tensor = convert_image_to_tensor(crop_im)
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = torch.stack(cropped_ims_tensors)
        feed_imgs.requires_grad = True

        if self.rnet_detector.use_cuda:
            feed_imgs = feed_imgs.cuda()

        cls_map, reg = self.rnet_detector(feed_imgs)

        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.thresh[1])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None

        keep = utils.nms(boxes, 0.7)

        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        boxes = np.vstack([ keep_boxes[:,0],
                              keep_boxes[:,1],
                              keep_boxes[:,2],
                              keep_boxes[:,3],
                              keep_cls[:,0],
                            ])

        align_topx = keep_boxes[:,0] + keep_reg[:,0] * bw
        align_topy = keep_boxes[:,1] + keep_reg[:,1] * bh
        align_bottomx = keep_boxes[:,2] + keep_reg[:,2] * bw
        align_bottomy = keep_boxes[:,3] + keep_reg[:,3] * bh

        boxes_align = np.vstack([align_topx,
                               align_topy,
                               align_bottomx,
                               align_bottomy,
                               keep_cls[:, 0],
                             ])

        boxes = boxes.T
        boxes_align = boxes_align.T

        return boxes, boxes_align

    def detect_onet(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes_align: numpy array
            boxes after calibration
        landmarks_align: numpy array
            landmarks after calibration

        """
        h, w, c = im.shape

        if dets is None:
            return None, None

        dets = self.square_bbox(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            # crop input image
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            crop_im = cv2.resize(tmp, (48, 48))
            crop_im_tensor = convert_image_to_tensor(crop_im)
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = torch.stack(cropped_ims_tensors)
        feed_imgs.requires_grad = True

        if self.rnet_detector.use_cuda:
            feed_imgs = feed_imgs.cuda()

        cls_map, reg, landmark = self.onet_detector(feed_imgs)

        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()
        landmark = landmark.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.thresh[2])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None

        keep = utils.nms(boxes, 0.7, mode="Minimum")

        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        keep_landmark = landmark[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        align_landmark_topx = keep_boxes[:, 0]
        align_landmark_topy = keep_boxes[:, 1]

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_cls[:, 0],
                                 ])

        boxes_align = boxes_align.T

        landmark =  np.vstack([
                                 align_landmark_topx + keep_landmark[:, 0] * bw,
                                 align_landmark_topy + keep_landmark[:, 1] * bh,
                                 align_landmark_topx + keep_landmark[:, 2] * bw,
                                 align_landmark_topy + keep_landmark[:, 3] * bh,
                                 align_landmark_topx + keep_landmark[:, 4] * bw,
                                 align_landmark_topy + keep_landmark[:, 5] * bh,
                                 align_landmark_topx + keep_landmark[:, 6] * bw,
                                 align_landmark_topy + keep_landmark[:, 7] * bh,
                                 align_landmark_topx + keep_landmark[:, 8] * bw,
                                 align_landmark_topy + keep_landmark[:, 9] * bh,
                                 ])

        landmark_align = landmark.T

        return boxes_align, landmark_align


    def detect_face(self,img):
        """Detect face over image
        """
        landmark_align =np.array([])

        t = time.time()

        # pnet
        if self.pnet_detector:
            boxes, p_boxes_align = self.detect_pnet(img)
            if p_boxes_align is None:
                return np.array([]), np.array([])

            t1 = time.time() - t
            t = time.time()

        # rnet
        if self.rnet_detector:
            boxes, r_boxes_align = self.detect_rnet(img, p_boxes_align)
            if r_boxes_align is None:
                return np.array([]), np.array([])

            t2 = time.time() - t
            t = time.time()

        # onet
        if self.onet_detector:
            o_boxes_align, landmark_align = self.detect_onet(img, r_boxes_align)
            if o_boxes_align is None:
                return np.array([]), np.array([])

            t3 = time.time() - t
            t = time.time()
            print("time cost " + '{:.3f}'.format(t1+t2+t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2, t3))

        return p_boxes_align, r_boxes_align, o_boxes_align, landmark_align
