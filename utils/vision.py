from matplotlib.patches import Circle
import os
import sys
import matplotlib.pyplot as plt
import pylab
sys.path.append(os.getcwd())


def vis_face(im_array, dets, landmarks, face_size, save_name):
    """Visualize detection results

    Parameters:
    ----------
    im_array: numpy.ndarray, shape(1, c, h, w)
        test image in rgb
    dets1: numpy.ndarray([[x1 y1 x2 y2 score]])
        detection results before calibration
    dets2: numpy.ndarray([[x1 y1 x2 y2 score]])
        detection results after calibration
    thresh: float
        boxes with scores > thresh will be drawn in red otherwise yellow

    Returns:
    -------
    """

    pylab.imshow(im_array)

    for i in range(dets.shape[0]):
        bbox = dets[i, :5]

        rect = pylab.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor='red', linewidth=0.9)
        score = bbox[4]
        plt.gca().text(bbox[0], bbox[1] - 2,
                       '{:.5f}'.format(score),
                       bbox=dict(facecolor='red', alpha=0.5), fontsize=8, color='white')

        pylab.gca().add_patch(rect)

    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            landmarks_one = landmarks[i, :]
            landmarks_one = landmarks_one.reshape((5, 2))
            for j in range(5):

                cir1 = Circle(xy=(landmarks_one[j, 0], landmarks_one[j, 1]), radius=face_size/12, alpha=0.4, color="red")
                pylab.gca().add_patch(cir1)

        pylab.savefig(save_name)
        pylab.show()