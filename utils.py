import torch
import cv2
import numpy as np
from PatchMatchOrig import init_nnf, upSample_nnf, avg_vote, propagate


def ts2np(x):
    x = x.squeeze(0)
    x = x.cpu().numpy()
    x = x.transpose(1, 2, 0)
    return x


def np2ts(x, device):
    x = x.transpose(2, 0, 1)
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    x = x.to(device)
    return x


def blend(response, f_a, r_bp, alpha=0.8, tau=0.05,
          device=torch.device('cuda:1')):
    """
    :param response:
    :param f_a: feature map (either F_A or F_BP)
    :param r_bp: reconstructed feature (R_BP or R_A)
    :param alpha: scalar balance the ratio of content and style in new feature
    map
    :param tau: threshold, default: 0.05 (suggested in paper)
    :return: (f_a*W + r_bp*(1-W)) where W=alpha*(response>tau)

    Following the official implementation, I replace the sigmoid function
    (stated in paper) with indicator function
    """
    weight = (response > tau).type(f_a.type()) * alpha
    weight = weight.expand(1, f_a.size(1), weight.size(
        2), weight.size(3)).to(device)

    r_bp = r_bp.to(device)

    f_ap = f_a*weight + r_bp*(1. - weight)
    return f_ap


def normalize(feature_map):
    """

    :param feature_map: either F_a or F_bp
    :return:
    normalized feature map
    response
    """
    response = torch.sum(feature_map*feature_map, dim=1, keepdim=True)
    normed_feature_map = feature_map/torch.sqrt(response)

    # response should be scaled to (0, 1)
    response = (response-torch.min(response)) / \
        (torch.max(response)-torch.min(response))
    return normed_feature_map, response


def load_image(file_A, resizeRatio=1.0):
    ori_AL = cv2.imread(file_A)

    # resize
    if (ori_AL.shape[0] > 700):
        ratio = 700 / ori_AL.shape[0]
        ori_AL = cv2.resize(ori_AL, None, fx=ratio, fy=ratio,
                            interpolation=cv2.INTER_CUBIC)

    if (ori_AL.shape[1] > 700):
        ratio = 700 / ori_AL.shape[1]
        ori_AL = cv2.resize(ori_AL, None, fx=ratio, fy=ratio,
                            interpolation=cv2.INTER_CUBIC)

    if (ori_AL.shape[0] < 200):
        ratio = 700 / ori_AL.shape[0]
        ori_AL = cv2.resize(ori_AL, None, fx=ratio, fy=ratio,
                            interpolation=cv2.INTER_CUBIC)
    if (ori_AL.shape[1] < 200):
        ratio = 700 / ori_AL.shape[1]
        ori_AL = cv2.resize(ori_AL, None, fx=ratio, fy=ratio,
                            interpolation=cv2.INTER_CUBIC)

    if (ori_AL.shape[0]*ori_AL.shape[1] > 350000):
        ratio = np.sqrt(350000 / (ori_AL.shape[1]*ori_AL.shape[0]))
        ori_AL = cv2.resize(ori_AL, None, fx=ratio, fy=ratio,
                            interpolation=cv2.INTER_CUBIC)

    img_A = cv2.resize(ori_AL, (240, 240), fx=resizeRatio,
                       fy=resizeRatio, interpolation=cv2.INTER_CUBIC)
    print(img_A.shape)

    return img_A


def get_nnf_part(arr_A, arr_AP, arr_B, arr_BP, bbox_A, bbox_B):
    y0, x0, h0, w0 = bbox_A
    y1, x1, h1, w1 = bbox_B

    assert bbox_A[2:] == bbox_B[2:]

    part_A = arr_A[y0:y0+h0, x0:x0+w0, :]
    part_AP = arr_AP[y0:y0+h0, x0:x0+w0, :]
    part_B = arr_B[y1:y1+h1, x1:x1+w1, :]
    part_BP = arr_BP[y1:y1+h1, x1:x1+w1, :]


    ann_AB = np.zeros((2,h0,w0),dtype = 'int')
    ann_AB[0] = np.random.randint(w0, size=(h0, w0))
    ann_AB[1] = np.random.randint(h0, size=(h0, w0))

    ann_AB = propagate(ann_AB, part_A, part_AP, part_B, part_BP, 3, 5)
    ann_AB[0]+=(h1-h0)
    ann_AB[1]+=(w1-w0)


