
import numpy as np


def whctrs(anchor):
    """
    将anchor的表示形式由（左上角坐标，右下角坐标）转换为（宽，高，中心点）的形式
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)

    return w, h, x_ctr, y_ctr
    
def mkanchors(ws, hs, x_ctr, y_ctr):
    """
    将（宽，高，中心点）的表示形式转换为（左上角坐标，右上角坐标）
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]

    left = x_ctr - 0.5 * (ws - 1)
    top = y_ctr - 0.5 * (hs - 1)
    right = x_ctr + 0.5 * (ws - 1)
    bottom = y_ctr + 0.5 * (hs - 1)

    anchors = np.hstack((left, top, right, bottom))
    return anchors

def ratio_enum(anchor, ratios):
    """
        给出 base anchor，在base anchor的基础上，
        根据不同的ratio，生成相应的anchor

        ratio = h / w ,h对应着y轴，w对应x轴
    """

    w, h, x_ctr, y_ctr = whctrs(anchor)
    area = w * h  # 面积
    ws = np.round(np.sqrt(area / ratios))  
    hs = np.round(ws * ratios)

    anchors = mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def scale_enum(anchor, scales):
    """
    从anchor中生成三种不同的尺度128*128,256 * 256,512 * 512
    """
    w, h, x_ctr, y_ctr = whctrs(anchor)

    ws = w * scales # 相对于 16 * 16 的倍数，128 -> 8,256 -> 16,512 -> 32
    hs = h * scales

    anchors = mkanchors(ws, hs, x_ctr, y_ctr)
    
    return anchors

def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=np.array([8, 16, 32])):
    
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = ratio_enum(base_anchor, ratios)
    
    #print(ratio_anchors)

    anchors = np.vstack([scale_enum(ratio_anchors[i,:], scales) for i in range(ratio_anchors.shape[0])])
    
    return anchors


if __name__ == '__main__':
    
    base_anchor_box = generate_anchors()
    print(base_anchor_box)

    feat_width = 2
    feat_height = 2

    stride = 16

    shift_x = np.arange(0, feat_width) * 16
    shift_y = np.arange(0, feat_height) * 16
    
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    print(shifts)

    anchors = np.vstack([a + base_anchor_box for a in shifts])

    anchors = anchors.reshape(4,36)
    print(anchors)


    a = anchors[0,:4]
    b = anchors[1,:4]



    a = whctrs(a)
    b = whctrs(b)

    print(a)
    print(b)