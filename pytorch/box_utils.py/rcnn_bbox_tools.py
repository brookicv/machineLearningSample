# -*- coding: utf-8 -*-


import numpy as np

def loc2bbox(src_bbox,loc):
    """
    为bbox添加偏移量和缩放尺度
    Args:
        src_bbox(array): bbox 的坐标,[p_{ymin},p_{xmin},p_{ymax},p_{xmax}]
        loc(array): bbox的偏移量和尺度,[t_y,t_x,t_h,t_w]
    """

    if src_bbox.shape[0] == 0:
        return np.zeros((0,4),dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype,copy=False)

    # bbox 的宽和高，以及中心位置坐标
    src_height = src_bbox[:,2] - src_bbox[:,0]
    src_width = src_bbox[:,3] - src_bbox[:,1]
    src_ctr_y = src_bbox[:,0] + 0.5 * src_height
    src_ctr_x = src_bbox[:,1] + 0.5 * src_width

    # 分别取出bbox的偏移量及其尺度
    dy = loc[:,0::4]
    dx = loc[:,1::4]
    dh = loc[:,2::4]
    dw = loc[:,3::4]

    """
    t_x = (x - x_a) / w_a
    t_y = (y - y_a) / h_a
    t_w = log(w / w_a)
    t_h = log(w / h_a)
    """
    ctr_y = dy * src_height[:,np.newaxis] + src_ctr_y[:,np.newaxis]
    ctr_x = dx * src_width[:,np.newaxis] + src_ctr_x[:,np.newaxis]
    h = np.exp(dh) * src_height[:,np.newaxis]
    w = np.exp(dw) * src_width[:,np.newaxis]

    # 转换为边框坐标的形式
    dst_bbox = np.zeros(loc.shape,dtype=loc.dtype)

    dst_bbox[:,0::4] = ctr_y - 0.5 * h
    dst_bbox[:,1::4] = ctr_x - 0.5 * w
    dst_bbox[:,2::4] = ctr_y + 0.5 * h
    dst_bbox[:,3::4] = ctr_x + 0.5 * w

    return dst_bbox

def bbox2loc(src_bbox,dst_bbox):
    """
    计算src_bbox和dst_bbox之间的loc
    """
    src_height = src_bbox[:,2] - src_bbox[:,0]
    src_width = src_bbox[:,3] - src_bbox[:,1]
    src_ctr_y = src_bbox[:,0] + 0.5 * src_height
    src_ctr_x = src_bbox[:,1] + 0.5 * src_width

    base_height = dst_bbox[:,2] - dst_bbox[:,0]
    base_width = dst_bbox[:,3] - dst_bbox[:,1]
    base_ctr_y = dst_bbox[:,0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:,1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height,eps)
    width  = np.maximum(width,eps)

    dy = (base_ctr_y - src_ctr_y) / height
    dx = (base_ctr_x - src_ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy,dx,dh,dw)).transpose()

    return  loc

def bbox_iou(bbox_a,bbox_b):

    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        return IndexError

    # top left
    tl = np.maximum(bbox_a[:,None,:2],bbox_b[:,None,:2])
    # bottom right
    br = np.maximum(bbox_a[:,None,2:],bbox_b[:,None,2:])

    area_i = np.prod(br - tl,axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:,2:] - bbox_a[:,:2],axis=1)
    area_b = np.prod(bbox_b[:,2:] - bbox_b[:,:2],axis=1)

    return area_i / (area_a[:,None] + area_b - area_i), 