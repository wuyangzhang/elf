import torch
from torch import nn

mse_loss = nn.MSELoss()


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, 0] * wh[:, 1]  # [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


def cal_loss(x, y):
    if x.shape[0] != y.shape[0]:
        if x.shape[0] < y.shape[0]:
            pad = torch.zeros([(y.shape[0] - x.shape[0]), 4]).cuda()
            x = torch.cat([x, pad])
        else:
            pad = torch.zeros([(x.shape[0] - y.shape[0]), 4]).cuda()
            y = torch.cat([y, pad])
    loss_x1 = mse_loss(x[:, 0], y[:, 0])
    loss_y1 = mse_loss(x[:, 1], y[:, 1])
    loss_x2 = mse_loss(x[:, 2], y[:, 2])
    loss_y2 = mse_loss(x[:, 3], y[:, 3])
    iou = box_iou(x, y)
    iou_loss = torch.mean(1 - iou)
    metrics = {'loss_x1': loss_x1.item(),
               'loss_y1': loss_y1.item(),
               'loss_x2': loss_x2.item(),
               'loss_y2': loss_y2.item(),
               'loss_iou': iou_loss.item()
               }
    return loss_x1 + loss_x2 + loss_y1 + loss_y2 + iou_loss, metrics


def cal_loss1(x, target):
    nonpad = target[:, :, 0] + target[:, :, 1] + target[:, :, 2] + target[:, :, 3] != 0
    target = target[nonpad]
    # x, target shape : batch, 32, 4
    x1 = torch.max(x[:, 0], target[:, 0])
    y1 = torch.max(x[:, 1], target[:, 1])
    x2 = torch.min(x[:, 2], target[:, 2])
    y2 = torch.min(x[:, 3], target[:, 3])

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    # pred_area = torch.max((x[:, :, 2] - x[:, :, 0]) * (x[:, :, 3] - x[:, :, 1]), torch.tensor([0.]).cuda())
    pred_area = (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])

    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    # union = torch.max(pred_area + target_area - inter_area, torch.tensor([0.]).cuda())
    union = pred_area + target_area - inter_area
    iou = inter_area / (torch.tensor([1e-6]).cuda() + union)
    iou_loss = torch.mean(1 - iou)

    loss_x1 = mse_loss(x[:, 0], target[:, 0])
    loss_y1 = mse_loss(x[:, 1], target[:, 1])
    loss_x2 = mse_loss(x[:, 2], target[:, 2])
    loss_y2 = mse_loss(x[:, 3], target[:, 3])

    total_loss = loss_x1 + loss_x2 + loss_y1 + loss_y2 + iou_loss

    return total_loss


def cal_loss2(x, y):
    def call1(x, y):
        y_non_pad = y.sum(dim=1) != 0.
        y = y[y_non_pad]
        x_non_pad = x.sum(dim=1) != 0.
        x = x[x_non_pad]

        x1_diff = torch.abs(x[:, None][..., 0] - y[..., 0]).min(dim=1)
        y1_diff = torch.abs(x[:, None][..., 1] - y[..., 1]).min(dim=1)
        x2_diff = torch.abs(x[:, None][..., 2] - y[..., 2]).min(dim=1)
        y2_diff = torch.abs(x[:, None][..., 3] - y[..., 3]).min(dim=1)

        x1_loss = torch.mean(x1_diff[0])
        y1_loss = torch.mean(y1_diff[0])
        x2_loss = torch.mean(x2_diff[0])
        y2_loss = torch.mean(y2_diff[0])
        return x1_loss + y1_loss + x2_loss + y2_loss

    def call(x, y):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

        Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

        Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
        partial code is taken from https://pytorch.org/docs/stable/_modules/torchvision/ops/boxes.html

        as most of rp are zero.. so if you are doing a shitting job, it won't feel it by averaging them.
        """

        # mask those empty boxes.
        y_non_pad = y.sum(dim=1) != 0.
        y = y[y_non_pad]
        x_non_pad = x.sum(dim=1) != 0.
        x = x[x_non_pad]

        x_area = (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
        y_area = (y[:, 2] - y[:, 0]) * (y[:, 3] - y[:, 1])

        lt = torch.max(x[:, None, :2], y[:, :2])
        rb = torch.min(x[:, None, :2], y[:, :2])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        inter = inter.clamp(min=0)
        iou = inter / (x_area[:, None] + y_area - inter)

        '''indices indicates the best matched bbox of x for each of y'''
        iou, indices = iou.max(dim=0)
        iou_loss = torch.mean(1 - iou)

        x1_loss = torch.abs(x[:, 0][indices] - y[:, 0])
        x1_loss = torch.mean(x1_loss)

        y1_loss = torch.abs(x[:, 1][indices] - y[:, 1])
        y1_loss = torch.mean(y1_loss)

        x2_loss = torch.abs(x[:, 2][indices] - y[:, 2])
        x2_loss = torch.mean(x2_loss)

        y2_loss = torch.abs(x[:, 3][indices] - y[:, 3])
        y2_loss = torch.mean(y2_loss)

        total_loss = x1_loss + y1_loss + x2_loss + y2_loss + iou_loss
        return total_loss

    loss = []
    for batch in range(x.shape[0]):
        ans1 = call(x[batch], y[batch])
        # ans = call(y[batch], x[batch]) + ans1
        loss.append(ans1)
    return torch.mean(torch.stack(loss))
