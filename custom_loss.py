import torch
import torch.nn as nn
import torch.nn.functional as F

class JiedianLossFunction(nn.Module):
    """
    实现截点与坏果关联的自定义损失函数
    
    将截点视为坏果的属性，而非独立类别，
    仅当检测到坏果时计算截点位置损失。
    """
    def __init__(self, bad_fruit_id=1, jiedian_id=4, 
                 jiedian_loss_weight=1.5, bad_fruit_loss_weight=1.2):
        """
        初始化截点关联损失函数
        
        参数:
            bad_fruit_id (int): 坏果类别ID
            jiedian_id (int): 截点类别ID
            jiedian_loss_weight (float): 截点损失权重
            bad_fruit_loss_weight (float): 坏果损失权重
        """
        super(JiedianLossFunction, self).__init__()
        self.bad_fruit_id = bad_fruit_id
        self.jiedian_id = jiedian_id
        self.jiedian_loss_weight = jiedian_loss_weight
        self.bad_fruit_loss_weight = bad_fruit_loss_weight
    
    def forward(self, predictions, targets):
        """
        计算自定义损失
        
        参数:
            predictions (tensor): 模型预测值，格式为 [batch_size, num_anchors, num_classes + 5]
            targets (tensor): 目标值，格式为 [batch_size, num_anchors, num_classes + 5]
            
        返回:
            tensor: 计算的总损失值
        """
        # 提取预测框和目标框的坐标
        pred_boxes = predictions[..., :4]  # [x, y, w, h]
        target_boxes = targets[..., :4]    # [x, y, w, h]
        
        # 提取预测和目标类别概率
        pred_cls = predictions[..., 4:4+targets.shape[-1]-5]  # 类别概率
        target_cls = targets[..., 4:4+targets.shape[-1]-5]    # 类别概率
        
        # 计算基本的定位损失 (CIoU损失)
        box_loss = self.ciou_loss(pred_boxes, target_boxes)
        
        # 计算基本的分类损失 (二元交叉熵损失)
        cls_loss = F.binary_cross_entropy_with_logits(pred_cls, target_cls)
        
        # 创建坏果和截点的掩码
        bad_fruit_mask = (target_cls[..., self.bad_fruit_id] > 0.5)  # 坏果标签
        jiedian_mask = (target_cls[..., self.jiedian_id] > 0.5)      # 截点标签
        
        # 增加坏果分类损失权重
        bad_fruit_loss = F.binary_cross_entropy_with_logits(
            pred_cls[..., self.bad_fruit_id], 
            target_cls[..., self.bad_fruit_id]
        )
        
        # 计算截点关联损失：仅当目标是坏果时计算截点损失
        # 筛选出坏果对应的框
        if torch.any(bad_fruit_mask):
            bad_fruit_boxes = pred_boxes[bad_fruit_mask]
            bad_fruit_targets = target_boxes[bad_fruit_mask]
            
            # 提高坏果定位精度
            bad_fruit_box_loss = self.ciou_loss(bad_fruit_boxes, bad_fruit_targets)
            
            # 对应的截点定位损失（如果有截点标签）
            if torch.any(jiedian_mask):
                jiedian_boxes = pred_boxes[jiedian_mask]
                jiedian_targets = target_boxes[jiedian_mask]
                
                # 对坏果和截点进行空间关联
                jiedian_box_loss = self.ciou_loss(jiedian_boxes, jiedian_targets)
                
                # 计算坏果与截点的空间关系一致性损失
                # 这里我们希望截点与坏果在空间上有一定的关联关系
                jiedian_spatial_loss = self.spatial_consistency_loss(
                    pred_boxes, target_boxes, bad_fruit_mask, jiedian_mask
                )
                
                # 加权组合损失
                total_loss = (
                    cls_loss + 
                    box_loss + 
                    self.bad_fruit_loss_weight * bad_fruit_loss + 
                    self.bad_fruit_loss_weight * bad_fruit_box_loss + 
                    self.jiedian_loss_weight * jiedian_box_loss +
                    self.jiedian_loss_weight * jiedian_spatial_loss
                )
            else:
                # 如果没有截点标签，则不计算截点相关损失
                total_loss = (
                    cls_loss + 
                    box_loss + 
                    self.bad_fruit_loss_weight * bad_fruit_loss + 
                    self.bad_fruit_loss_weight * bad_fruit_box_loss
                )
        else:
            # 如果没有坏果标签，则只计算基本损失
            total_loss = cls_loss + box_loss
        
        return total_loss
    
    def ciou_loss(self, pred_boxes, target_boxes, eps=1e-7):
        """
        计算Complete IoU损失

        参数:
            pred_boxes (tensor): 预测框 [x, y, w, h]
            target_boxes (tensor): 目标框 [x, y, w, h]
            eps (float): 数值稳定性常数
            
        返回:
            tensor: CIoU损失值
        """
        # 转换为 [x1, y1, x2, y2] 格式
        pred_x1y1 = pred_boxes[..., :2] - pred_boxes[..., 2:] / 2
        pred_x2y2 = pred_boxes[..., :2] + pred_boxes[..., 2:] / 2
        target_x1y1 = target_boxes[..., :2] - target_boxes[..., 2:] / 2
        target_x2y2 = target_boxes[..., :2] + target_boxes[..., 2:] / 2
        
        # 计算交集区域
        inter_x1 = torch.max(pred_x1y1[..., 0], target_x1y1[..., 0])
        inter_y1 = torch.max(pred_x1y1[..., 1], target_x1y1[..., 1])
        inter_x2 = torch.min(pred_x2y2[..., 0], target_x2y2[..., 0])
        inter_y2 = torch.min(pred_x2y2[..., 1], target_x2y2[..., 1])
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # 计算并集区域
        pred_area = pred_boxes[..., 2] * pred_boxes[..., 3]
        target_area = target_boxes[..., 2] * target_boxes[..., 3]
        union_area = pred_area + target_area - inter_area + eps
        
        # 计算IoU
        iou = inter_area / union_area
        
        # 计算中心点距离
        pred_center = pred_boxes[..., :2]
        target_center = target_boxes[..., :2]
        center_dist_squared = torch.sum((pred_center - target_center) ** 2, dim=-1)
        
        # 计算包围两个框的最小矩形
        enclosing_x1 = torch.min(pred_x1y1[..., 0], target_x1y1[..., 0])
        enclosing_y1 = torch.min(pred_x1y1[..., 1], target_x1y1[..., 1])
        enclosing_x2 = torch.max(pred_x2y2[..., 0], target_x2y2[..., 0])
        enclosing_y2 = torch.max(pred_x2y2[..., 1], target_x2y2[..., 1])
        
        enclosing_w = enclosing_x2 - enclosing_x1
        enclosing_h = enclosing_y2 - enclosing_y1
        enclosing_diagonal_squared = enclosing_w ** 2 + enclosing_h ** 2 + eps
        
        # 计算宽高比一致性项
        pred_aspect_ratio = pred_boxes[..., 2] / (pred_boxes[..., 3] + eps)
        target_aspect_ratio = target_boxes[..., 2] / (target_boxes[..., 3] + eps)
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(target_aspect_ratio) - torch.atan(pred_aspect_ratio), 2
        )
        
        # 计算CIoU
        alpha = v / (1 - iou + v + eps)
        ciou = iou - (center_dist_squared / enclosing_diagonal_squared + alpha * v)
        
        return 1 - ciou
    
    def spatial_consistency_loss(self, pred_boxes, target_boxes, bad_fruit_mask, jiedian_mask):
        """
        计算坏果和截点之间的空间一致性损失
        
        参数:
            pred_boxes (tensor): 预测框
            target_boxes (tensor): 目标框
            bad_fruit_mask (tensor): 坏果掩码
            jiedian_mask (tensor): 截点掩码
            
        返回:
            tensor: 空间一致性损失
        """
        # 如果没有坏果或截点，则返回零损失
        if not torch.any(bad_fruit_mask) or not torch.any(jiedian_mask):
            return torch.tensor(0.0, device=pred_boxes.device)
        
        # 获取坏果和截点的预测与目标位置
        pred_bad_fruit_boxes = pred_boxes[bad_fruit_mask]
        target_bad_fruit_boxes = target_boxes[bad_fruit_mask]
        pred_jiedian_boxes = pred_boxes[jiedian_mask]
        target_jiedian_boxes = target_boxes[jiedian_mask]
        
        # 计算预测的坏果与截点距离
        pred_bad_fruit_centers = pred_bad_fruit_boxes[..., :2]
        pred_jiedian_centers = pred_jiedian_boxes[..., :2]
        
        # 为每个坏果找到最近的截点
        spatial_losses = []
        
        for i, bad_fruit_center in enumerate(pred_bad_fruit_centers):
            # 计算当前坏果与所有截点的距离
            dists = torch.sum((bad_fruit_center.unsqueeze(0) - pred_jiedian_centers) ** 2, dim=-1)
            min_dist_idx = torch.argmin(dists)
            
            # 获取目标中对应的坏果和截点位置
            target_bad_fruit_center = target_bad_fruit_boxes[i, :2]
            target_jiedian_center = target_jiedian_boxes[min_dist_idx, :2]
            
            # 计算目标中坏果与截点的距离向量
            target_vector = target_jiedian_center - target_bad_fruit_center
            
            # 计算预测中坏果与最近截点的距离向量
            pred_vector = pred_jiedian_centers[min_dist_idx] - bad_fruit_center
            
            # 计算向量差异作为空间一致性损失
            # 我们希望预测的空间关系与目标中的空间关系一致
            vector_diff = torch.sum((pred_vector - target_vector) ** 2)
            spatial_losses.append(vector_diff)
        
        if spatial_losses:
            # 返回平均空间一致性损失
            return torch.mean(torch.stack(spatial_losses))
        else:
            return torch.tensor(0.0, device=pred_boxes.device)

def create_custom_loss(model_config):
    """
    根据模型配置创建自定义损失函数
    
    参数:
        model_config (dict): 模型配置字典
        
    返回:
        JiedianLossFunction: 自定义损失函数实例
    """
    if 'jiedian_loss' in model_config and model_config['jiedian_loss'].get('enabled', False):
        jiedian_config = model_config['jiedian_loss']
        return JiedianLossFunction(
            bad_fruit_id=jiedian_config.get('bad_fruit_id', 1),
            jiedian_id=jiedian_config.get('jiedian_id', 4),
            jiedian_loss_weight=jiedian_config.get('jiedian_loss_weight', 1.5),
            bad_fruit_loss_weight=jiedian_config.get('bad_fruit_loss_weight', 1.2)
        )
    return None

# 后处理函数，用于筛选截点，只保留与坏果相关的截点
def filter_jiedian_boxes(prediction, bad_fruit_id=1, jiedian_id=4, iou_threshold=0.5, conf_ratio=0.8):
    """
    过滤截点检测结果，只保留与坏果关联的截点
    
    参数:
        prediction (list): 检测结果列表，每个元素为一张图像的检测结果
        bad_fruit_id (int): 坏果类别ID
        jiedian_id (int): 截点类别ID
        iou_threshold (float): IoU阈值，用于确定截点与坏果的关联性
        conf_ratio (float): 截点置信度比例，相对于坏果置信度
        
    返回:
        list: 过滤后的检测结果
    """
    filtered_predictions = []
    
    for pred in prediction:
        # 如果没有检测结果，则跳过
        if pred is None or len(pred) == 0:
            filtered_predictions.append(pred)
            continue
        
        # 分离坏果和截点检测结果
        bad_fruit_boxes = []
        jiedian_boxes = []
        other_boxes = []
        
        for det in pred:
            if det[-1] == bad_fruit_id:
                bad_fruit_boxes.append(det)
            elif det[-1] == jiedian_id:
                jiedian_boxes.append(det)
            else:
                other_boxes.append(det)
        
        # 转换为张量形式，方便处理
        bad_fruit_boxes = torch.stack(bad_fruit_boxes) if bad_fruit_boxes else torch.zeros((0, pred.shape[1]), device=pred.device)
        jiedian_boxes = torch.stack(jiedian_boxes) if jiedian_boxes else torch.zeros((0, pred.shape[1]), device=pred.device)
        other_boxes = torch.stack(other_boxes) if other_boxes else torch.zeros((0, pred.shape[1]), device=pred.device)
        
        # 筛选有效的截点：只保留与坏果关联的截点
        valid_jiedian_boxes = []
        
        if len(bad_fruit_boxes) > 0 and len(jiedian_boxes) > 0:
            for jiedian_box in jiedian_boxes:
                jiedian_conf = jiedian_box[4]
                
                # 计算截点与所有坏果的IoU
                max_iou = 0
                best_bad_fruit_idx = -1
                
                for i, bad_fruit_box in enumerate(bad_fruit_boxes):
                    # 计算IoU
                    iou = box_iou(jiedian_box[:4], bad_fruit_box[:4])
                    
                    if iou > max_iou:
                        max_iou = iou
                        best_bad_fruit_idx = i
                
                # 如果IoU超过阈值，则认为这个截点与坏果关联
                if max_iou >= iou_threshold:
                    # 调整截点的置信度为坏果置信度的一定比例
                    bad_fruit_conf = bad_fruit_boxes[best_bad_fruit_idx][4]
                    jiedian_box[4] = bad_fruit_conf * conf_ratio
                    valid_jiedian_boxes.append(jiedian_box)
        
        # 合并有效的截点、坏果和其他类别
        valid_jiedian_boxes = torch.stack(valid_jiedian_boxes) if valid_jiedian_boxes else torch.zeros((0, pred.shape[1]), device=pred.device)
        filtered_pred = torch.cat([bad_fruit_boxes, valid_jiedian_boxes, other_boxes], dim=0)
        
        filtered_predictions.append(filtered_pred)
    
    return filtered_predictions

def box_iou(box1, box2):
    """
    计算两个框的IoU
    
    参数:
        box1 (tensor): 第一个框 [x, y, w, h]
        box2 (tensor): 第二个框 [x, y, w, h]
        
    返回:
        float: IoU值
    """
    # 转换为 [x1, y1, x2, y2] 格式
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    
    # 计算交集区域
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    # 计算交集面积
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # 计算两个框的面积
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    # 计算IoU
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-8)
    
    return iou 