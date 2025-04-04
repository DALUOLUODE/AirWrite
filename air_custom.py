import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import os
import platform
import time

# 用PIL绘制中文文本到OpenCV图像
def cv2_put_chinese_text(img, text, position, textColor=(0, 255, 0), textSize=30):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 创建一个可以在给定图像上绘制的对象
    draw = ImageDraw.Draw(img)

    # 根据操作系统选择字体路径
    if platform.system() == 'Windows':
        system_root = os.environ.get('SystemRoot', 'C:\\Windows')
        possible_fonts = [
            os.path.join(system_root, 'Fonts', 'simhei.ttf'),  # 黑体
            os.path.join(system_root, 'Fonts', 'simsun.ttc'),  # 宋体
            os.path.join(system_root, 'Fonts', 'msyh.ttc'),  # 微软雅黑
        ]
    elif platform.system() == 'Darwin':  # macOS
        possible_fonts = [
            '/System/Library/Fonts/STHeiti Light.ttc',  # macOS 自带黑体
            '/System/Library/Fonts/PingFang.ttc',  # 苹果自带苹方字体
            os.path.join(os.path.expanduser('~'), 'Library', 'Fonts', 'SimHei.ttf'),  # 用户目录下的黑体
        ]
    else:  # 其他系统（如 Linux）
        possible_fonts = [
            '/usr/share/fonts/truetype/SimHei.ttf',  # Linux 示例路径（需手动安装）
        ]

    # 尝试加载字体
    fontStyle = None
    for font_path in possible_fonts:
        if os.path.exists(font_path):
            try:
                fontStyle = ImageFont.truetype(font_path, textSize)
                break
            except:
                continue

    # 如果找不到中文字体，使用默认字体（可能不支持中文）
    if fontStyle is None:
        fontStyle = ImageFont.load_default()
        print("警告：未找到中文字体，可能无法正常显示中文")

    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)

    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


"""CNN架构模型定义"""
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 26)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 添加卷积和池化层序列
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_ROI(canvas):
    """获取包含绘制内容的ROI区域"""
    # 转换为黑底白字
    gray = cv2.bitwise_not(canvas)
    # 二值化
    ret, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("警告：未检测到任何轮廓")
        return gray  # 如果没有轮廓，返回整个画布

    # 找出最大的轮廓（可能是整个画布）和第二大的轮廓（可能是字符）
    areas = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        areas.append((area, i))

    areas.sort(reverse=True)
    print(f"找到 {len(areas)} 个轮廓，面积分别为: {[area for area, _ in areas[:3] if len(areas) >= 3]}")

    # 如果只有一个轮廓，使用它
    idx = 0 if len(areas) == 1 else 1

    if idx >= len(areas):
        print("警告：没有足够的轮廓")
        return gray

    x, y, w, h = cv2.boundingRect(contours[areas[idx][1]])
    print(f"ROI区域: x={x}, y={y}, w={w}, h={h}")

    # 适当扩大ROI范围，确保字符完整
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(w + 2 * padding, gray.shape[1] - x)
    h = min(h + 2 * padding, gray.shape[0] - y)

    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 0), 1)
    roi = gray[y:y + h, x:x + w]
    return roi


def character_prediction(roi, model):
    """预测绘制的字符"""
    print(f"ROI形状: {roi.shape}")
    # 调整大小为28x28，符合EMNIST数据集格式
    img = cv2.resize(roi, (28, 28))
    # 应用高斯模糊减少噪点
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = Image.fromarray(img)

    # 图像预处理
    normalize = transforms.Normalize(
        mean=[0.5],  # 修改为单通道
        std=[0.5]  # 修改为单通道
    )
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        normalize
    ])

    p_img = preprocess(img)

    # 进行预测
    model.eval()
    p_img = p_img.reshape([1, 1, 28, 28]).float()
    output = model(torch.transpose(p_img, 2, 3))
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds 

# 新增的手部检测相关函数
def get_background(cap, num_frames=30):
    """通过平均多帧图像获取背景模型"""
    print("正在获取背景模型...")
    background = None
    frame_shape = None
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面！")
            return None
            
        # 水平翻转图像
        frame = cv2.flip(frame, 1)
        
        # 调整帧大小为固定尺寸 (480, 640)
        frame = cv2.resize(frame, (640, 480))
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if background is None:
            background = gray.copy().astype("float")
            frame_shape = gray.shape
            print(f"初始化背景模型，形状: {frame_shape}")
            continue
            
        # 累加当前帧到背景模型
        cv2.accumulateWeighted(gray, background, 0.5)
        
        # 显示进度
        progress_frame = frame.copy()
        cv2.putText(progress_frame, f"获取背景: {i+1}/{num_frames}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('MediaPipe', progress_frame)
        cv2.waitKey(1)
    
    # 将背景模型转换为8位格式
    background = background.astype("uint8")
    print(f"背景模型获取完成！形状: {background.shape}")
    return background


def detect_hand(frame, background, threshold=20):
    """通过背景差分检测手部运动"""
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # 确保尺寸匹配
    if gray.shape != background.shape:
        print(f"尺寸不匹配: gray={gray.shape}, background={background.shape}")
        # 调整灰度图和背景图像大小为相同尺寸
        gray = cv2.resize(gray, (background.shape[1], background.shape[0]))
    
    # 计算当前帧与背景的差异
    diff = cv2.absdiff(gray, background)
    
    # 阈值处理，得到二值图像
    thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # 使用更多的膨胀和腐蚀操作，以获得更好的手形
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.erode(thresh, None, iterations=2)
    
    return thresh


def detect_skin(frame):
    """基于肤色检测手部区域"""
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 扩展肤色范围，更宽容的检测阈值
    lower_skin = np.array([0, 15, 60], dtype=np.uint8)
    upper_skin = np.array([30, 255, 255], dtype=np.uint8)
    
    # 创建肤色掩码
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # 添加更多的肤色范围 (处理不同肤色)
    lower_skin2 = np.array([160, 15, 60], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    
    # 合并掩码
    mask = cv2.bitwise_or(mask, mask2)
    
    # 应用形态学操作改善掩码质量
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    
    return mask


def get_hand_contour(binary_mask, min_area=3000):
    """获取二值图像中最大的轮廓（假设是手部）"""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 按轮廓面积排序（降序）
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 找出面积足够大的轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            return contour
    
    return None


def find_fingertip(contour, frame_shape, prev_tip=None):
    """改进的指尖检测算法，带平滑和多指尖识别"""
    if contour is None:
        return None
        
    # 计算轮廓的凸包
    hull = cv2.convexHull(contour)
    
    # 找出轮廓的最高点和最外点作为候选指尖
    candidate_points = []
    
    # 方法1: 寻找最高点（y坐标最小的点）
    min_y = float('inf')
    top_point = None
    
    for point in hull:
        if point[0][1] < min_y:
            min_y = point[0][1]
            top_point = (point[0][0], point[0][1])
    
    if top_point:
        candidate_points.append(top_point)
    
    # 方法2: 使用凸缺陷来找出可能的指尖
    # 计算凸缺陷
    try:
        defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
        
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                # 计算三个点之间的角度
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                
                try:
                    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                except:
                    continue
                
                # 如果角度小于90度，则可能是指尖
                if angle <= np.pi / 2:  # 90度
                    # 选择靠近图像上部的点
                    if start[1] < frame_shape[0] * 0.6:  # 只选择在图像上部60%的点
                        candidate_points.append(start)
                    if end[1] < frame_shape[0] * 0.6:
                        candidate_points.append(end)
    except:
        pass  # 如果凸缺陷计算失败，忽略错误
    
    # 如果没有候选点，退回到使用最高点
    if not candidate_points and top_point:
        candidate_points = [top_point]
    
    # 选择最可能的指尖点（优先使用最高的点）
    fingertip = None
    min_y = float('inf')
    
    for point in candidate_points:
        if point[1] < min_y:
            min_y = point[1]
            fingertip = point
    
    # 添加平滑处理，如果有前一个指尖位置
    if prev_tip is not None and fingertip is not None:
        # 计算新旧位置之间的距离
        dist = np.sqrt((fingertip[0] - prev_tip[0])**2 + (fingertip[1] - prev_tip[1])**2)
        
        # 如果距离太大（可能是突然跳变），使用平滑处理
        max_movement = 40  # 每帧允许的最大移动像素
        if dist > max_movement:
            # 使用前一个点和当前点的加权平均
            alpha = 0.7  # 平滑因子，越大越平滑，但也会有更多的延迟
            smoothed_x = int(alpha * prev_tip[0] + (1 - alpha) * fingertip[0])
            smoothed_y = int(alpha * prev_tip[1] + (1 - alpha) * fingertip[1])
            fingertip = (smoothed_x, smoothed_y)
    
    # 确保点在图像范围内
    if fingertip:
        h, w = frame_shape[:2]
        x, y = fingertip
        if 0 <= x < w and 0 <= y < h:
            return fingertip
    
    return None


def detect_and_track_fingertip(frame, background, prev_tip=None, drawing_activated=False, hand_already_confirmed=False):
    """改进的指尖检测和跟踪算法，实现两阶段检测：先检测手部，再检测指尖"""
    # 创建调试帧
    debug_frame = frame.copy()
    
    # 如果已经确认手部，且有上一个指尖位置，直接进入局部跟踪模式
    if hand_already_confirmed and prev_tip is not None:
        # 创建一个以上一个指尖位置为中心的局部区域掩码
        fingertip_region_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        # 如果正在绘制，使用较小的搜索半径，减少背景干扰
        search_radius = 40 if drawing_activated else 60
        cv2.circle(fingertip_region_mask, prev_tip, search_radius, 255, -1)
        
        # 在这个区域内检测肤色
        skin_mask = detect_skin(frame)
        local_skin_mask = cv2.bitwise_and(skin_mask, fingertip_region_mask)
        
        # 在调试帧上绘制搜索区域
        circle_color = (0, 255, 0) if drawing_activated else (0, 255, 255)
        cv2.circle(debug_frame, prev_tip, search_radius, circle_color, 2)
        
        # 在局部区域内寻找轮廓
        contours, _ = cv2.findContours(local_skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果在局部区域找到轮廓
        local_contour = None
        if contours:
            # 选择最大的轮廓
            local_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(local_contour) > 100:
                # 在调试帧上绘制局部轮廓
                cv2.drawContours(debug_frame, [local_contour], -1, (255, 0, 255), 2)
                
                # 寻找局部轮廓中的最高点作为指尖
                min_y = float('inf')
                fingertip = None
                
                for point in local_contour:
                    if point[0][1] < min_y:
                        min_y = point[0][1]
                        fingertip = (point[0][0], point[0][1])
                
                # 如果找到指尖，进行平滑处理
                if fingertip and prev_tip:
                    # 计算新旧位置之间的距离
                    dist = np.sqrt((fingertip[0] - prev_tip[0])**2 + (fingertip[1] - prev_tip[1])**2)
                    
                    # 在绘制模式下使用较大的平滑因子，提高稳定性
                    if drawing_activated:
                        # 绘制模式下，优先稳定性
                        alpha = 0.6  # 增大平滑系数，增加稳定性
                        smoothed_x = int(alpha * prev_tip[0] + (1 - alpha) * fingertip[0])
                        smoothed_y = int(alpha * prev_tip[1] + (1 - alpha) * fingertip[1])
                        fingertip = (smoothed_x, smoothed_y)
                    else:
                        # 非绘制模式下，只对大幅度跳变进行平滑
                        max_movement = 50
                        if dist > max_movement:
                            alpha = 0.3
                            smoothed_x = int(alpha * prev_tip[0] + (1 - alpha) * fingertip[0])
                            smoothed_y = int(alpha * prev_tip[1] + (1 - alpha) * fingertip[1])
                            fingertip = (smoothed_x, smoothed_y)
                
                # 在调试帧上标记指尖
                if fingertip:
                    cv2.circle(debug_frame, fingertip, 10, (0, 0, 255), -1)
                    cv2.circle(debug_frame, fingertip, 15, (0, 255, 0), 2)
                    
                    # 在调试画面上显示状态文字
                    mode_text = "绘制模式" if drawing_activated else "指尖锁定模式"
                    cv2.putText(debug_frame, mode_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    return fingertip, debug_frame, local_skin_mask
                    
    # 第一步：检测手部区域
    skin_mask = detect_skin(frame)
    motion_mask = detect_hand(frame, background)
    
    # 优先使用肤色检测，辅以运动检测
    hand_mask = skin_mask.copy()
    
    # 找到手部轮廓
    hand_contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 筛选最大的轮廓作为手部
    hand_contour = None
    if hand_contours:
        # 按面积排序轮廓
        hand_contours = sorted(hand_contours, key=cv2.contourArea, reverse=True)
        # 选择最大的轮廓，最好大于一定面积
        if hand_contours and cv2.contourArea(hand_contours[0]) > 2000:
            hand_contour = hand_contours[0]
    
    # 如果未找到手部，返回空
    if hand_contour is None:
        return None, debug_frame, hand_mask
    
    # 创建一个空白掩码，仅包含手部
    hand_only_mask = np.zeros_like(skin_mask)
    cv2.drawContours(hand_only_mask, [hand_contour], -1, 255, -1)
    
    # 在调试帧上绘制手部轮廓
    cv2.drawContours(debug_frame, [hand_contour], -1, (0, 255, 0), 2)
    
    # 如果是绘制模式且已经有了之前的指尖位置，使用局部搜索
    if drawing_activated and prev_tip is not None:
        # 创建一个以上一个指尖位置为中心的局部区域掩码
        fingertip_region_mask = np.zeros_like(skin_mask)
        search_radius = 50  # 搜索半径
        cv2.circle(fingertip_region_mask, prev_tip, search_radius, 255, -1)
        
        # 该区域内仅考虑手部区域
        local_hand_mask = cv2.bitwise_and(hand_only_mask, fingertip_region_mask)
        
        # 在调试帧上绘制搜索区域
        cv2.circle(debug_frame, prev_tip, search_radius, (0, 255, 255), 2)
        
        # 在局部区域内寻找轮廓
        local_contours, _ = cv2.findContours(local_hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果在局部区域找到轮廓
        local_contour = None
        if local_contours:
            local_contour = max(local_contours, key=cv2.contourArea)
            if cv2.contourArea(local_contour) > 100:
                # 在调试帧上绘制局部轮廓
                cv2.drawContours(debug_frame, [local_contour], -1, (255, 0, 255), 2)
                
                # 寻找局部轮廓中的最高点作为指尖
                min_y = float('inf')
                fingertip = None
                
                for point in local_contour:
                    if point[0][1] < min_y:
                        min_y = point[0][1]
                        fingertip = (point[0][0], point[0][1])
                
                # 如果找到指尖，应用平滑
                if fingertip and prev_tip:
                    # 计算新旧位置之间的距离
                    dist = np.sqrt((fingertip[0] - prev_tip[0])**2 + (fingertip[1] - prev_tip[1])**2)
                    
                    # 应用较小的平滑，减少延迟感
                    max_movement = 30
                    if dist > max_movement:
                        alpha = 0.5  # 降低平滑系数，减少延迟
                        smoothed_x = int(alpha * prev_tip[0] + (1 - alpha) * fingertip[0])
                        smoothed_y = int(alpha * prev_tip[1] + (1 - alpha) * fingertip[1])
                        fingertip = (smoothed_x, smoothed_y)
                
                # 在调试帧上标记指尖
                if fingertip:
                    cv2.circle(debug_frame, fingertip, 10, (0, 0, 255), -1)
                    
                    # 如果有前一个指尖位置，绘制轨迹
                    if prev_tip:
                        cv2.line(debug_frame, prev_tip, fingertip, (255, 0, 255), 2)
                
                # 在调试画面上显示使用的掩码类型
                cv2.putText(debug_frame, "指尖局部跟踪", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 在调试画面上显示掩码
                small_mask = cv2.resize(local_hand_mask, (160, 120))
                small_mask_color = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
                debug_frame[10:130, 10:170] = small_mask_color
                
                return fingertip, debug_frame, local_hand_mask
            
    # 如果不在绘制模式或者局部搜索失败，进行全局搜索
    
    # 第二步：在整个手部区域中寻找指尖点
    # 获取手部轮廓的凸包
    hull = cv2.convexHull(hand_contour)
    
    # 计算指尖候选点
    fingertip_candidates = []
    
    # 方法1：找最高点（y坐标最小）
    min_y = float('inf')
    top_point = None
    
    for point in hull:
        if point[0][1] < min_y:
            min_y = point[0][1]
            top_point = (point[0][0], point[0][1])
    
    if top_point:
        fingertip_candidates.append(top_point)
    
    # 方法2: 使用凸缺陷找出可能的指尖
    try:
        defects = cv2.convexityDefects(hand_contour, cv2.convexHull(hand_contour, returnPoints=False))
        
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(hand_contour[s][0])
                end = tuple(hand_contour[e][0])
                far = tuple(hand_contour[f][0])
                
                # 计算角度
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                
                try:
                    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                except:
                    continue
                
                # 如果角度小于90度，可能是指尖
                if angle <= np.pi / 2:  # 90度
                    # 优先选择图像上部的点
                    if start[1] < frame.shape[0] * 0.6:  # 只选择在图像上部60%的点
                        fingertip_candidates.append(start)
                    if end[1] < frame.shape[0] * 0.6:
                        fingertip_candidates.append(end)
    except:
        pass
    
    # 如果没有找到候选点，使用最高点
    if not fingertip_candidates and top_point:
        fingertip_candidates = [top_point]
    
    # 从候选点中选择最可能的指尖（优先使用最高的点）
    fingertip = None
    min_y = float('inf')
    
    for point in fingertip_candidates:
        if point[1] < min_y:
            min_y = point[1]
            fingertip = point
    
    # 如果找到指尖，进行平滑处理
    if fingertip and prev_tip:
        # 计算距离
        dist = np.sqrt((fingertip[0] - prev_tip[0])**2 + (fingertip[1] - prev_tip[1])**2)
        
        # 如果距离太大，应用平滑
        max_movement = 40
        if dist > max_movement:
            # 使用较小的平滑系数，减少延迟
            alpha = 0.5
            smoothed_x = int(alpha * prev_tip[0] + (1 - alpha) * fingertip[0])
            smoothed_y = int(alpha * prev_tip[1] + (1 - alpha) * fingertip[1])
            fingertip = (smoothed_x, smoothed_y)
    
    # 如果找到指尖，在调试图上标记
    if fingertip:
        # 当手部已确认，显示绿色圆圈，否则显示黄色圆圈
        circle_color = (0, 255, 0) if hand_already_confirmed else (0, 255, 255)
        cv2.circle(debug_frame, fingertip, 10, (0, 0, 255), -1)  # 红色实心圆表示指尖
        cv2.circle(debug_frame, fingertip, 15, circle_color, 2)  # 彩色空心圆表示状态
        
        # 如果有前一个指尖位置，绘制轨迹
        if prev_tip:
            cv2.line(debug_frame, prev_tip, fingertip, (255, 0, 255), 2)
    
    # 在调试画面上显示使用的掩码
    small_mask = cv2.resize(hand_only_mask, (160, 120))
    small_mask_color = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
    debug_frame[10:130, 10:170] = small_mask_color
    
    # 在调试画面上显示状态文字
    status = "手部已锁定" if hand_already_confirmed else "等待确认"
    cv2.putText(debug_frame, f"状态: {status}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return fingertip, debug_frame, hand_only_mask


def main():
    # 尝试不同的摄像头索引
    camera_indices = [0, 1, 2]  # 尝试前三个摄像头索引
    cap = None
    
    for index in camera_indices:
        print(f"尝试打开摄像头索引 {index}...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"成功打开摄像头索引 {index}")
            break
        else:
            print(f"无法打开摄像头索引 {index}")
    
    # 验证摄像头是否成功打开
    if not cap or not cap.isOpened():
        print("错误：无法打开任何摄像头，请检查摄像头连接或权限设置")
        print("请确保已授予应用程序访问摄像头的权限")
        return

    # 设置摄像头分辨率为固定值
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 打印摄像头信息
    print(f"摄像头成功打开！")
    print(f"摄像头分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # 创建可调整大小的窗口
    cv2.namedWindow('MediaPipe', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Hand Detection', cv2.WINDOW_NORMAL)

    # 设置初始窗口大小
    cv2.resizeWindow('MediaPipe', 640, 480)
    cv2.resizeWindow('Canvas', 640, 480)
    cv2.resizeWindow('Hand Detection', 640, 480)

    # 创建空白画布
    canvas = np.ones((480, 640), np.uint8) * 255

    # 加载模型
    try:
        print("正在加载字符识别模型...")
        model = Cnn()
        model.load_state_dict(torch.load('model_emnist.pt', map_location='cpu'))
        print("模型加载成功！")
    except Exception as e:
        print(f"错误：无法加载模型 - {e}")
        print("请确保model_emnist.pt文件存在于当前目录")
        cap.release()
        cv2.destroyAllWindows()
        return

    # 获取背景模型
    try:
        print("开始获取背景模型，请确保摄像头视野内没有移动物体...")
        background = get_background(cap)
        if background is None:
            print("获取背景模型失败，程序退出")
            cap.release()
            cv2.destroyAllWindows()
            return
    except Exception as e:
        print(f"获取背景模型出错: {e}")
        import traceback
        traceback.print_exc()
        cap.release()
        cv2.destroyAllWindows()
        return

    # 状态变量
    is_drawing = False
    made_prediction = False
    prediction_result = ""
    last_fingertip_pos = None
    current_stroke = []  # 当前正在绘制的笔画
    all_stroke_points = []  # 存储所有绘制点的列表，每个元素是一组点
    
    # 手部检测状态
    hand_detected = False     # 是否检测到手部
    hand_confirmed = False    # 用户是否已确认手部检测正确
    fingertip_locked = False  # 是否锁定了指尖
    
    # 指尖跟踪历史和绘制平滑
    fingertip_history = []  # 最近几帧的指尖位置
    history_max_size = 10   # 历史记录的最大长度
    missing_frames = 0      # 连续丢失指尖跟踪的帧数
    max_missing_frames = 15 # 最大允许的丢失帧数，超过此值则取消绘制状态
    drawing_points = []     # 用于平滑绘制的点列表
    drawing_points_max = 3  # 平滑绘制时使用的点数
    
    # 显示操作指南
    print("\n=== 基于传统计算机视觉的空中写字识别系统 ===")
    print("操作指南：")
    print("按 'h' - 确认当前检测到的手部/指尖")
    print("按 'd' - 开始绘制 (必须先确认手部检测)")
    print("按 'e' - 结束绘制并准备预测")
    print("按 'p' - 预测绘制的字符")
    print("按 'c' - 清除画布")
    print("按 'q' - 退出程序")
    print("===============================================\n")

    # 主循环
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("警告：无法获取摄像头画面")
                break
            
            # 调整帧大小为固定的640x480
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)  # 水平翻转图像
            display_image = frame.copy()

            # 处理键盘输入
            key = cv2.waitKey(10) & 0xFF

            # 确认当前检测到的手部/指尖
            if key == ord('h'):
                if hand_detected:
                    hand_confirmed = True
                    fingertip_locked = True
                    print("已确认手部和指尖位置！按'd'开始绘制")
                else:
                    print("未检测到手部/指尖，无法确认")

            # 开始绘制 (仅当手部和指尖已确认后)
            elif key == ord('d'):
                if hand_confirmed:
                    is_drawing = True
                    current_stroke = []  # 开始新的笔画
                    print("绘制模式已启动！")
                else:
                    print("请先确认手部检测 (按'h'键)，然后再开始绘制")

            # 结束绘制
            elif key == ord('e'):
                if is_drawing:
                    is_drawing = False
                    if current_stroke:
                        all_stroke_points.append(current_stroke.copy())
                        current_stroke = []
                    print("绘制已结束，按'p'进行预测或'd'继续绘制")

            # 清除画布
            elif key == ord('c'):
                canvas = np.ones((480, 640), np.uint8) * 255
                all_stroke_points.clear()
                current_stroke = []
                made_prediction = False
                print("画布已清除！")

            # 预测字符
            elif key == ord('p'):
                if len(all_stroke_points) > 0:
                    try:
                        print("\n----- 开始预测 -----")
                        roi = get_ROI(canvas)
                        cv2.imwrite("debug_roi.jpg", roi)
                        print("已保存ROI图像到debug_roi.jpg")

                        prediction = character_prediction(roi, model)
                        prediction_result = chr(prediction + 65)  # A-Z对应65-90的ASCII码
                        print(f"预测结果: {prediction_result}")
                        made_prediction = True

                        cv2.imwrite(f"{prediction_result}.jpg", roi)
                        print("----- 预测完成 -----\n")
                    except Exception as e:
                        print(f"预测错误: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("请先绘制字符再进行预测！")

            # 退出
            elif key == ord('q'):
                break

            # 指尖检测和跟踪
            try:
                # 如果手部已确认，只需要跟踪指尖
                if hand_confirmed and last_fingertip_pos:
                    fingertip, debug_frame, hand_mask = detect_and_track_fingertip(
                        frame, background, last_fingertip_pos,
                        drawing_activated=is_drawing,
                        hand_already_confirmed=True
                    )
                    if fingertip:
                        current_pos = fingertip
                        missing_frames = 0
                    else:
                        # 指尖检测丢失
                        missing_frames += 1
                        if missing_frames >= max_missing_frames:
                            if is_drawing:
                                print("丢失指尖跟踪，绘制已暂停")
                                is_drawing = False
                            # 不重置hand_confirmed状态，只是暂停绘制
                        current_pos = None
                # 如果手部未确认，需要检测手部和指尖
                else:
                    fingertip, debug_frame, hand_mask = detect_and_track_fingertip(
                        frame, background, last_fingertip_pos,
                        drawing_activated=False,
                        hand_already_confirmed=False
                    )
                    
                    if fingertip:
                        # 重置丢失帧计数
                        missing_frames = 0
                        hand_detected = True
                        
                        # 添加到历史记录
                        fingertip_history.append(fingertip)
                        if len(fingertip_history) > history_max_size:
                            fingertip_history.pop(0)
                        
                        # 在确认前使用历史平均位置
                        if fingertip_history:
                            avg_x = int(sum(p[0] for p in fingertip_history) / len(fingertip_history))
                            avg_y = int(sum(p[1] for p in fingertip_history) / len(fingertip_history))
                            current_pos = (avg_x, avg_y)
                        else:
                            current_pos = fingertip
                    else:
                        # 指尖检测丢失
                        missing_frames += 1
                        if missing_frames >= max_missing_frames:
                            hand_detected = False
                        current_pos = None
                
            except Exception as e:
                print(f"指尖检测出错: {e}")
                current_pos = None
                debug_frame = frame.copy()
                cv2.putText(debug_frame, "检测错误", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 在显示图像上添加状态信息
            status_text = []
            if not hand_confirmed:
                if hand_detected:
                    status_text.append("【按 'h' 确认检测到的手部】")
                else:
                    status_text.append("【正在搜索手部...】")
            elif not is_drawing:
                status_text.append("【按 'd' 开始绘制】")
            else:
                status_text.append("【正在绘制中...】")

            # 使用支持中文的文本绘制函数
            y_pos = 30
            for text in status_text:
                pos = (display_image.shape[1] - 300, y_pos)
                display_image = cv2_put_chinese_text(display_image, text, pos, (0, 0, 255), 24)
                y_pos += 30

            # 绘制功能
            if is_drawing and current_pos and hand_confirmed:
                # 添加到当前笔画
                current_stroke.append(current_pos)
                
                # 加入绘制点队列进行平滑
                drawing_points.append(current_pos)
                if len(drawing_points) > drawing_points_max:
                    drawing_points.pop(0)
                
                # 平滑绘制点计算 - 增加平滑强度
                draw_x = 0
                draw_y = 0
                
                # 使用加权平均，最近的点权重最大
                weights = [0.15, 0.25, 0.6]  # 更偏向当前点，但保持较高平滑度
                weight_sum = sum(weights[:len(drawing_points)])
                
                for i, point in enumerate(drawing_points):
                    w = weights[i] if i < len(weights) else weights[-1]
                    draw_x += point[0] * w
                    draw_y += point[1] * w
                
                if weight_sum > 0:
                    draw_x = int(draw_x / weight_sum)
                    draw_y = int(draw_y / weight_sum)
                    smoothed_draw_point = (draw_x, draw_y)
                else:
                    smoothed_draw_point = current_pos
                
                # 如果有上一个位置，则绘制线段
                if last_fingertip_pos is not None:
                    # 检查两点间距离，如果太大则不绘制（避免跳变）
                    dist = np.sqrt((smoothed_draw_point[0] - last_fingertip_pos[0])**2 + 
                                   (smoothed_draw_point[1] - last_fingertip_pos[1])**2)
                    
                    if dist < 40:  # 降低距离阈值，更严格的跳变检测
                        # 在原始图像上绘制轨迹
                        cv2.line(display_image, last_fingertip_pos, smoothed_draw_point, (255, 0, 255), 4)
                        # 在画布上绘制轨迹
                        cv2.line(canvas, last_fingertip_pos, smoothed_draw_point, (0, 0, 0), 8)
                else:
                    # 如果是第一个点，绘制一个点
                    cv2.circle(canvas, smoothed_draw_point, 4, (0, 0, 0), -1)

                last_fingertip_pos = smoothed_draw_point
            
            # 如果检测到手部但未确认，显示指示器
            elif current_pos and not hand_confirmed:
                # 显示黄色圆圈，表示待确认
                cv2.circle(display_image, current_pos, 15, (0, 255, 255), 2)
                cv2.putText(display_image, "按'h'确认", (current_pos[0] - 40, current_pos[1] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 如果手部已确认但未开始绘制，显示绿色指示器
            elif current_pos and hand_confirmed and not is_drawing:
                # 显示绿色圆圈，表示已确认
                cv2.circle(display_image, current_pos, 15, (0, 255, 0), 2)
                cv2.putText(display_image, "按'd'绘制", (current_pos[0] - 40, current_pos[1] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 重绘所有笔画
            for stroke in all_stroke_points:
                for i in range(1, len(stroke)):
                    cv2.line(display_image, stroke[i - 1], stroke[i], (255, 0, 255), 4)

            # 重绘当前笔画
            for i in range(1, len(current_stroke)):
                cv2.line(display_image, current_stroke[i - 1], current_stroke[i], (255, 0, 255), 4)

            # 创建结果显示区域
            result_panel = np.ones((200, display_image.shape[1], 3), dtype=np.uint8) * 240

            # 如果已经进行预测，显示结果
            if made_prediction:
                # 在结果面板上使用更大更醒目的字体显示预测结果
                result_text = f"【预测结果: {prediction_result}】"
                result_panel = cv2_put_chinese_text(
                    result_panel,
                    result_text,
                    (20, 100),
                    (0, 0, 255),
                    36
                )
                
                # 添加操作提示
                hint_text = "【按 'c' 清除画布重新绘制】"
                result_panel = cv2_put_chinese_text(
                    result_panel,
                    hint_text,
                    (20, 150),
                    (0, 0, 0),
                    24
                )
            else:
                # 显示当前状态
                status = status_text[0] if status_text else "【等待操作...】"
                result_panel = cv2_put_chinese_text(
                    result_panel,
                    status,
                    (20, 100),
                    (0, 0, 0),
                    36
                )

            # 为画布添加边框
            canvas_display = canvas.copy()
            cv2.rectangle(canvas_display, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), (0, 0, 0), 2)

            # 将结果面板与显示图像垂直拼接
            combined_image = np.vstack((display_image, result_panel))

            # 当有效跟踪时转换为彩色以便显示文字
            canvas_display_color = cv2.cvtColor(canvas_display, cv2.COLOR_GRAY2BGR)

            # 在画布上显示状态提示
            status_color = (0, 255, 0) if hand_confirmed and is_drawing else (0, 165, 255)
            canvas_status = status_text[0] if status_text else "【等待操作...】"
            canvas_display_color = cv2_put_chinese_text(
                canvas_display_color,
                canvas_status,
                (20, 30),
                status_color,
                30
            )

            # 显示画面和画布
            cv2.imshow('MediaPipe', combined_image)
            cv2.imshow('Canvas', canvas_display_color)
            cv2.imshow('Hand Detection', debug_frame)
            
    except Exception as e:
        print(f"运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")


if __name__ == '__main__':
    main() 