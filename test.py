import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import os


# 用PIL绘制中文文本到OpenCV图像
def cv2_put_chinese_text(img, text, position, textColor=(0, 255, 0), textSize=30):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 创建一个可以在给定图像上绘制的对象
    draw = ImageDraw.Draw(img)

    # 字体文件路径
    fontStyle = None
    # 尝试加载系统中文字体
    possible_fonts = [
        os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'Fonts', 'simhei.ttf'),  # 黑体
        os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'Fonts', 'simsun.ttc'),  # 宋体
        os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'Fonts', 'msyh.ttc'),  # 微软雅黑
    ]

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


# 手部检测和手势识别类
class HandDetector:
    def __init__(self):
        # 改进背景减除器参数，降低varThreshold使更容易捕获手部
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=20, detectShadows=False)
        
        # 使用更大的内核进行形态学操作，以更好地连接手部区域
        self.kernel = np.ones((7, 7), np.uint8)
        self.close_kernel = np.ones((9, 9), np.uint8)
        
        self.history_points = []  # 存储历史轨迹点
        self.finger_tip = None  # 当前手指尖点
        self.palm_center = None  # 手掌中心
        self.fingers = []  # 检测到的手指点
        self.is_calibrated = False  # 背景校准标志
        self.calibration_frames = 0  # 背景校准帧计数
        self.min_calibration_frames = 50  # 增加校准帧数，使背景模型更稳定
        self.tracking_state = False  # 追踪状态
        
        # 添加指尖位置平滑处理
        self.finger_history = []  # 存储最近几帧的指尖位置
        self.max_history = 5  # 历史帧数
        
        # 添加手部检测状态历史，避免单帧检测失败导致状态快速变化
        self.detection_history = [False] * 5  # 最近5帧的检测状态
        
        # 添加最大轮廓面积缓存，用于轮廓筛选
        self.last_contour_area = 0

    def calibrate_background(self, frame):
        """校准背景"""
        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        self.bg_subtractor.apply(blurred)
        self.calibration_frames += 1
        if self.calibration_frames >= self.min_calibration_frames:
            self.is_calibrated = True
            print("背景校准完成！")
        return self.is_calibrated

    def detect_hand(self, frame):
        """检测手部区域"""
        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # 应用背景减除器
        fg_mask = self.bg_subtractor.apply(blurred, learningRate=0.01)  # 降低学习率，使背景模型更稳定
        
        # 更强的噪声处理
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.close_kernel)
        
        # 阈值处理，使用自适应阈值
        _, thresh = cv2.threshold(fg_mask, 100, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 保存原始帧，用于绘制
        result_frame = frame.copy()
        
        # 调试时显示阈值图像
        # cv2.imshow("Thresh", thresh)
        
        # 如果没有检测到轮廓，保持历史指尖位置短时间不变
        if not contours:
            # 更新检测历史
            self.detection_history.pop(0)
            self.detection_history.append(False)
            
            # 只有连续多帧未检测到才清空指尖位置
            if not any(self.detection_history):
                self.finger_tip = None
                self.palm_center = None
                self.fingers = []
            
            return result_frame, any(self.detection_history)
        
        # 找到最大的轮廓（假设是手部）
        max_contour = max(contours, key=cv2.contourArea)
        current_area = cv2.contourArea(max_contour)
        
        # 面积太小的轮廓不考虑，但记录面积突变
        min_area = 2000  # 降低最小面积阈值以增加敏感度
        
        # 使用面积比例变化而不是绝对值，防止因距离变化导致的误判
        area_ratio = 1.0
        if self.last_contour_area > 0:
            area_ratio = current_area / self.last_contour_area
            
        # 更新面积历史
        self.last_contour_area = current_area
        
        # 面积太小或变化太剧烈，可能不是手
        if current_area < min_area or (area_ratio > 3.0 or area_ratio < 0.3):
            # 更新检测历史
            self.detection_history.pop(0)
            self.detection_history.append(False)
            
            if not any(self.detection_history):
                self.finger_tip = None
                self.palm_center = None
                self.fingers = []
                
            return result_frame, any(self.detection_history)
        
        # 更新检测历史为成功
        self.detection_history.pop(0)
        self.detection_history.append(True)
        
        # 绘制轮廓
        cv2.drawContours(result_frame, [max_contour], 0, (0, 255, 0), 2)
        
        # 计算凸包
        hull = cv2.convexHull(max_contour, returnPoints=False)
        
        # 凸缺陷
        try:
            defects = cv2.convexityDefects(max_contour, hull)
            # 找到手指尖点
            self.find_fingertips(max_contour, defects, result_frame)
        except Exception as e:
            print(f"凸缺陷计算错误: {e}")
            defects = None
            
        # 标记食指尖（假设是第一个检测到的手指尖）
        if self.fingers:
            # 更新指尖历史
            self.finger_history.append(self.fingers[0])
            if len(self.finger_history) > self.max_history:
                self.finger_history.pop(0)
                
            # 计算平滑后的指尖位置（使用最近几帧的平均位置）
            if len(self.finger_history) > 0:
                x_sum = sum(p[0] for p in self.finger_history)
                y_sum = sum(p[1] for p in self.finger_history)
                smooth_x = x_sum // len(self.finger_history)
                smooth_y = y_sum // len(self.finger_history)
                
                self.finger_tip = (smooth_x, smooth_y)
                cv2.circle(result_frame, self.finger_tip, 10, (0, 255, 0), -1)
                
                # 在指尖处绘制一个较大的圆以显示跟踪状态
                cv2.circle(result_frame, self.finger_tip, 20, (255, 255, 0), 2)
        
        return result_frame, True

    def find_fingertips(self, contour, defects, frame):
        """查找手指尖点"""
        # 重置手指列表
        self.fingers = []
        
        # 计算轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 绘制外接矩形
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        # 计算手掌中心（使用轮廓的矩单作为近似）
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            self.palm_center = (cx, cy)
            cv2.circle(frame, self.palm_center, 5, (0, 0, 255), -1)
            
            # 绘制较大的圆表示手掌中心
            cv2.circle(frame, self.palm_center, 20, (0, 0, 255), 2)
        
        # 如果没有凸缺陷，使用轮廓最顶部的点作为指尖
        if defects is None:
            # 找到轮廓最上方的点
            top_point = min(contour, key=lambda p: p[0][1])
            if self.palm_center and top_point[0][1] < self.palm_center[1]:
                self.fingers.append((top_point[0][0], top_point[0][1]))
            return
        
        # 基于凸缺陷寻找手指尖点
        finger_candidates = []
        
        # 识别凸缺陷点
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # 计算三角形各边长度
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            
            # 避免除零错误
            if b * c == 0:
                continue
                
            # 余弦定理计算角度
            try:
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / np.pi
            except:
                continue
            
            # 小于80度的角可能是手指之间的角度（降低阈值使检测更准确）
            if angle <= 80:
                # 绘制凸缺陷点
                cv2.circle(frame, far, 5, [128, 0, 255], -1)
                
                # 手指候选点
                finger_candidates.extend([start, end])
                
                # 在指间凹陷处绘制线段，帮助可视化
                cv2.line(frame, start, far, (255, 0, 0), 1)
                cv2.line(frame, end, far, (255, 0, 0), 1)
        
        # 过滤候选点，选择靠近轮廓顶部的点
        if finger_candidates:
            # 按y坐标排序（越小越靠上）
            finger_candidates.sort(key=lambda p: p[1])
            
            # 取最靠上的点作为食指尖（最可能的绘制点）
            if self.palm_center:
                # 过滤：只选择在手掌中心上方的点
                valid_fingers = [p for p in finger_candidates if p[1] < self.palm_center[1] - 10]
                
                # 按与轮廓中心的距离排序
                if valid_fingers:
                    valid_fingers.sort(
                        key=lambda p: (p[0] - self.palm_center[0]) ** 2 + (p[1] - self.palm_center[1]) ** 2,
                        reverse=True)
                    
                    # 选择1-3个点作为可能的手指
                    self.fingers = valid_fingers[:min(3, len(valid_fingers))]
                    
                    # 如果没有有效的手指，使用轮廓最上方的点
                    if not self.fingers:
                        top_point = min(contour, key=lambda p: p[0][1])
                        self.fingers.append((top_point[0][0], top_point[0][1]))
                else:
                    # 如果没有手指在手掌上方，使用轮廓最上方的点
                    top_point = min(contour, key=lambda p: p[0][1])
                    self.fingers.append((top_point[0][0], top_point[0][1]))

    def detect_victory_gesture(self, frame):
        """检测耶手势"""
        if not self.fingers or not self.palm_center:
            return False
            
        # 如果检测到至少2个手指，且手指间距离足够
        if len(self.fingers) >= 2:
            # 计算前两个手指之间的距离
            dist = np.sqrt((self.fingers[0][0] - self.fingers[1][0]) ** 2 +
                           (self.fingers[0][1] - self.fingers[1][1]) ** 2)
                           
            # 如果手指间距离足够大，可能是耶手势
            if dist > 40:  # 降低阈值使检测更敏感
                # 在图像上标记检测到的耶手势
                cv2.line(frame, self.fingers[0], self.fingers[1], (0, 255, 255), 2)
                cv2.putText(frame, "Victory", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                return True
                
        return False

    def reset_tracking(self):
        """重置追踪状态"""
        self.finger_tip = None
        self.palm_center = None
        self.fingers = []
        self.finger_history = []
        self.tracking_state = False
        self.is_calibrated = False
        self.calibration_frames = 0
        self.detection_history = [False] * 5
        self.last_contour_area = 0


def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)

    # 验证摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头，请检查摄像头连接或尝试其他摄像头索引")
        print("尝试其他摄像头索引，例如：cv2.VideoCapture(1)")
        return

    # 打印摄像头信息
    print(f"摄像头成功打开！")
    print(f"摄像头分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # 创建空白画布
    canvas = np.ones((480, 640), np.uint8) * 255  # 调整画布大小与摄像头匹配

    # 初始化手部检测器
    hand_detector = HandDetector()

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

    # 状态变量
    is_tracking = False  # 是否正在追踪手部
    is_drawing = False  # 是否正在绘制
    made_prediction = False  # 是否已经做出预测
    ready_to_predict = False  # 是否准备好进行预测（握拳后）
    prediction = 0  # 预测结果
    prediction_result = ""  # 预测结果字符串
    last_finger_pos = None  # 上一个指尖位置
    
    # 新增：检测冻结状态
    is_detection_frozen = False  # 是否冻结了检测
    frozen_finger_tip = None  # 冻结时的指尖位置
    
    # 存储手指指尖位置或鼠标位置
    finger_points = []
    last_mouse_pos = None  # 用于鼠标控制方案

    # 添加新状态变量用于防止误触发
    victory_detection_counter = 0  # 耶手势检测计数器
    victory_detection_threshold = 3  # 需要连续检测到耶手势的次数

    # 添加轨迹绘制相关变量
    all_stroke_points = []  # 存储所有绘制点的列表，每个元素是一组点
    current_stroke = []  # 当前正在绘制的笔画

    # 鼠标回调函数（如果手势检测不可用）
    def mouse_callback(event, x, y, flags, param):
        nonlocal last_mouse_pos
        if event == cv2.EVENT_MOUSEMOVE and is_drawing:
            current_pos = (x, y)
            if last_mouse_pos:
                # 在画布上绘制
                cv2.line(canvas, last_mouse_pos, current_pos, (0, 0, 0), 4)
                # 存储点
                finger_points.append(current_pos)
                # 限制点的数量
                if len(finger_points) > 1000:
                    finger_points.pop(0)
            last_mouse_pos = current_pos

    # 设置鼠标回调
    cv2.namedWindow('画布')
    cv2.setMouseCallback('画布', mouse_callback)

    # 显示帮助信息
    print("\n=== 空中写字识别系统 (OpenCV版) ===")
    print("操作指南：")
    print("按 's' - 开始手部追踪和背景校准")
    print("按 'd' - 开始绘制")
    print("按 'f' - 冻结/解冻当前检测（当找到好的指尖位置时非常有用）")
    print("做出耶手势 - 结束绘制并准备预测")
    print("按 'p' - 预测绘制的字符")
    print("按 'c' - 清除画布")
    print("按 'q' - 退出程序")
    print("另外：也可以使用鼠标直接在画布上绘制")
    print("==============================\n")

    frame_count = 0  # 帧计数器，用于调试
    
    # 新增：用于显示冻结状态的标志
    freeze_notice_time = 0  # 记录冻结通知的时间

    # 主循环
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("警告：无法获取摄像头画面")
            # 尝试再次读取
            continue

        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧打印一次状态
            print(f"程序运行中... 帧大小: {image.shape}")

        # 水平翻转图像，使其更自然
        image = cv2.flip(image, 1)

        # 创建原始图像的副本，用于显示
        display_image = image.copy()

        # 在图像右上角显示状态信息
        status_text = []
        if is_tracking:
            status_text.append("追踪: 开启")
            if is_detection_frozen:
                status_text.append("检测: 已冻结")
            else:
                status_text.append("检测: 实时")
        else:
            status_text.append("追踪: 关闭")

        if is_drawing:
            status_text.append("绘制: 开启")
        else:
            status_text.append("绘制: 关闭")

        # 使用支持中文的文本绘制函数
        y_pos = 30
        for i, text in enumerate(status_text):
            pos = (display_image.shape[1] - 200, y_pos)
            display_image = cv2_put_chinese_text(display_image, text, pos, (0, 0, 255), 24)
            y_pos += 30
            
        # 处理键盘输入 - 增加等待时间，确保按键能被检测到
        key = cv2.waitKey(10) & 0xFF

        # 记录按键用于调试
        if key != 255:  # 255表示没有按键
            print(f"检测到按键: {chr(key) if key >= 32 and key <= 126 else key}")

        # 开始追踪
        if key == ord('s'):
            is_tracking = True
            is_detection_frozen = False  # 开始追踪时解冻检测
            hand_detector.reset_tracking()
            print("手部追踪已启动！请在校准期间将手移出摄像头视野...")
            
            # 在画面上显示校准指导
            cv2.rectangle(display_image, (50, 120), (display_image.shape[1]-50, 360), (255, 255, 255), -1)
            display_image = cv2_put_chinese_text(
                display_image,
                "背景校准中，请将手移出摄像头",
                (100, 200),
                (0, 0, 255),
                40
            )
            display_image = cv2_put_chinese_text(
                display_image,
                "校准完成后再将手放入画面",
                (100, 260),
                (0, 0, 255),
                40
            )
            cv2.imshow('OpenCV空中写字', display_image)
            cv2.waitKey(100)  # 稍微暂停以确保用户看到指示
            
            # 添加一个小延迟，给用户时间移出手部
            for i in range(3, 0, -1):
                temp_display = display_image.copy()
                cv2.rectangle(temp_display, (display_image.shape[1]//2-100, 300), 
                             (display_image.shape[1]//2+100, 350), (255, 255, 255), -1)
                temp_display = cv2_put_chinese_text(
                    temp_display,
                    f"{i}秒后开始校准",
                    (display_image.shape[1]//2-80, 330),
                    (0, 0, 255),
                    30
                )
                cv2.imshow('OpenCV空中写字', temp_display)
                cv2.waitKey(1000)  # 等待1秒
            
            # 重置背景减除器，确保完全从零开始
            hand_detector.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=20, detectShadows=False)
            
            # 在主循环中执行校准

        # 冻结/解冻检测
        elif key == ord('f'):
            if is_tracking and hand_detector.is_calibrated:
                is_detection_frozen = not is_detection_frozen
                if is_detection_frozen:
                    # 冻结当前指尖位置
                    frozen_finger_tip = hand_detector.finger_tip
                    print("检测已冻结")
                    freeze_notice_time = frame_count  # 记录冻结时间
                else:
                    print("检测已解冻")
                    freeze_notice_time = frame_count  # 记录解冻时间
                        
        # 开始绘制
        elif key == ord('d'):
            if not hand_detector.is_calibrated:
                print("请先等待背景校准完成...")
                display_image = cv2_put_chinese_text(
                    display_image,
                    "请先等待背景校准完成",
                    (100, 200),
                    (0, 0, 255),
                    40
                )
            else:
                is_drawing = True
                last_mouse_pos = None  # 重置鼠标位置
                last_finger_pos = None  # 重置指尖位置
                current_stroke = []  # 开始新的笔画
                print("绘制模式已启动！状态：", is_drawing)
                
                # 显示绘制指导
                cv2.rectangle(display_image, (50, 120), (display_image.shape[1]-50, 200), (255, 255, 255), -1)
                display_image = cv2_put_chinese_text(
                    display_image,
                    "请保持手部稳定，尽量只移动食指",
                    (100, 170),
                    (0, 0, 255),
                    30
                )
                cv2.imshow('OpenCV空中写字', display_image)
                cv2.waitKey(1500)  # 显示1.5秒让用户看到

        # 清除画布
        elif key == ord('c'):
            canvas = np.ones((480, 640), np.uint8) * 255
            finger_points.clear()
            all_stroke_points.clear()  # 清除所有笔画
            current_stroke = []  # 清除当前笔画
            made_prediction = False
            ready_to_predict = False  # 重置预测准备状态
            victory_detection_counter = 0  # 重置耶手势检测计数器
            last_mouse_pos = None
            print("画布已清除！")

        # 预测字符
        elif key == ord('p'):
            if len(finger_points) > 0:
                is_drawing = False
                try:
                    print("\n----- 开始预测 -----")
                    roi = get_ROI(canvas)

                    # 保存ROI图像用于调试
                    cv2.imwrite("debug_roi.jpg", roi)
                    print("已保存ROI图像到debug_roi.jpg")

                    prediction = character_prediction(roi, model)
                    prediction_result = chr(prediction + 65)
                    print(f"预测结果: {prediction_result}")
                    made_prediction = True
                    ready_to_predict = False  # 重置预测准备状态

                    # 保存ROI图
                    cv2.imwrite(f"{prediction_result}.jpg", roi)
                    print("----- 预测完成 -----\n")
                except Exception as e:
                    print(f"预测错误: {e}")
                    import traceback
                    traceback.print_exc()  # 打印完整错误信息
            else:
                print("请先绘制字符再进行预测！")

        # 退出
        elif key == ord('q'):
            break

        # 如果追踪模式开启，进行手部检测
        if is_tracking:
            # 如果未校准背景，先进行校准
            if not hand_detector.is_calibrated:
                # 显示校准状态
                cv2.putText(display_image,
                            f"Background calibration: {hand_detector.calibration_frames}/{hand_detector.min_calibration_frames}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 校准背景
                calibrated = hand_detector.calibrate_background(image)

                if calibrated:
                    print("背景校准完成，现在可以检测手部")
            else:
                # 如果检测未冻结，进行手部检测
                if not is_detection_frozen:
                    # 进行手部检测
                    display_image, detected = hand_detector.detect_hand(display_image)
                    
                    # 如果检测成功且未冻结，更新冻结位置（为了后续如果冻结能用）
                    if detected and hand_detector.finger_tip:
                        frozen_finger_tip = hand_detector.finger_tip
                else:
                    # 显示冻结状态通知
                    if frame_count - freeze_notice_time < 30:  # 显示3秒（假设10FPS）
                        cv2.rectangle(display_image, (50, 50), (400, 100), (255, 255, 255), -1)
                        display_image = cv2_put_chinese_text(
                            display_image,
                            "检测已冻结，按'f'解冻",
                            (60, 80),
                            (0, 0, 255),
                            30
                        )
                    
                    # 标记冻结的指尖位置
                    if frozen_finger_tip:
                        cv2.circle(display_image, frozen_finger_tip, 15, (0, 255, 255), -1)
                        cv2.circle(display_image, frozen_finger_tip, 25, (0, 255, 255), 2)
                    
                    # 在冻结状态下，视为检测成功
                    detected = frozen_finger_tip is not None
                    
                    # 设置hand_detector的finger_tip为冻结值，以便后续绘制
                    if detected:
                        hand_detector.finger_tip = frozen_finger_tip

                # 后续处理检测结果...
                if detected:
                    # 检测耶手势 (如果未冻结)
                    if is_drawing and not is_detection_frozen and hand_detector.detect_victory_gesture(display_image):
                        victory_detection_counter += 1
                        print(f"检测到可能的耶手势 ({victory_detection_counter}/{victory_detection_threshold})")

                        # 只有连续多次检测到耶手势才认为是真正的结束手势
                        if victory_detection_counter >= victory_detection_threshold:
                            if len(finger_points) > 5:  # 确保有足够的绘制点
                                is_drawing = False
                                ready_to_predict = True
                                victory_detection_counter = 0  # 重置计数器

                                # 如果当前笔画有点，则保存到所有笔画中
                                if current_stroke:
                                    all_stroke_points.append(current_stroke.copy())
                                    current_stroke = []

                                print("确认耶手势，绘制结束，准备预测")

                                # 在主窗口上显示明显的"绘制结束"提示
                                cv2.rectangle(display_image, (10, 60), (400, 160), (255, 255, 255), -1)
                                display_image = cv2_put_chinese_text(
                                    display_image,
                                    "绘制结束！按p键预测",
                                    (20, 120),
                                    (0, 0, 255),
                                    40
                                )
                    else:
                        # 如果不是耶手势，重置计数器
                        victory_detection_counter = 0

                    # 如果正在绘制且检测到指尖
                    if is_drawing and hand_detector.finger_tip:
                        current_pos = hand_detector.finger_tip

                        # 添加调试输出
                        if frame_count % 30 == 0:
                            print(f"正在绘制，当前指尖位置: {current_pos}")

                        # 确保current_pos在画布范围内
                        if 0 <= current_pos[0] < canvas.shape[1] and 0 <= current_pos[1] < canvas.shape[0]:
                            # 增加过滤条件：仅当位置变化显著且合理时才绘制
                            is_valid_move = True
                            
                            # 如果有上一个位置，检查移动距离是否合理
                            if last_finger_pos:
                                # 计算移动距离
                                distance = np.sqrt((current_pos[0] - last_finger_pos[0])**2 + 
                                                   (current_pos[1] - last_finger_pos[1])**2)
                                
                                # 如果移动太小（抖动）或太大（跳跃），则忽略
                                if distance < 3 or distance > 50:
                                    is_valid_move = False
                            
                            if is_valid_move:
                                finger_points.append(current_pos)
                                current_stroke.append(current_pos)  # 添加到当前笔画

                                # 如果有上一个位置，则绘制线段
                                if last_finger_pos:
                                    # 在原始图像上绘制轨迹 (只绘制当前线段)
                                    cv2.line(display_image, last_finger_pos, current_pos, (255, 0, 255), 4)
                                    # 在画布上绘制轨迹（确保线粗一些，更容易看到）
                                    cv2.line(canvas, last_finger_pos, current_pos, (0, 0, 0), 8)
                                else:
                                    # 如果是第一个点，绘制一个点
                                    cv2.circle(canvas, current_pos, 4, (0, 0, 0), -1)

                                last_finger_pos = current_pos

                                # 限制点的数量以防止性能问题
                                if len(finger_points) > 1000:
                                    finger_points.pop(0)
                else:
                    # 暂时保持绘制状态，防止短暂的检测失败导致绘制中断
                    if is_drawing and last_finger_pos and frame_count % 60 == 0:
                        print("暂时未检测到手部，但维持绘制状态")
                    
                    # 只在每5帧显示一次警告，减少屏幕闪烁
                    if frame_count % 5 == 0:
                        display_image = cv2_put_chinese_text(
                            display_image,
                            "未检测到手部",
                            (10, 70),
                            (0, 0, 255),
                            30
                        )

            # 始终重绘当前笔画和所有之前笔画，确保它们在原始图像上可见
            # 先绘制之前完成的笔画
            for stroke in all_stroke_points:
                for i in range(1, len(stroke)):
                    cv2.line(display_image, stroke[i - 1], stroke[i], (255, 0, 255), 4)  # 4px粗细

            # 然后重绘当前正在绘制的笔画
            for i in range(1, len(current_stroke)):
                cv2.line(display_image, current_stroke[i - 1], current_stroke[i], (255, 0, 255), 4)  # 4px粗细

        # 创建一个结果显示区域
        result_panel = np.ones((200, display_image.shape[1], 3), dtype=np.uint8) * 240

        # 如果已经进行预测，显示结果
        if made_prediction:
            # 减少输出频率，避免日志过多
            if frame_count % 30 == 0:
                print(f"显示预测结果: {prediction_result}")

            # 创建一个全新的结果面板，避免叠加显示
            result_panel = np.ones((200, display_image.shape[1], 3), dtype=np.uint8) * 240

            # 在结果面板上使用更大更醒目的字体显示预测结果
            cv2.putText(
                result_panel,
                prediction_result,
                (result_panel.shape[1] // 2 - 50, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                5.0,  # 更大的字体
                (0, 0, 255),
                15,  # 更粗的线条
                cv2.LINE_AA
            )
        elif ready_to_predict:
            # 在结果面板上显示"准备预测"状态
            result_panel = np.ones((200, display_image.shape[1], 3), dtype=np.uint8) * 240
            result_panel = cv2_put_chinese_text(
                result_panel,
                "绘制完成，按'p'键进行预测",
                (20, 100),
                (255, 0, 0),
                40
            )
        else:
            if is_drawing:
                status = "正在绘制..."
            elif is_tracking:
                status = "手部追踪中，按'd'开始绘制"
            else:
                status = "按's'开始手部追踪"

            result_panel = cv2_put_chinese_text(
                result_panel,
                status,
                (20, 100),
                (0, 0, 0),
                36
            )

        # 为画布添加边框，使其更容易看到
        canvas_display = canvas.copy()
        cv2.rectangle(canvas_display, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), (0, 0, 0), 2)

        # 始终重新绘制所有轨迹点到画布上，确保轨迹可见
        # 先清除画布上的绘制内容但保留边框
        if ready_to_predict or made_prediction:
            # 只在准备预测或已预测状态下保留画布内容
            pass
        else:
            # 在其他状态下，重新绘制所有轨迹
            temp_canvas = np.ones((480, 640), np.uint8) * 255
            cv2.rectangle(temp_canvas, (0, 0), (temp_canvas.shape[1] - 1, temp_canvas.shape[0] - 1), (0, 0, 0), 2)

            # 重绘当前正在绘制的笔画
            for i in range(1, len(current_stroke)):
                cv2.line(temp_canvas, current_stroke[i - 1], current_stroke[i], (0, 0, 0), 8)

            # 重绘所有之前完成的笔画
            for stroke in all_stroke_points:
                for i in range(1, len(stroke)):
                    cv2.line(temp_canvas, stroke[i - 1], stroke[i], (0, 0, 0), 8)

            canvas_display = temp_canvas

        # 将结果面板与显示图像垂直拼接
        combined_image = np.vstack((display_image, result_panel))

        # 如果处于非追踪状态，在画布上添加提示文字
        if not is_tracking:
            # 转换灰度图为BGR以便使用颜色绘制文字
            canvas_display_color = cv2.cvtColor(canvas_display, cv2.COLOR_GRAY2BGR)
            canvas_display_color = cv2_put_chinese_text(
                canvas_display_color,
                "请先按's'开启手部追踪，然后按'd'开始绘制",
                (canvas.shape[1] // 2 - 300, canvas.shape[0] // 2),
                (0, 0, 255),
                36
            )
            canvas_display = canvas_display_color
        else:
            # 当有效跟踪时也转换为彩色以便显示文字
            canvas_display_color = cv2.cvtColor(canvas_display, cv2.COLOR_GRAY2BGR)

            # 如果正在绘制，显示提示
            if is_drawing:
                canvas_display_color = cv2_put_chinese_text(
                    canvas_display_color,
                    "正在绘制...",
                    (20, 30),
                    (0, 255, 0),
                    30
                )
            elif ready_to_predict:
                canvas_display_color = cv2_put_chinese_text(
                    canvas_display_color,
                    "绘制结束，按'p'预测",
                    (20, 30),
                    (0, 0, 255),
                    30
                )

            canvas_display = canvas_display_color

        # 显示画面和画布
        cv2.imshow('OpenCV空中写字', combined_image)
        cv2.imshow('画布', canvas_display)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()