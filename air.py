import cv2
import numpy as np
import mediapipe as mp
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import os
import platform



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

    # for font_path in possible_fonts:
    #     if os.path.exists(font_path):
    #         try:
    #             fontStyle = ImageFont.truetype(font_path, textSize)
    #             break
    #         except:
    #             continue

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


# 添加握拳检测函数
def is_fist(hand_landmarks):
    """
    检测是否为握拳手势
    通过检查手指尖的位置是否低于手指第二关节来判断
    """
    # 获取手指关键点坐标
    finger_tips = [8, 12, 16, 20]  # 食指、中指、无名指、小指的指尖
    finger_pips = [6, 10, 14, 18]  # 食指、中指、无名指、小指的第二关节

    # 大拇指判断方式不同（水平方向）
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]

    # 检查大拇指是否弯曲（放宽条件）
    thumb_bent = thumb_tip.x > thumb_ip.x  # 对于右手

    # 判断其他手指是否弯曲（当指尖y值大于第二关节y值时，表示手指弯曲）
    # 放宽阈值要求
    fingers_bent = 0
    for tip_id, pip_id in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip_id].y > hand_landmarks.landmark[pip_id].y:
            fingers_bent += 1

    # 放宽条件：大拇指弯曲，且至少3个其他手指弯曲
    return thumb_bent and fingers_bent >= 3


def is_victory_gesture(hand_landmarks):
    """
    检测是否为耶（✌️）手势
    通过检查食指和中指是否伸直，其他手指是否弯曲来判断
    """
    # 获取各手指的关键点
    index_tip = hand_landmarks.landmark[8]  # 食指尖
    index_pip = hand_landmarks.landmark[6]  # 食指第二关节
    middle_tip = hand_landmarks.landmark[12]  # 中指尖
    middle_pip = hand_landmarks.landmark[10]  # 中指第二关节
    ring_tip = hand_landmarks.landmark[16]  # 无名指尖
    ring_pip = hand_landmarks.landmark[14]  # 无名指第二关节
    pinky_tip = hand_landmarks.landmark[20]  # 小指尖
    pinky_pip = hand_landmarks.landmark[18]  # 小指第二关节
    thumb_tip = hand_landmarks.landmark[4]  # 拇指尖
    thumb_ip = hand_landmarks.landmark[3]  # 拇指第二关节

    # 检查食指和中指是否伸直（指尖y值小于关节y值）
    index_straight = index_tip.y < index_pip.y
    middle_straight = middle_tip.y < middle_pip.y

    # 检查其他手指是否弯曲
    ring_bent = ring_tip.y > ring_pip.y
    pinky_bent = pinky_tip.y > pinky_pip.y
    thumb_bent = thumb_tip.x > thumb_ip.x  # 拇指弯曲判断（对右手）

    # 返回True如果：
    # 1. 食指和中指伸直
    # 2. 其他手指弯曲
    return index_straight and middle_straight and ring_bent and pinky_bent and thumb_bent


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

    # 初始化MediaPipe Hands
    mp_hands = None
    hands = None
    use_mediapipe = True

    try:
        print("正在初始MediaPipe手部追踪...")
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # 只检测一只手
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils
        print("MediaPipe手部追踪初始化完成！")
    except Exception as e:
        print(f"MediaPipe初始化失败: {e}")
        print("将使用简单的鼠标控制作为替代方案")
        use_mediapipe = False

    # 加载模型
    try:
        print("正在加载字符识别模型...")
        model = Cnn()
        model.load_state_dict(torch.load('model_emnist.pt', map_location='cpu'))
        print("模型加载成功！")
    except Exception as e:
        print(f"错误：无法加载模型 - {e}")
        print("请确保model_emnist.pt文件存在于当前目录")
        if hands:
            hands.close()
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

    # 存储手指指尖位置或鼠标位置
    finger_points = []
    last_mouse_pos = None  # 用于鼠标控制方案

    # 鼠标回调函数（如果MediaPipe不可用）
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

    # 如果MediaPipe不可用，设置鼠标回调
    if not use_mediapipe:
        cv2.namedWindow('画布')
        cv2.setMouseCallback('画布', mouse_callback)

    # 显示帮助信息
    print("\n=== 空中写字识别系统 ===")
    if use_mediapipe:
        print("操作指南：")
        print("按 's' - 开始手部追踪")
        print("按 'd' - 开始绘制")
        print("握拳 - 结束绘制并准备预测")
        print("按 'p' - 预测绘制的字符")
        print("按 'c' - 清除画布")
        print("按 'q' - 退出程序")
    else:
        print("【使用鼠标模式】")
        print("按 'd' - 开始绘制（移动鼠标进行绘制）")
        print("按 'c' - 清除画布")
        print("按 'p' - 预测绘制的字符")
        print("按 'q' - 退出程序")
    print("========================\n")

    frame_count = 0  # 帧计数器，用于调试

    # 添加新状态变量用于防止误触发
    victory_detection_counter = 0  # 耶手势检测计数器
    victory_detection_threshold = 3  # 需要连续检测到耶手势的次数

    # 添加轨迹绘制相关变量
    all_stroke_points = []  # 存储所有绘制点的列表，每个元素是一组点
    current_stroke = []  # 当前正在绘制的笔画

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
        if use_mediapipe:
            if is_tracking:
                status_text.append("追踪: 开启")
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

        # 开始追踪 (仅MediaPipe模式)
        if key == ord('s') and use_mediapipe:
            is_tracking = True
            print("手部追踪已启动！")

        # 开始绘
        elif key == ord('d'):
            is_drawing = True
            last_mouse_pos = None  # 重置鼠标位置
            last_finger_pos = None  # 重置指尖位置
            current_stroke = []  # 开始新的笔画
            print("绘制模式已启动！状态：", is_drawing)

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

                    # 保存ROI图像
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

        # 如果使用MediaPipe且追踪模式开启，进行手部检测
        if use_mediapipe and is_tracking:
            # 将BGR图像转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 使用MediaPipe处理图像
            results = hands.process(image_rgb)

            # 如果检测到手
            if results.multi_hand_landmarks:
                # 绘制手部标记
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # 获取食指指尖位置 (索引为8)
                    finger_tip = hand_landmarks.landmark[8]
                    h, w, c = image.shape
                    px, py = int(finger_tip.x * w), int(finger_tip.y * h)

                    # 在图像上显示指尖位置
                    cv2.circle(display_image, (px, py), 10, (0, 255, 0), -1)

                    # 检测耶手势
                    if is_drawing and is_victory_gesture(hand_landmarks):
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

                    # 如果正在绘制，将点添加到轨迹中
                    if is_drawing:
                        current_pos = (px, py)

                        # 添加调试输出
                        if frame_count % 30 == 0:
                            print(f"正在绘制，当前指尖位置: {current_pos}")

                        # 确保current_pos在画布范围内
                        if 0 <= current_pos[0] < canvas.shape[1] and 0 <= current_pos[1] < canvas.shape[0]:
                            # 只有当位置变化足够大时才添加新点（避免小抖动）
                            if last_finger_pos is None or ((abs(current_pos[0] - last_finger_pos[0]) > 3) or
                                                           (abs(current_pos[1] - last_finger_pos[1]) > 3)):
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

                                # 在每一帧都重新绘制完整的当前笔画轨迹，确保可见性
                                temp_display = display_image.copy()
                                for i in range(1, len(current_stroke)):
                                    cv2.line(display_image, current_stroke[i - 1], current_stroke[i], (255, 0, 255), 4)

                if frame_count % 30 == 0:
                    print(
                        f"已检测到手部，手指位置: ({px}, {py}), 绘制状态: {is_drawing}, 轨迹点数: {len(finger_points)}")
            else:
                if is_tracking and frame_count % 30 == 0:
                    print("未检测到手部，请将手放在摄像头视野内")
                if is_tracking:
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

        # 如果不是使用MediaPipe或者处于非追踪状态，在画布上添加提示文字
        if not use_mediapipe or not is_tracking:
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
        cv2.imshow('MediaPipe空中写字', combined_image)
        cv2.imshow('画布', canvas_display)

        # 移除这里的额外waitKey调用，避免干扰键盘输入
        # cv2.waitKey(1)  # 这行代码是导致问题的原因，现已注释掉

    # 释放资源
    if hands:
        hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
