import os

from pynput import keyboard, mouse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import locale

os.environ['YOLO_VERBOSE'] = str(False)  # 禁用YOLO的详细输出，减少控制台干扰

# 导入必要的库
import time  # 用于时间控制和延时
import cv2  # 图像处理库
import mss  # 用于屏幕截图
import pygetwindow as gw  # 窗口管理
from ultralytics import YOLO  # 导入YOLO

# 设置系统默认编码
locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

# 加载YOLO模型
model = YOLO('best.pt')

# 定义一个固定的窗口标题
WINDOW_TITLE = ' '

# 添加全局变量的定义
closest_distance = None  # 存储当前最近的目标距离

def get_window_rect(window_title):
    # 获取指定标题的窗口
    window = gw.getWindowsWithTitle(window_title)[0]
    return {
        "top": window.top,
        "left": window.left,
        "width": window.width,
        "height": window.height
    }


def is_window_active(window_title):
    try:
        window = gw.getWindowsWithTitle(window_title)[0]
        return window.isActive
    except IndexError:
        print("未找到指定窗口")
        return False
    except Exception as e:
        print(f"检查窗口状态时发生错误: {str(e)}")
        return False


# 基准系数（需要通过实验调整）
DISTANCE_TO_TIME_RATIO = 3.37  # 每单位距离对应的按压时间（毫秒）

# 鼠标控制器
mouse_controller = mouse.Controller()
# 键盘控制器
keyboard_controller = keyboard.Controller()

def activate_window(window_title):
    try:
        window = gw.getWindowsWithTitle(window_title)[0]
        if not window.isActive:
            window.activate()
            time.sleep(0.1)  # 等待窗口激活
        return window
    except Exception as e:
        print(f"激活窗口失败: {str(e)}")
        return None


def move_mouse_to_window_center(window):
    try:
        # 计算窗口中心位置
        center_x = window.left + window.width // 2
        center_y = window.top + window.height // 2
        # 移动鼠标到窗口中心
        mouse_controller.position = (center_x, center_y)
        time.sleep(0.1)  # 等待鼠标移动
        return True
    except Exception as e:
        print(f"移动鼠标失败: {str(e)}")
        return False


# 添加跳跃冷却控制
last_jump_time = 0
JUMP_COOLDOWN = 2.1  # 跳跃冷却时间（秒）
last_frame = None  # 存储最后一帧的绘制结果
save_last_frame = False  # 控制是否保存最后一帧

def can_jump():
    current_time = time.time()
    global last_jump_time
    if current_time - last_jump_time >= JUMP_COOLDOWN:
        last_jump_time = current_time
        return True
    return False

def jump(distance: float):
    # 添加参数类型检查
    if not isinstance(distance, (int, float)) or distance <= 0:
        # print("无效的跳跃距离")
        return

    if not can_jump():
        return  # 如果在冷却中，直接返回

    # 先激活窗口
    window = activate_window("跳一跳")
    if not window:
        print("无法激活目标窗口")
        return

    # 移动鼠标到窗口中心
    if not move_mouse_to_window_center(window):
        print("无法移动鼠标到窗口中心")
        return

    # 计算按压时间（毫秒）
    press_time = distance * DISTANCE_TO_TIME_RATIO
    # print(f"距离: {distance:.1f}, 按压时间: {press_time:.1f}ms")

    # 按下鼠标左键
    mouse_controller.press(mouse.Button.left)
    # 等待计算出的时间
    time.sleep(press_time / 1000.0)  # 转换为秒
    # 释放鼠标左键
    mouse_controller.release(mouse.Button.left)


def on_space_press(key):
    try:
        if key == keyboard.Key.space and isinstance(closest_distance, float):  # 确保是float类型
            # print(f'跳跃距离: {closest_distance}')
            jump(closest_distance)
    except Exception as e:
        print(f"跳跃时发生错误: {str(e)}")


# 添加函数用于绘制中文文字
def put_chinese_text(img, text, position, font_size, color):
    # 创建PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建画布
    draw = ImageDraw.Draw(img_pil)
    # 使用微软雅黑字体
    font = ImageFont.truetype("msyh.ttc", font_size)
    # 绘制文字
    draw.text(position, text, font=font, fill=color[::-1])
    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def process_frame(frame, _window_height, _window_width):
    global closest_distance, last_frame, save_last_frame
    closest_distance = 0.0  # 初始化为0.0而不是None

    # 如果在冷却中且有上一帧且允许使用最后一帧
    current_time = time.time()
    if current_time - last_jump_time < JUMP_COOLDOWN and last_frame is not None and save_last_frame:
        cv2.imshow(WINDOW_TITLE, last_frame)
        cv2.waitKey(1)
        return

    # 运行YOLO检测
    results = model.predict(
        frame,
        half=False,
        imgsz=(640, 640),
        batch=1,
        conf=0.70,
        iou=0.45,
        max_det=10,
        device='cuda:0',
        verbose=False,
        augment=False,
        visualize=False,
        stream=True
    )

    # 处理所有结果
    for r in results:
        boxes = r.boxes
        annotated_frame = frame.copy()

        # 存储own和target的信息
        own_boxes = []
        target_boxes = []

        # 分类所有检测框
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            if cls == 1:  # target
                target_boxes.append((x1, y1, x2, y2, conf))
            else:  # own
                own_boxes.append((x1, y1, x2, y2, conf))

        # 绘制所有own并计算距离
        for x1, y1, x2, y2, conf in own_boxes:
            # 绘制own
            cv2.rectangle(annotated_frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 0, 255), 2)
            # 使用中文标签
            annotated_frame = put_chinese_text(
                annotated_frame,
                # f"流萤衔星泊野渡, {conf:.2f}",
                f"流萤衔星泊野渡,",
                (int(x1), int(y1 - 30)),
                18,
                (0, 0, 255)
            )

            # 获取own的底部中心点（使用原始坐标）
            own_bottom_x = (x1 + x2) / 2
            own_bottom_y = y2

            closest_target = None
            min_distance = float('inf')

            for tx1, ty1, tx2, ty2, tconf in target_boxes:
                target_center_x = (tx1 + tx2) / 2
                target_center_y = (ty1 + ty2) / 2

                if target_center_y < own_bottom_y:
                    # 使用原始坐标计算距离
                    distance = ((target_center_x - own_bottom_x) ** 2 +
                                (target_center_y - own_bottom_y) ** 2) ** 0.5

                    if distance < min_distance:
                        min_distance = distance
                        closest_target = (tx1, ty1, tx2, ty2, tconf, distance)

            # 绘制最近的target和显示距离
            if closest_target:
                tx1, ty1, tx2, ty2, tconf, distance = closest_target
                cv2.rectangle(annotated_frame,
                              (int(tx1), int(ty1)),
                              (int(tx2), int(ty2)),
                              (0, 255, 0), 2)

                # 使用中文标签
                annotated_frame = put_chinese_text(
                    annotated_frame,
                    # f"坠露凝光卧掌纹。 {tconf:.2f}",
                    f"坠露凝光卧掌纹.",
                    (int(tx1), int(ty1 - 30)),
                    18,
                    (0, 0, 255)
                )

                # 绘制从own底部中心到target中心的连线
                target_center_x = (tx1 + tx2) / 2
                target_center_y = (ty1 + ty2) / 2
                cv2.line(annotated_frame,
                         (int(own_bottom_x), int(own_bottom_y)),
                         (int(target_center_x), int(target_center_y)),
                         (0, 165, 255), 2)

                # 更新全局距离值
                closest_distance = float(distance)  # 确保转换为float类型
                # print(f"更新距离: {closest_distance}")

                # 显示距离和按压时间信息
                press_time = distance * DISTANCE_TO_TIME_RATIO
                distance_info = f"Distance: {distance:.1f}, Press: {press_time:.1f}ms"
                annotated_frame = put_chinese_text(
                    annotated_frame,
                    distance_info,
                    (int(tx1 - 70), int(ty1 - 60)),
                    18,
                    (0, 165, 255)
                )

        # 在左上角显示窗口信息
        # window_info = f"window size: {_window_width}x{_window_height}"
        window_info = f"《萤星渡掌河》"
        annotated_frame = put_chinese_text(
            annotated_frame,
            window_info,
            (10, 20),
            18,
            (128, 0, 128)
        )

        # 使用UTF-8编码的标题显示窗口
        cv2.imshow(WINDOW_TITLE, annotated_frame)
        cv2.waitKey(1)
        # 只在允许的情况下保存最后一帧
        if save_last_frame:
            last_frame = annotated_frame.copy()
        break


def set_window_size(window_title, target_width=460, target_height=857):
    try:
        window = gw.getWindowsWithTitle(window_title)[0]
        # 如果窗口大小不是目标大小
        if window.width != target_width or window.height != target_height:
            # 调整窗口大小
            window.resizeTo(target_width, target_height)
            print(f"\n已将窗口调整为: {target_width}x{target_height}")
            # time.sleep(0.5)  # 等待窗口调整完成
        return True
    except Exception as e:
        print(f"\n调整窗口大小失败: {str(e)}")
        return False


def main():
    global closest_distance
    closest_distance = None

    # 设置键盘监听
    listener = keyboard.Listener(on_press=on_space_press)
    listener.start()
    
    # 创建窗口
    window_title = create_window()
    
    # 初始化屏幕捕获
    with mss.mss() as sct:
        try:
            while True:
                # 检查并调整窗口大小
                if not set_window_size("跳一跳"):
                    print("\n未找到目标窗口")
                    time.sleep(1)
                    continue
                
                window_rect = get_window_rect("跳一跳")
                
                # 调整窗口大小时使用相同的标题
                cv2.resizeWindow(window_title, window_rect["width"], window_rect["height"])
                
                monitor = {
                    "top": window_rect["top"],
                    "left": window_rect["left"],
                    "width": window_rect["width"],
                    "height": window_rect["height"],
                    "mon": 1
                }

                # 捕获指定窗口
                screenshot = sct.grab(monitor)
                # 转换为numpy数组
                frame = np.array(screenshot)
                # 转换颜色空间从BGRA到BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # 处理帧并显示结果，添加调试信息
                process_frame(frame, window_rect["height"], window_rect["width"])
                # print(f"主循环中的closest_distance: {closest_distance}")
                
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                
        except IndexError:
            print("未找到标题为'跳一跳'的窗口")
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"主循环发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            listener.stop()
            cv2.destroyAllWindows()


# 添加切换函数（可选）
def toggle_frame_save():
    global save_last_frame
    save_last_frame = not save_last_frame
    print(f"{'启用' if save_last_frame else '禁用'}最后一帧保存")


def create_window():
    # 创建窗口时使用UTF-8编码的标题
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    return WINDOW_TITLE


if __name__ == '__main__':
    main()




















"""
《双璧谜·凝华版》 流萤衔星泊野渡，坠露凝光卧掌纹

注：两句暗藏三重机关与柔情——

解谜密钥：
"流萤"谐音"刘"（流→刘），萤火虫衔来"星"辰停泊
"坠露"即"雨"滴化身，在掌心凝成指纹般的光痕
爱意显影：
"衔星泊渡"喻穿越夜色奔赴的执着
"光卧掌纹"状小心翼翼珍藏星雨的温柔
通过昆虫与天象的意象嫁接，让姓名化作天地馈赠的掌中星河✨
"""




