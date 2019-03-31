import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import pytesseract
import os


def identify(pic_path, show_process=False, print_info=False):
    """
    身份证信息识别
    :param pic_path: 图片路径
    :param show_process: 显示处理过程
    :param print_info: 显示识别信息
    :return: 识别出的信息dict
    """
    try:
        img = cv2.imread(pic_path)[..., :: -1]  # bgr2rgb
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)  # 图像去噪
        k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # 构造滤波器
        img = cv2.filter2D(img, -1, k)  # 锐化图像
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)  # 转灰度
        mean = np.mean(gray)  # 求平均色彩值
        gray = gray - mean  # 对每个像素减去平均值
        gray = gray * 1.5 + mean * 1.3  # 修正对比度和亮度
        _, thresh = cv2.threshold(gray.copy(), 127, 255, 0)
        # thresh = cv2.adaptiveThreshold(gray.copy(), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

        text_area = thresh[100: 700, 250: 860]  # 裁切信息区域
        text_area[110: 210, 100: 280] = 255  # 涂掉民族两个字
        id_area = thresh[700: 850, 450: 1300]  # 裁切证件号区域

        if show_process:
            plt.subplot(221)  # 选择绘图区域
            plt.title(pic_path)  # 设置标题
            plt.imshow(img)  # 加载图片
            plt.subplot(222)  # 选择绘图区域
            plt.title('Gray')  # 设置标题
            plt.imshow(gray, cmap='gray')  # 加载图片
            plt.subplot(223)  # 选择绘图区域
            plt.title('Info')  # 设置标题
            plt.imshow(text_area, cmap='gray')  # 加载图片
            plt.subplot(224)  # 选择绘图区域
            plt.title('Id')  # 设置标题
            plt.imshow(id_area, cmap='gray')  # 加载图片
            plt.show()  # 显示

        text = pytesseract.image_to_string(text_area, lang='chi_sim')  # 识别中文信息
        # card_id = pytesseract.image_to_string(id_area).replace(' ', '')  # 识别身份证号
        # 下面一行是定制版本，需要手动修改配置文件
        card_id = pytesseract.image_to_string(id_area, config='-psm 7 sfz').replace(' ', '')

        text_list = text.split('\n')  # 按行分割成list
        text_df = pd.DataFrame({"text": text_list})  # 生成DataFrame
        text_df['len'] = text_df.text.apply(len)  # 统计每行的长度
        text_df = text_df[text_df.len > 1].reset_index(drop=True)
        # 处理生日
        year_num = card_id[6: 10]
        month_num = card_id[10: 12]
        day_num = card_id[12: 14]

        address = ''.join(text_df.text.values[3:]).replace(' ', '')  # 处理地址

        if print_info:
            print('\t姓名:\t', text_df.text[0].replace(' ', ''))
            print('\t性别:\t', text_df.text[1].split(' ')[0])
            print('\t民族:\t', text_df.text[1].split(' ')[-1])
            print('\t生日:\t {}年{}月{}日'.format(year_num, month_num, day_num))
            print('\t地址:\t', address)
            print('\t身份证号:\t', card_id)

        return {
            'name': text_df.text[0].replace(' ', ''),
            'sex': text_df.text[1].split(' ')[0],
            'nation': text_df.text[1].split(' ')[-1],
            'birthday': '{}年{}月{}日'.format(year_num, month_num, day_num),
            'address': address,
            'id': card_id,
        }
    except Exception as e:
        print('OCR模块遇到未知错误', e.args)


def find_cross_point(line1, line2):
    """
    求两直线交点
    :param line1: 直线1
    :param line2: 直线2
    :return: 交点坐标
    """
    x0, y0, x1, y1 = line1
    a0, b0, c0 = y0 - y1, x1 - x0, x0 * y1 - x1 * y0
    x0, y0, x1, y1 = line2
    a1, b1, c1 = y0 - y1, x1 - x0, x0 * y1 - x1 * y0
    d = a0 * b1 - a1 * b0
    return int((b0 * c1 - b1 * c0) / d), int((a1 * c0 - a0 * c1) / d)


def resize(pic_path, save_path, show_process=False):
    """
    检测最大轮廓并进行透视变换和裁剪
    默认大小1400x900 （身份证比例
    :param save_path: 存储路径, 处理后的图像保存在指定路径, 文件名和源文件相同
    :param show_process: 显示处理过程
    :param pic_path: 原图路径
    :return:
    """
    try:
        img = cv2.imread(pic_path)  # 加载图片
        img2 = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)  # 图像去噪
        k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # 构造滤波器
        img2 = cv2.filter2D(img2, -1, k)  # 锐化图像
        gray = cv2.cvtColor(img2.copy(), cv2.COLOR_BGR2GRAY)  # 灰度
        # gray = np.uint8(np.clip((3 * gray - 100), 0, 255))  # 修改对比度和亮度(该项调整对于多数照片有负面效果, 暂时不使用)
        _, thresh = cv2.threshold(gray.copy(), 127, 255, 0)  # 二值化
        # thresh = cv2.adaptiveThreshold(gray.copy(), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 按面积排序
        if cv2.contourArea(contours[0]) < 5000000:
            cv2.imwrite(os.path.join(save_path, pic_path.split('/')[-1]), img)
        fill = cv2.rectangle(thresh.copy(), (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)  # 将图片涂黑
        fill = cv2.drawContours(fill.copy(), contours, 0, (255, 255, 255), -1)  # 将最大轮廓涂白
        edges = cv2.Canny(fill.copy(), 50, 150, apertureSize=3)  # Canny算子边缘检测
        lines = cv2.HoughLines(edges, 1, np.pi / 270, 180)  # 霍夫直线检测
        horizontal, vertical = [], []  # 创建水平和垂直线list
        for line in lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)

                # 画出所有直线
                # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                if abs(x1 - x2) > abs(y1 - y2):
                    horizontal.append([x1, y1, x2, y2])
                else:
                    vertical.append([x1, y1, x2, y2])
        # 按直线中点位置排序
        horizontal = sorted(horizontal, key=lambda l: l[1] + l[1] - l[3])
        vertical = sorted(vertical, key=lambda l: l[0] + l[0] - l[2])
        # 选出上下左右边缘的四条线
        top = horizontal[0]
        bottom = horizontal[-1]
        left = vertical[0]
        right = vertical[-1]
        # 计算交点
        t_l_point = find_cross_point(top, left)
        t_r_point = find_cross_point(top, right)
        b_l_point = find_cross_point(bottom, left)
        b_r_point = find_cross_point(bottom, right)
        # # 用红色画出四个顶点
        # for point in t_l_point, t_r_point, b_l_point, b_r_point:
        #     cv2.circle(img, point, 8, (0, 0, 255), 2)
        # # 用蓝色画出四条边
        # cv2.line(img, t_l_point, t_r_point, (255, 0, 0), 3)
        # cv2.line(img, b_r_point, t_r_point, (255, 0, 0), 3)
        # cv2.line(img, b_r_point, b_l_point, (255, 0, 0), 3)
        # cv2.line(img, b_l_point, t_l_point, (255, 0, 0), 3)

        # 透视变换
        width = 1400  # 生成图的宽
        height = 900  # 生成图的高
        # 原图中的四个角点
        pts1 = np.float32([list(t_l_point), list(t_r_point), list(b_l_point), list(b_r_point)])
        # 变换后分别在左上、右上、左下、右下四个点
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        # 生成透视变换矩阵
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # 进行透视变换
        dst = cv2.warpPerspective(img, M, (width, height))
        # 保存图片
        cv2.imwrite(os.path.join(save_path, pic_path.split('/')[-1]), dst)
        if show_process:
            plt.subplot(231)
            plt.title('Gray')
            plt.axis('off')
            plt.imshow(gray, cmap='gray')
            plt.subplot(232)
            plt.title('Thresh')
            plt.axis('off')
            plt.imshow(thresh, cmap='gray')
            plt.subplot(233)
            plt.title('Fill')
            plt.axis('off')
            plt.imshow(fill, cmap='gray')
            plt.subplot(234)
            plt.title('Edges')
            plt.axis('off')
            plt.imshow(edges, cmap='gray')
            plt.subplot(235)
            plt.title(pic_path)
            plt.axis('off')
            plt.imshow(img[..., :: -1])
            plt.subplot(236)
            plt.title('Dst')
            plt.axis('off')
            plt.imshow(dst[..., :: -1])
            plt.show()
    except AttributeError as e:
        print('读取文件失败: ', e.args)
    except Exception as e:
        print('图像矫正模块遇到未知错误: ', e.args)


if __name__ == '__main__':
    img_dir = 'images'
    save_dir = 'dst'
    files = os.listdir(img_dir)
    for idx, file in enumerate(files):
        if not os.path.isdir(file) and file[0] != '.' and file == 'IMG_6593.JPG':
            print('=' * 20 + os.path.join(img_dir, file).center(30) + '=' * 20)
            tic = time.time()
            resize(os.path.join(img_dir, file), save_dir, show_process=False)
            _ = identify(os.path.join(save_dir, file), show_process=False, print_info=True)
            toc = time.time()
            print('=' * 20 + 'Time: {}s'.format(toc - tic).center(30) + '=' * 20)
