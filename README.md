# IDCardIdentify

身份证正面信息识别



## 环境

Python3.6.x，第三方库安装命令为`pip install -r requirement.txt`



## 使用方法

将需要识别的图片放置在`/images/`目录下，经过矫正的图片结果会自动存放在`/dst/`目录下



## 原型

```python
def resize(pic_path, save_path, show_process=False):
    """
    检测最大轮廓并进行透视变换和裁剪
    默认大小1400x900 （身份证比例
    :param save_path: 存储路径, 处理后的图像保存在指定路径, 文件名和源文件相同
    :param show_process: 显示处理过程
    :param pic_path: 原图路径
    :return:
    """
```



```python
def identify(pic_path, show_process=False, print_info=False):
    """
    身份证信息识别
    :param pic_path: 图片路径
    :param show_process: 显示处理过程
    :param print_info: 显示识别信息
    :return: 识别出的信息dict
    """
```

