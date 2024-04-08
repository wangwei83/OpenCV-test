import cv2
import numpy as np

def bilateral_filter(image, d, sigmaColor, sigmaSpace):
    """
    对图像应用双边滤波。

    参数:
    - image: 输入图像。
    - d: 滤波时周围每个像素领域的直径。
    - sigmaColor: 颜色空间的标准差。颜色空间的大的sigma值意味着颜色越远的像素将彼此影响，从而产生更大范围的颜色混合。
    - sigmaSpace: 坐标空间中的标准差。空间参数的大的sigma值意味着距离较远的像素将相互影响，只要它们的颜色足够接近。

    返回:
    - 过滤后的图像。
    """
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

# 加载图像
image = cv2.imread('5.jpg')

# 应用双边滤波
filtered_image = bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Bilateral Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
