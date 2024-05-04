import cv2
import numpy as np
from sklearn.neighbors import KDTree

def match_features(descriptors1, descriptors2, ratio=0.75):
    # 构建 KD 树
    kdtree = KDTree(descriptors2)

    matches = []

    for i in range(len(descriptors1)):
        # 查询 KD 树，找到距离最近的两个特征点
        query_descriptor = descriptors1[i]
        distances, indices = kdtree.query([query_descriptor], k=2)

        # 检查最近邻的距离是否满足最近邻和次近邻距离的比率条件
        if distances[0][0] < ratio * distances[0][1]:
            # 如果满足条件，则将匹配的特征点索引添加到匹配列表中
            matches.append(cv2.DMatch(i, indices[0][0], distances[0][0]))

    return matches

# 读取两幅图像
image1 = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

# 创建 SIFT 特征检测器
sift = cv2.SIFT_create()

# 检测图像中的特征点和计算描述符
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# 调用 match_features 函数进行匹配
matches = match_features(descriptors1, descriptors2)

# 绘制匹配结果
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 显示匹配结果
cv2.imshow("Matches", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
