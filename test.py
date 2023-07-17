import cv2
import numpy as np
import torch

# def gray_norm(img):
#     """
#     灰度归一化
#     :param img:
#     :return:
#     """
#     min_value = np.min(img)
#     max_value = np.max(img)
#     if max_value == min_value:
#         return img
#     (n, m) = img.shape
#     for i in range(n):
#         for j in range(m):
#             img[i, j] = int(255 * (img[i][j] - min_value) / (max_value - min_value))
#     return img


# def normailiztaion(img, dets, shape_list):
#     """
#     图像尺度灰度归一化
#     :param img:
#     :param dets:
#     :param shape_list:
#     :return:
#     """
#     # 灰度归一化
#     # img = gray_norm(img)
#
#     # 尺度归一化
#     img_list = []
#     pt_pos_list = []
#     for index, face in enumerate(dets):
#         left = face.left()
#         top = face.top()
#         right = face.right()
#         bottom = face.bottom()
#         img1 = img[top:bottom, left:right]
#         size = (48, 48)
#         img1 = cv.resize(img1, size, interpolation=cv.INTER_LINEAR)
#
#         pos = []
#         for _, pt in enumerate(shape_list[index].parts()):
#             pt_pos = (int((pt.x - left) / (right - left) * 90), int((pt.y - top) / (bottom - top) * 100))
#             pos.append(pt_pos)
#             cv.circle(img1, pt_pos, 2, (255, 0, 0), 1)
#         pt_pos_list.append(pos)
#         img_list.append(img1)
#     return img_list, pt_pos_list


map = {0:'angry',1:'disgust',2:'fear',3:'happiness',4:'sadness',6:'surprise',5:'neural'}
face = cv2.imread('D:\FERNet-master\FERNet-master\datasets\\test\\2.jpg')
# 读取单通道灰度图
face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
face_noise = cv2.blur(face_gray,(5,5))
# 直方图均衡化
face_hist = cv2.equalizeHist(face_noise)
face_resize = cv2.resize(face_hist,(48,48))
face_normalized = face_resize.reshape(1, 48, 48) / 255.0
face_tensor = torch.from_numpy(face_normalized)
face_tensor = face_tensor.type('torch.FloatTensor')
model  = torch.load("model_net.pkl")
model.eval
with torch.no_grad():
    output = model(face_tensor.unsqueeze(0))
print(output)
# l = output.tolist()[0]
# m = l
# f = max(l)
# print(f"I guess that your mood is {map[m.index(f)]}")
# l.remove(f)
# s = max(l)
# if s>3:
#     print(f"well or maybe {map[m.index(s)]}")
#     l.remove(s)
#     t = max(l)
#     if t > 3:
#         print(f"or {map[m.index(t)]}?")