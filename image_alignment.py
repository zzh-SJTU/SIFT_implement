#!/usr/bin/env python3
import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter,gaussian_filter
import math
def detect_blobs(image):
  """Laplacian blob detector.
  Args:
  - image (2D float64 array): A grayscale image.
  Returns:
  - corners (list of 2-tuples): A list of 2-tuples representing the locations
      of detected blobs. Each tuple contains the (x, y) coordinates of a
      pixel, which can be indexed by image[y, x].
  - scales (list of floats): A list of floats representing the scales of
      detected blobs. Has the same max_sig_round as `corners`.
  - orientations (list of floats): A list of floats representing the dominant
      orientation of the blobs.
  """
  height, width = image.shape
  sigma_0 = 1
  s = 1.5
  sigma_space_list = []
  gaus = []
  dog = []
  max_sig_round = 10
  ther1 = 0.04
  #将360度（对应math.atan2的-pi,pi）分成seperate个部分，建立orientation的直方图
  #用不同sigma的高斯滤波器进行滤波，构建sigma space的矩阵
  seperate = 36
  for t in range(max_sig_round):
    sigma_k = sigma_0 * (s**t)
    sigma_space_list.append(sigma_k)
    gau = cv2.GaussianBlur(image, ksize=[0,0], sigmaX=sigma_k, sigmaY=sigma_k)
    gaus.append(gau)
    if t>0:
      dog.append(gaus[t] - gaus[t-1])

  corners = []
  scales = []
  thre0 = 12.1
  orientations = []
  blob_list_max = []
  for k in range(1, max_sig_round-2):
    for i in range(1, height-1):
      for j in range(1,width-1):
          #由于需要在3个方向均取得极值，故需要将每个点与邻居28个点进行比较
        if dog[k][i,j] > dog[k][i+1,j] and dog[k][i,j] > dog[k][i-1,j] and dog[k][i,j] > dog[k][i,j+1] and dog[k][i,j] > dog[k][i,j-1] and dog[k][i,j] > dog[k][i+1,j-1] and dog[k][i,j] > dog[k][i-1,j-1] and dog[k][i,j] > dog[k][i+1,j+1] and dog[k][i,j] > dog[k][i-1,j+1] and dog[k][i,j] > dog[k-1][i,j] and dog[k][i,j] > dog[k-1][i-1,j] and dog[k][i,j] > dog[k-1][i+1,j] and dog[k][i,j] > dog[k-1][i,j-1] and dog[k][i,j] > dog[k-1][i,j+1] and dog[k][i,j] > dog[k-1][i-1,j-1] and dog[k][i,j] > dog[k-1][i+1,j-1] and dog[k][i,j] > dog[k-1][i+1,j+1] and dog[k][i,j] > dog[k-1][i-1,j+1] and dog[k][i,j] > dog[k+1][i,j] and dog[k][i,j] > dog[k+1][i-1,j] and dog[k][i,j] > dog[k+1][i+1,j] and dog[k][i,j] > dog[k+1][i,j-1] and dog[k][i,j] > dog[k+1][i,j+1] and dog[k][i,j] > dog[k+1][i-1,j-1] and dog[k][i,j] > dog[k+1][i+1,j-1] and dog[k][i,j] > dog[k+1][i+1,j+1] and dog[k][i,j] > dog[k+1][i-1,j+1]:
            #为了消除边缘效应，构建Hessian矩阵
            Dyy = dog[k][i+1,j]+dog[k][i-1,j]-2*dog[k][i][j]
            Dxx = dog[k][i,j+1]+dog[k][i,j-1]-2*dog[k][i][j]
            Dxy = ((dog[k][i+1,j+1]+dog[k][i-1,j-1])-(dog[k][i-1,j+1]+dog[k][i+1,j-1]))/4
            DetH = Dxx*Dyy- Dxy*Dxy
            trH = Dxx + Dyy
            #需要同时满足2个条件才能认为该点对应一个blob
            if trH*trH / DetH < thre0 and abs(dog[k][i,j])>ther1:
              hudu = 0
                #构建orientation的直方图，3sigma范围内
              orientation_his = np.zeros(seperate+1)
              ma_value = 0
            #对于每个blob在小矩形区域内进行循环求解orientation
              for m_index in range(int(i-3*sigma_space_list[k]), int(i+3*sigma_space_list[k])):
                for n_index in range(int(j-3*sigma_space_list[k]), int(j+3*sigma_space_list[k])):
                    #忽略边缘
                  if  m_index>=height-1 or n_index <=0 or n_index >=width-1 or m_index<=0:
                    continue
                  gr_abs = math.sqrt((dog[k][m_index+1,n_index]-dog[k][m_index-1,n_index])**2+(dog[k][m_index,n_index+1]-dog[k][m_index,n_index-1])**2)
                  #用差分计算梯度
                  theta = math.atan2((dog[k][m_index,n_index+1]-dog[k][m_index,n_index-1]),(dog[k][m_index+1,n_index]-dog[k][m_index-1,n_index]))
                  theta += math.pi
                  index = round(theta*seperate/(2*math.pi))
                  orientation_his[index] += gr_abs
                  #得到主方向
                  if orientation_his[index] > ma_value:
                    ma_value = orientation_his[round(index)]
                    hudu = index * 2 * math.pi / seperate
              blob_list_max.append(((j,i),hudu,sigma_space_list[k]))
              orientations.append(hudu)
              corners.append((j,i))
              scales.append(sigma_space_list[k])
    r_thre = 4
    #做一下非极大值抑制
    if max_sig_round<r_thre:
        for i,ele in enumerate(blob_list_max):
            m,n,k,value,sig = ele
            for j,ele2 in enumerate(blob_list_max):
                p,q,r,value2,sig2 = ele2
                if i == j:
                    continue
                else:
                    if (m-p)*(m-p)+(n-q)*(n-q)<r_thre:
                        if value>=value2:
                            blob_list_max.pop(j)
                        else:
                            blob_list_max.pop(i)             
  return corners, scales, orientations

def compute_descriptors(image, corners, scales, orientations):
  height, width = image.shape
  descriptors = []
  offs = 1
  """Compute descriptors for corners at specified scales.

  Args:
  - image (2d float64 array): A grayscale image.
  - corners (list of 2-tuples): A list of (x, y) coordinates.
  - scales (list of floats): A list of scales corresponding to the corners.
      Must have the same max_sig_round as `corners`.
  - orientations (list of floats): A list of floats representing the dominant
      orientation of the blobs.

  Returns:
  - descriptors (list of 1d array): A list of desciptors for each corner.
      Each element is an 1d array of max_sig_round 128.
  """
  if len(corners) != len(scales) or len(corners) != len(orientations):
    raise ValueError(
        '`corners`, `scales` and `orientations` must all have the same max_sig_round.')
  for co_index in range(len(scales)):
    (x, y) = corners[co_index]
    sigma = scales[co_index]
    ori = orientations[co_index]
    d = int(math.sqrt(3*sigma)) + 1
    rad = int(2 * d * math.sqrt(2))
    rad = min(rad, min(x, y)) 
    if d <offs:
        r = rad
        for co_id in range(len(corners)):
          dic_part = {}
        for v in range(d*d):
              dic_part[v] = []
        for i in range(2*r):
          for j in range(2*r):
              if i<r/2 and j<r/2:
                  dic_part[0].append((i,j))
              elif i>r/2 and j<r/2 and i<r:
                  dic_part[1].append((i,j))
              elif i>r and i<3*r/2 and j<r/2:
                  dic_part[2].append((i,j))
              elif i>3*r/2 and j<r/2:
                  dic_part[3].append((i,j))
              elif i<r/2 and j>r/2 and j<r:
                  dic_part[7].append((i,j))
              elif i>r/2 and j>r/2 and i<r and j<r:
                  dic_part[6].append((i,j))
              elif i>r and i<r+r/2 and j>r/2 and j<r:
                  dic_part[5].append((i,j))
              elif i>r+r/2 and j>r/2 and j<r:
                  dic_part[4].append((i,j))
              elif i<r/2 and j<r+r/2 and j>r:
                  dic_part[8].append((i,j))
              elif i>r/2 and i<r and j<r+r/2 and j>r:
                  dic_part[9].append((i,j))
              elif i>r and i<r+r/2 and j<r+r/2 and j>r:
                  dic_part[10].append((i,j))
              elif i>r+r/2 and j<r+r/2 and j>r:
                  dic_part[11].append((i,j))
              elif i<r/2 and j>r+r/2:
                  dic_part[15].append((i,j))
              elif i>r/2 and i<r and j>r+r/2:
                  dic_part[14].append((i,j))
              elif i>r and i<r+r/2 and j>r+r/2:
                  dic_part[13].append((i,j))
              elif i>r+r/2 and j>r+r/2:
                  dic_part[12].append((i,j))
    #取出每个blob对应r的小矩形
    square_new = image[int(max(0,y-rad)):int(min(height, y+rad+1)),int(max(0,x-rad)):int(min(width, x+rad+1))]
    sq = np.zeros((int(4*d+1),int(4*d+1)))
    for i in range(int(2*rad + 1)):
      for j in range(int(2*rad+1)):
        if(i>=square_new.shape[0] or j>=square_new.shape[1]):
          continue
        wq = j - rad
        wp = i - rad
        #旋转主方向
        new_x = int(wq * math.cos(ori) - wp * math.sin(ori))
        new_y = int(wp * math.cos(ori) + wq * math.sin(ori))
        if abs(new_x)<=2*d and abs(new_y)<=2*d:
          sq[int(new_y + 2*d),int(new_x+2*d)] = square_new[i,j]

    des_one = np.zeros(128)
    for i in range(1,4*d):
      for j in range(1,4*d):
        m = math.sqrt((sq[i+1,j]-sq[i-1,j])**2+(sq[i,j+1]-sq[i,j-1])**2)
        theta = math.atan2((sq[i,j+1]-sq[i,j-1]),(sq[i+1,j]-sq[i-1,j]))
        ind_sub = int(i / d) * 4 + int(j/d)
        pre_theta = int((theta + math.pi) / (2*math.pi) * 8)
        if pre_theta is None :
            continue
        if theta is None:
            continue
        if pre_theta == 8:
          pre_theta -= 1
        des_one[int(ind_sub*8+pre_theta)]+=m
    his_sum = sum(des_one)
    if pre_theta is None :
            continue
    if theta is None:
            continue
    des_one = des_one / math.sqrt(his_sum)
    descriptors.append(des_one)
  
  return descriptors


def match_descriptors(descriptors1, descriptors2):
    """Match descriptors based on their L2-distance and the "ratio test".

    Args:
    - descriptors1 (list of 1d arrays):
    - descriptors2 (list of 1d arrays):

    Returns:
    - matches (list of 2-tuples): A list of 2-tuples representing the matching
        indices. Each tuple contains two integer indices. For example, tuple
        (0, 42) indicates that corners1[0] is matched to corners2[42].
    """
    thre = 0.70
    matches=[]
    dist_mat = np.zeros((len(descriptors1), len(descriptors2)))
    #创建一个矩阵用于保存两个descriptors各个元素之间的距离
    for m, descriptor1 in enumerate(descriptors1):
        for n, descriptor2 in enumerate(descriptors2):
            #遍历每个descriptor计算l2-norm保存子在一个二维矩阵里面
            dist_mat[m, n] = np.linalg.norm((descriptor1 - descriptor2), ord=2)
        sorted_matrix = np.argsort(dist_mat[m, :])
        top1, top2 = sorted_matrix[:2]
        if dist_mat[m, top1] / dist_mat[m, top2] <= thre:
            matches.append((m, top1))
    return matches


def draw_matches(image1, image2, corners1, corners2, matches,
    outlier_labels=None):
    h1 = image1.shape[0]
    h2 = image2.shape[0]
    if outlier_labels is None:
        co = [(255, 0, 0)] * len(matches)
    else:
        co = [(0, 0, 255) if flag else (255, 0, 0)for flag in outlier_labels]
    max_height = max(image1.shape[0], image2.shape[0])
    #尺寸统一
    ori1 = np.pad(image1, [(0, max(0, max_height-h1)), (0, 0), (0, 0)])
    ori2 = np.pad(image2, [(0, max(0, max_height-h2)), (0, 0), (0, 0)])
    conca = np.concatenate((ori1, ori2), axis=1)
    for color, (i, j) in zip(co, matches):
        h1, w1 = list(map(int, corners1[i]))
        h2, w2 = list(map(int, corners2[j]))
        conca = cv2.circle(conca,(h1, w1),radius=3,color=(0, 255, 0),thickness=-1)
        conca = cv2.circle(conca,(h2+image1.shape[1], w2),radius=3,color=(0, 255, 0),thickness=-1)
        conca = cv2.line(conca,(h1, w1),(h2+image1.shape[1], w2),color=color,thickness=3)
    return conca


def compute_affine_xform(corners1, corners2, matches):
  """Compute affine transformation given matched  locations.

  Args:
  - corners1 (list of 2-tuples)
  - corners2 (list of 2-tuples)
  - matches (list of 2-tuples)

  Returns:
  - xform (2D float64 array): A 3x3 matrix representing the affine
      transformation that maps coordinates in image1 to the corresponding
      coordinates in image2.
  - outlier_labels (list of bool): A list of Boolean values indicating whether
      the corresponding match in `matches` is an outlier or not. For example,
      if `matches[42]` is determined as an outlier match after RANSAC, then
      `outlier_labels[42]` should have value `True`.
  """
  list_co1 = []  
  list_co2 = []
  inli_thre = 0
  thre_inlier = 13
  num_of_rounds = 7000
  for a, b in matches:
    list_co1.append(corners1[a])
    list_co2.append(corners2[b])
  list_co1= np.array(list_co1)
  list_co2= np.array(list_co2)
  temp1 = np.ones_like(list_co1[:, :1])
  temp2 = np.ones_like(list_co2[:, :1])
  list_co1 = np.concatenate([list_co1, temp1], 1)
  list_co2 = np.concatenate([list_co2, temp2], 1)
  #RANSAC算法，这里设置为1000轮，为了避免小概率事件的发生
  for i in range(num_of_rounds):
    #随机取点进行 affine transformation的求解
    rand_samp = np.random.randint(len(list_co1), size=[6])
    #用最小二乘法拟合数据得到一个线性方程组的解
    aff = np.linalg.lstsq(list_co1[rand_samp], list_co2[rand_samp], rcond=1e-6)[0]
    #判断那些是inlier
    inli = np.linalg.norm(list_co1.dot(aff) - list_co2, axis=1) < thre_inlier
    sum_in = inli.sum()
    if sum_in > inli_thre:
      inli_thre = inli.sum()
      xform = np.linalg.lstsq(list_co1[inli], list_co2[inli], rcond=1e-6)[0]
      outlier_labels =~inli
  return xform, outlier_labels


def stitch_images(image1, image2, xform):
  """Stitch two matched images given the transformation between them.

  Args:
  - image1 (3D uint8 array): A color image.
  - image2 (3D uint8 array): A color image.
  - xform (2D float64 array): A 3x3 matrix representing the transformation
      between image1 and image2. This transformation should map coordinates
      in image1 to the corresponding coordinates in image2.

  Returns:
  - image_stitched (3D uint8 array)
  """
  #先pad
  temp = 1e-3
  pad_1 = np.pad(image1, [(150, 150), (150, 150), (0, 0)])
  pad_2 = np.pad(image2, [(150, 150), (150, 150), (0, 0)])
  #调用cv2.warpAffine将image1进行变换
  warp_img = cv2.warpAffine(pad_1, xform.T[:2], pad_2.shape[:2][::-1])
  stitched_img = pad_2.astype(np.float32) + warp_img.astype(np.float32)
  stitched_img[(pad_2 > temp) & (warp_img > temp)] /= 2
  return stitched_img.astype(np.uint8)


def main():
  #在此处改变文件路径即可换成其他图像的匹配
  img_path1 = 'data/bikes1.png'
  img_path2 = 'data/bikes3.png'
  img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
  img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
  gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255.0
  gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255.0
  blobs1 = detect_blobs(gray1)
  blobs2 = detect_blobs(gray2)
  descriptor1 = compute_descriptors(gray1, *blobs1)
  descriptor2 = compute_descriptors(gray2, *blobs2)
  match = match_descriptors(descriptor1, descriptor2)
  trans, outlier = compute_affine_xform(blobs1[0], blobs2[0], match)
  print(trans)
  cv2.imwrite("match111.png", draw_matches(img1, img2, blobs1[0], blobs2[0], match, outlier))
  cv2.imwrite("sti_img111.png", stitch_images(img1, img2, trans))
if __name__ == '__main__':
  main()


