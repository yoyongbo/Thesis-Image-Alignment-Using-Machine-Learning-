#Image Translation
import os.path
import csv
import cv2
import numpy as np
import tqdm as tqdm
import os
import matplotlib.pyplot as plt


def write_to_datas(file_path, row_content):
    with open(file_path, 'a')as file:
        fileWrite = csv.writer(file)
        fileWrite.writerow(row_content)




header = ['Source_Image_PATH', 'Transform_Image_PATH', 'A11', 'A12', 'tx', 'A21', 'A22', 'ty']
rotate = [0, 90, 180, 270]

data_height = 240
data_width = 240
images_path = 'Images/'
#src_img_path = 'src1/src_cropped.png'
src_img_dir = 'center images'
transed_img_path ='transformed_images/'
transImg_csv_path = 'transformed.csv'
data_set_sheet = 'data_set.csv'
directions = [[0, 1], [0, -1], [1, 0], [-1, 0],
              [1, 1], [-1, -1], [1, -1], [-1, 1]]
distance = 20
steps = 6
images_list = [f for f in os.listdir(images_path) if not f.startswith('.')]

#img_dir_path = 'src1/transform images/'


#images_list = [f for f in os.listdir(images_path) if not f.startswith('.')]

write_to_datas(data_set_sheet, header)

def img_read(img_name):
    with open(transImg_csv_path) as file:
        csv_file = csv.reader(file, delimiter=',')
        for i in csv_file:
            if str(img_name) == ''.join(i):
                return True
    return False

def store_transform_images(img, down, right, rotate_angle, src_img_path, src_name):
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    h, w = data_height, data_width
    h = int(h)
    w = int(w)
    T = np.float32([[1, 0, right], [0, 1, down]])
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)

    M_extend= np.append(M, [[0,0,1]], axis=0)
    T_extend = np.float32([[1, 0, right], [0, 1, down], [0, 0, 1]])

    # translation matrix (affine matrix - the goal)
    matrix_mat = np.matmul(M_extend, T_extend)
    img_r_t = cv2.warpAffine(img, matrix_mat[:2], (width, height))
    img_r_t_crop = crop_center_img(img_r_t, data_height, data_width)

    img_name= ' down_'+str(down)+' right_'+str(right)+' roate_'+str(rotate_angle)+src_name+'.png'
    trans_img_path = transed_img_path+'/'+img_name
    cv2.imwrite(trans_img_path, img_r_t_crop)
    row_content =[]
    row_content.append(src_img_path)
    row_content.append(trans_img_path)
    for i in matrix_mat[:2]:
        for j in i:
            row_content.append(j)
    write_to_datas(data_set_sheet, row_content)

def src_img(img, h, w,img_name):
    data_height = h
    data_width = w
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    starting_y = center[1] - (h / 2)
    starting_x = center[0]- (w / 2)
    starting_y = int(starting_y)
    starting_x = int(starting_x)
    h = int(h)
    w = int(w)
    img_crop = img[starting_y: starting_y + h, starting_x: starting_x + w]
    src_path = src_img_dir+'/'+img_name+'.png'
    cv2.imwrite(src_path, img_crop)
    return src_path


def crop_center_img(img, h, w):
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    starting_y = center[1] - (h / 2)
    starting_x = center[0] - (w / 2)
    starting_y = int(starting_y)
    starting_x = int(starting_x)
    h = int(h)
    w = int(w)
    img_crop = img[starting_y: starting_y + h, starting_x: starting_x + w]
    return img_crop


for i in images_list:
    img = cv2.imread(images_path+'/'+i)
    if not (img_read(str(i))):
        write_to_datas(transImg_csv_path, [str(i)])
        src_current_path = src_img(img, data_height, data_width, str(i[: -4]))
        distance = 20
        store_transform_images(img, 0, 0, 90, src_current_path, str(i[: -4]))
        store_transform_images(img, 0, 0, 180, src_current_path, str(i[: -4]))
        store_transform_images(img, 0, 0, 270, src_current_path, str(i[: -4]))
        for step in range(steps):
            for dir in directions:
                vertical_trans = distance * dir[0]
                horizontal_trans = distance * dir[1]
                for angle in rotate:
                    store_transform_images(img, vertical_trans, horizontal_trans, angle, src_current_path, str(i[:-4]))
            distance += 20













