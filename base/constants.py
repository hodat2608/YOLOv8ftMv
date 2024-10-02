import argparse
import numpy as np
import torch
import cv2,os,shutil

DATA_C = """
1 88.000000 101.000000 141.785406 14.314282 89.065522
2 218.000000 102.500000 140.785539 14.297973 89.065522
3 347.112440 102.401162 140.959435 14.303368 89.401449
4 477.548407 102.450004 140.874897 13.174511 -89.889658
5 607.206324 104.474281 140.810931 13.887130 89.400333
6 736.263139 103.401610 140.957455 13.999232 89.400333
7 864.673210 103.993445 139.999925 12.798633 89.940639
8 994.426958 106.000442 141.985419 15.001028 89.940639
9 1120.268580 105.044902 141.818407 16.071710 -88.947033
10 1251.219656 106.542994 140.860909 14.039567 -89.798347
11 148.000000 386.000000 141.785406 14.314282 89.065522
12 280.343811 384.883274 142.030041 14.998005 89.065522
13 410.389410 383.452512 142.891648 14.907583 89.948372
14 540.889410 383.952512 143.892549 13.908485 89.948372
15 670.889410 387.952512 143.892549 13.908485 89.948372
16 800.435171 386.446245 144.904263 12.999094 89.948372
17 929.889410 386.952512 143.892549 13.908485 89.948372
18 1059.435171 386.946245 143.904263 12.999995 89.948372
19 1186.935171 386.946245 143.905164 13.999994 89.948372
20 1315.435171 387.946245 143.904263 12.999995 89.948372
30 1252.935171 667.946245 143.905164 13.999994 89.948372
29 1123.435165 667.952553 143.891648 14.999994 89.948372
28 994.935165 668.952102 143.891648 13.999994 89.948372
27 864.935165 666.952102 143.891648 13.999994 89.948372
26 734.935165 668.952102 143.891648 13.999994 89.948372
25 605.935165 667.952102 143.891648 13.999994 89.948372
24 473.935165 667.952102 143.891648 13.999994 89.948372
23 344.499994 668.506307 145.000843 14.869337 89.948372
22 212.935165 668.952102 143.891648 13.999994 89.948372
21 81.441021 668.886945 144.009232 15.009902 89.160614
"""

def str2cache(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 'ram', '1'):
        return True
    elif v.lower() in ('false', 'disk i/o', '0'):
        return False
    elif v.lower() == 'disk':
        return 'disk'
    else:
        raise argparse.ArgumentTypeError('Expected True, "disk", or False for --cache')
    
def cache_option(value):
    if value == 'RAM':
        return True
    elif value == 'Disk I/O':
        return False
    elif value == 'HDD':
        return 'disk'
    else:
        raise ValueError('Invalid cache option.')
    
def xywhr2xyxyxyxy_original_ops(class_id,x,img_width,img_height):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in degrees from 0 to 90.

    Args:
        x (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2

    corners = torch.stack([pt1, pt2, pt3, pt4], dim=-2)
    corners_normalized = corners.clone()
    corners_normalized[..., 0] = corners[..., 0] / img_width
    corners_normalized[..., 1] = corners[..., 1] / img_height 

    return [int(class_id)] + corners_normalized.view(-1).tolist()

def get_params_xywhr2xyxyxyxy_original_ops(des_path,progress_label):
    input_folder = des_path
    os.makedirs(os.path.join(input_folder,'instance'),exist_ok=True)
    output_folder = (os.path.join(input_folder,'instance'))
    total_fl = len(des_path) 
    for index,txt_file in enumerate(os.listdir(input_folder)):
        if txt_file.endswith('.txt'):
            if txt_file == 'classes.txt':
                continue
            input_path = os.path.join(input_folder, txt_file)
            im = cv2.imread(input_path[:-4]+'.jpg')
            im_height, im_width, _ = im.shape
            output_path = os.path.join(output_folder, txt_file)
            with open(input_path, 'r') as file:
                lines = file.readlines()
            with open(output_path, 'w') as out_file:
                for line in lines:
                    line = line.strip()
                    if "YOLO_OBB" in line:
                        continue
                    params = list(map(float, line.split()))
                    class_id = params[0]
                    bbox_list = params[1:]
                    bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)
                    bbox_tensor[-1] = torch.abs(bbox_tensor[-1]) if torch.sign(bbox_tensor[-1])==-1 else 180-bbox_tensor[-1]
                    bbox_tensor_2d = bbox_tensor.unsqueeze(0)
                    converted_label = xywhr2xyxyxyxy_original_ops(class_id,bbox_tensor_2d,im_width,im_height)
                    out_file.write(" ".join(map(str, converted_label)) + '\n')
            progress_retail = (index + 1) / total_fl * 100
            progress_label.config(text=f"Converting YOLO OBB Dataset Format to DOTA Format: {progress_retail:.2f}%")
            progress_label.update_idletasks()
            os.replace(output_path, input_path)
    shutil.rmtree(output_folder)

def xyxyxyxy_to_xywhr_low_tolerance(class_id,x1, y1, x2, y2, x3, y3, x4, y4, img_width, img_height):
    '''
    - dung sai góc alpha giữa 2 lần convert thấp (~ 0.07 độ)
    - tọa độ tâm không đổi
    - tuy nhiên kích thước width & height sẽ có sự chênh lệch nhỏ
    '''
    x_center_norm = (x1 + x2 + x3 + x4) / 4
    y_center_norm = (y1 + y2 + y3 + y4) / 4
    width_norm = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    height_norm = np.sqrt((x4 - x1)**2 + (y4 - y1)**2)
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = -(np.degrees(angle_rad))
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    width = width_norm * img_height
    height = height_norm * img_width
    return class_id, x_center, y_center, width, height, angle_deg

def xyxyxyxy_to_xywhr_high_tolerance(class_id, x1, y1, x2, y2, x3, y3, x4, y4, img_width, img_height):
    '''
    - dung sai góc alpha giữa 2 lần convert cao (~ 0.7 độ)
    - tọa độ tâm không đổi
    - tuy nhiên kích thước width & height không có sự chênh lệch
    '''
    points = np.array([
        [x1 * img_width, y1 * img_height],
        [x2 * img_width, y2 * img_height],
        [x3 * img_width, y3 * img_height],
        [x4 * img_width, y4 * img_height]
    ])
    center = points.mean(axis=0)
    width = np.linalg.norm(points[1] - points[0])
    height = np.linalg.norm(points[3] - points[0]) 
    angle_deg = (np.degrees(np.arctan2(points[1][1] - points[0][1], points[1][0] - points[0][0])) % 180)
    return class_id, center[0],center[1], width, height, angle_deg
    
def xyxyxyxy2xywhr_direct(self,des_path,progress_label):
    input_folder = des_path
    os.makedirs(os.path.join(input_folder,'instance'),exist_ok=True)
    output_folder = (os.path.join(input_folder,'instance'))
    total_fl = len(des_path) 
    for index,txt_file in enumerate(os.listdir(input_folder)):
        if txt_file.endswith('.txt'):
            if txt_file == 'classes.txt':
                continue
            input_path = os.path.join(input_folder, txt_file)
            im_height, im_width, _ = im.shape
            im = cv2.imread(input_path[:-4]+'.jpg')
            output_path = os.path.join(output_folder, txt_file)
            with open(input_path, 'r') as file:
                lines = file.readlines()
            with open(output_path, 'w') as out_file:
                out_file.write('YOLO_OBB\n')
                for line in lines:
                    line = line.strip()
                    params = list(map(float, line.split()))
                    class_id,x1, y1, x2, y2, x3, y3, x4, y4 = params
                    class_id, x_center, y_center, width, height, angle_deg = xyxyxyxy_to_xywhr_low_tolerance(class_id,x1, y1, x2, y2, x3, y3, x4, y4,im_height,im_width)
                    formatted_values = ["{:.6f}".format(value) for value in [x_center, y_center, width, height, angle_deg]]
                    output_line = "{} {}\n".format(str(int(class_id)), ' '.join(formatted_values))
                    out_file.write(output_line)
            progress_retail = (index + 1) / total_fl * 100
            progress_label.config(text=f"Converting DOTA Format Format to YOLO 0BB Format: {progress_retail:.2f}%")
            progress_label.update_idletasks()
            os.replace(output_path, input_path)
    shutil.rmtree(output_folder)

def tracking_id(a,obj_x,obj_y):
    tolerance = 50
    for id, (x, y) in a.items():
        if (x - tolerance <= obj_x <= x + tolerance) and (y - tolerance <= obj_y <= y + tolerance):
            return id,obj_x,obj_y,x,y
        
def check_x_y(id,obj_x,obj_y,x,y):
    tolerance = 5
    if obj_x < x-tolerance or obj_x > x+tolerance and obj_y < y-tolerance or obj_y>y+tolerance:
        return id,obj_x,obj_y,x,y,'NG',False
    else :
        return id,obj_x,obj_y,x,y,'OK',True
    
def track_conn():
    a = {}
    for line in DATA_C.strip().split('\n'):
        values = line.split()
        id = int(values[0])
        x_c = float(values[1])
        y_c = float(values[2])
        a[id] = [x_c, y_c]
    return a


# def check_x_y(id,obj_x,obj_y,x,y):
#     tolerance = 5
#     iter = {}
#     if obj_x < x-tolerance or obj_x > x+tolerance and obj_y < y-tolerance or obj_y>y+tolerance:
#         iter[id] = [obj_x,obj_y,x,y,'NG',False]
#     else :
#         iter[id] = [obj_x,obj_y,x,y,'OK',False]
#     # return iter
#     print(iter)