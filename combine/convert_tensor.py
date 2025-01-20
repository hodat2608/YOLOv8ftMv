# For appropriation, here is ready-made code, hoping it will be helpful to you
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
ultralytics_main_dir = current_dir
sys.path.append(str(ultralytics_main_dir))
from ultralytics import YOLO
import time
import torch
import math
import cv2
import numpy as np
import ultralytics

from ultralytics.utils import ops

def tensor_to_numpy(tensor):
    
    img=cv2.imread(r"C:\ultralytics-main\2024-03-05_00-01-31-398585-C1.jpg")

    #4 points
    x1y1=[550.0,400.0]
    x2y2=[450.0,400.0]
    x3y3=[450.0,600.0]
    x4y4=[550.0,600.0]

    # Example bbox with rotation at 45 degrees (converted to radians if needed)
    r1=0 * np.pi / 180
    r2=-45 * np.pi / 180
    bbox = [500, 500, 200, 100, r1] # Convert degrees to radians
    xyxyxyxy =ops.xywhr2xyxyxyxy(torch.tensor(bbox))
    xywhr=ops.xyxyxyxy2xywhr(xyxyxyxy.reshape(1,-1,2))

    rotate=xywhr[:,4]*180/np.pi
    #clockwise
    xywhr1=ops.xyxyxyxy2xywhr(torch.tensor([[x1y1,x2y2,x3y3,x4y4]]))
    xywhr2=ops.xyxyxyxy2xywhr(torch.tensor([[x2y2,x3y3,x4y4,x1y1]]))
    xywhr3=ops.xyxyxyxy2xywhr(torch.tensor([[x3y3,x4y4,x1y1,x2y2]]))
    xywhr4=ops.xyxyxyxy2xywhr(torch.tensor([[x4y4,x1y1,x2y2,x3y3]]))

    #anticlockwise
    xywhr5=ops.xyxyxyxy2xywhr(torch.tensor([[x1y1,x4y4,x3y3,x2y2]]))
    xywhr6=ops.xyxyxyxy2xywhr(torch.tensor([[x4y4,x3y3,x2y2,x1y1]]))
    xywhr7=ops.xyxyxyxy2xywhr(torch.tensor([[x3y3,x2y2,x1y1,x4y4]]))
    xywhr8=ops.xyxyxyxy2xywhr(torch.tensor([[x2y2,x1y1,x4y4,x3y3]]))

    r=[xywhr1[:,4],xywhr2[:,4],xywhr3[:,4],xywhr4[:,4],xywhr5[:,4],xywhr6[:,4],xywhr7[:,4],xywhr8[:,4]]

    #xyxyxyxy=ops.xywhr2xyxyxyxy(torch.tensor([161,152,100,170,math.radians(0)]))

    data_array=xyxyxyxy.to(torch.int32).cpu().numpy()
    cv2.drawContours(img, [data_array], -1, (255, 0, 0), 2)

    x=data_array[0][0]
    y=data_array[0][1]
    x1y1=[data_array[0][0], data_array[0][1]]
    x2y2=[data_array[1][0], data_array[1][1]]
    x3y3=[data_array[2][0], data_array[2][1]]
    x4y4=[data_array[3][0], data_array[3][1]]

    xywhr=ops.xyxyxyxy2xywhr(torch.tensor([[x1y1, x2y2,x3y3, x4y4]]).to(torch.float32))

    cv2.line(img, (x - 10, y), (x + 10, y), (0, 255, 0), 2)
    cv2.line(img, (x, y - 10), (x, y + 10), (0, 255, 0), 2)

    cv2.putText(img, "1", (data_array[0][0],data_array[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(img, "2", (data_array[1][0],data_array[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(img, "3", (data_array[2][0],data_array[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(img, "4", (data_array[3][0],data_array[3][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    cv2.imshow('Rectangle', img)
    cv2.waitKey(0)

    #[tensor([1.5708]), tensor([1.5708]), tensor([1.5708]), tensor([1.5708]), tensor([1.5708]), tensor([1.5708]), tensor([1.5708]), tensor([1.5708])]

    sd2=ops.xywhr2xyxyxyxy(torch.tensor([500,500,200,100,math.radians(-90)]))

    sd3=ops.xywhr2xyxyxyxy(torch.tensor([500,500,200,100,math.radians(180)]))
    sd4=ops.xywhr2xyxyxyxy(torch.tensor([500,500,200,100,math.radians(270)]))
    sd5=ops.xywhr2xyxyxyxy(torch.tensor([500,500,200,100,math.radians(360)]))

    sd6=ops.xywhr2xyxyxyxy(torch.tensor([500,500,200,100,math.radians(45)]))

    radians=math.radians(-90)

    test1=ops.xyxyxyxy2xywhr(torch.tensor([[100, 100, 100, 461.999104, 911.0016, 272.999424, 1066.99776, 527.0016]]))
    test2=ops.xyxyxyxy2xywhr(torch.tensor([[ 640.999424, 461.999104, 911.0016, 272.999424, 1066.99776, 527.0016,797.999104, 706.999296]]))
    test3=ops.xyxyxyxy2xywhr(torch.tensor([[ 461.999104, 911.0016, 272.999424, 1066.99776, 527.0016,797.999104, 706.999296, 640.999424]]))
    test4=ops.xyxyxyxy2xywhr(torch.tensor([[706.999296, 640.999424,461.999104, 911.0016, 272.999424, 1066.99776, 527.0016,797.999104 ]]))



def transpose_matrix_torch(
        class_id: int = None,
        x_center: int | float = 0,
        y_center: int | float = 0,
        width: int | float = 0,
        height: int | float = 0,
        angle: int | float = 0,
        im_height: int = 0,
        im_width: int = 0,
    ):
        half_width = width / 2
        half_height = height / 2

        angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float64))

        rotation_matrix = torch.tensor(
            [
                [torch.cos(angle_rad), -torch.sin(angle_rad)],
                [torch.sin(angle_rad),  torch.cos(angle_rad)],
            ],
            dtype=torch.float64,
        )
        corners = torch.tensor(
            [
                [-half_width, -half_height],
                [ half_width, -half_height],
                [ half_width,  half_height],
                [-half_width,  half_height],
            ],
            dtype=torch.float64,
        )
        rotated_corners = torch.matmul(corners, rotation_matrix)
        final_corners = rotated_corners + torch.tensor([x_center, y_center], dtype=torch.float64)
        normalized_corners = final_corners / torch.tensor([im_width, im_height], dtype=torch.float64)
        return [int(class_id)] + normalized_corners.flatten().tolist()

#     # Ví dụ sử dụng
# class_id = 1
# x_center, y_center = 88.0, 102.0
# width, height = 141.785406, 14.314282
# angle = 89.065522
# im_height, im_width = 1050, 1400

# result = transpose_matrix_torch(
#     class_id=class_id,
#     x_center=x_center,
#     y_center=y_center,
#     width=width,
#     height=height,
#     angle=angle,
#     im_height=im_height,
#     im_width=im_width,
# )

# print("Kết quả:", result)

def yolo_obb_to_corners(
    x: float, y: float, w: float, h: float, degree: float,
    image_width: int, image_height: int
):
    # Chuyển tọa độ YOLO OBB từ chuẩn hóa sang pixel
    x_pixel = x * image_width
    y_pixel = y * image_height
    w_pixel = w * image_width
    h_pixel = h * image_height

    # Chuyển độ sang radian
    radian = torch.deg2rad(torch.tensor(degree, dtype=torch.float64))

    # Tính toán ma trận xoay
    cos_theta = torch.cos(radian)
    sin_theta = torch.sin(radian)

    # Tạo vector bán kính
    dx = w_pixel / 2
    dy = h_pixel / 2

    # Các góc ban đầu
    corners = torch.tensor([
        [-dx, -dy],
        [ dx, -dy],
        [ dx,  dy],
        [-dx,  dy],
    ], dtype=torch.float64)

    # Ma trận xoay
    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ], dtype=torch.float64)

    # Tính các đỉnh sau khi xoay và dịch về tâm
    rotated_corners = corners @ rotation_matrix.T + torch.tensor([x_pixel, y_pixel], dtype=torch.float64)

    return rotated_corners, radian

# Ví dụ sử dụng hàm
rotated_corners, radian = yolo_obb_to_corners(
    x=88.0 / 1400, y=102.0 / 1050, w=141.785406 / 1400, h=14.314282 / 1050,
    degree=89.065522, image_width=1400, image_height=1050
)

print("Tọa độ các đỉnh sau chuyển đổi:", rotated_corners)
print("Góc (radian):", radian)

# 0.05691973012696166 0.16453956996149205 0.05857142830637418 0.02952380951383518 0.06879455558732406 0.029746144324222238 0.06714285740791154 0.1647619047718791