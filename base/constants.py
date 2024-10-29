import argparse
import numpy as np
import torch
import cv2,os,shutil
import tkinter as tk
from tkinter import ttk
from base.extention import *
import torch
import cv2
import numpy as np
# import tensorrt as trt
import random
import ctypes
# import pycuda.driver as cuda
import time
import pandas as pd
# import pycuda.autoinit

class CFG_Table(tk.Tk):
    def __init__(self, frame_n):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"), anchor="center")
        style.configure("Treeview", rowheight=30, font=("Arial", 10), borderwidth=1)
        style.map("Treeview", background=[("selected", "#ececec")])
        style.configure("Treeview", background="white", foreground="black")
        style.configure("Treeview.Row", background="#f2f2f2", foreground="black")
        style.map("Treeview", background=[("selected", "#ececec")])
        style.configure("ng_row", foreground="red")
        self.tree = ttk.Treeview(frame_n, columns=("No.", "Default_X", "Default_Y", "Result_X", "Result_Y", 'Angle', "Status"), show="headings", height=5)
        self.tree.heading("No.", text="No.", anchor="center")
        self.tree.heading("Default_X", text="Default_X", anchor="center") 
        self.tree.heading("Default_Y", text="Default_Y", anchor="center") 
        self.tree.heading("Result_X", text="Result_X", anchor="center")  
        self.tree.heading("Result_Y", text="Result_Y", anchor="center") 
        self.tree.heading("Angle", text="Angle", anchor="center")
        self.tree.heading("Status", text="Status", anchor="center")
        self.tree.column("No.", width=50, anchor="center")
        self.tree.column("Default_X", width=170, anchor="center")  
        self.tree.column("Default_Y", width=170, anchor="center") 
        self.tree.column("Result_X", width=170, anchor="center")  
        self.tree.column("Result_Y", width=170, anchor="center")  
        self.tree.column("Angle", width=100, anchor="center")
        self.tree.column("Status", width=80, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)

    def __call__(self,data):
        if data:
            for row in self.tree.get_children():
                self.tree.delete(row)
        for row in data:
            No = row[0]
            Default_X = f"{round(row[3],1)}"
            Default_Y = f"{round(row[4],1)}"
            Detect_X = f"{round(row[1],1)}"
            Detect_Y = f"{round(row[2],1)}"
            Angle = f'{row[5]}°'
            State = row[6]
            if State == 'NG':
                self.tree.insert("", "end", values=(No, Default_X, Default_Y, Detect_X, Detect_Y, Angle, State), tags=('ng_row',))
            else:
                self.tree.insert("", "end", values=(No, Default_X, Default_Y, Detect_X, Detect_Y, Angle, State))
        self.tree.tag_configure('ng_row', foreground="red")

class setupTools():

    @staticmethod
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
    @staticmethod
    def cache_option(value):
        if value == 'RAM':
            return True
        elif value == 'Disk I/O':
            return False
        elif value == 'HDD':
            return 'disk'
        else:
            raise ValueError('Invalid cache option.')
    @staticmethod    
    def xywhr2xyxyxyxy_original_ops(self,class_id,x,img_width,img_height):
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
    
    @staticmethod
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
                        converted_label = setupTools.xywhr2xyxyxyxy_original_ops(class_id,bbox_tensor_2d,im_width,im_height)
                        out_file.write(" ".join(map(str, converted_label)) + '\n')
                progress_retail = (index + 1) / total_fl * 100
                progress_label.config(text=f"Converting YOLO OBB Dataset Format to DOTA Format: {progress_retail:.2f}%")
                progress_label.update_idletasks()
                os.replace(output_path, input_path)
        shutil.rmtree(output_folder)

    def xyxyxyxy_to_xywhr_low_tolerance(self,class_id,x1, y1, x2, y2, x3, y3, x4, y4, img_width, img_height):
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

    def xyxyxyxy_to_xywhr_high_tolerance(self,class_id, x1, y1, x2, y2, x3, y3, x4, y4, img_width, img_height):
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
                        class_id, x_center, y_center, width, height, angle_deg = self.xyxyxyxy_to_xywhr_low_tolerance(class_id,x1, y1, x2, y2, x3, y3, x4, y4,im_height,im_width)
                        formatted_values = ["{:.6f}".format(value) for value in [x_center, y_center, width, height, angle_deg]]
                        output_line = "{} {}\n".format(str(int(class_id)), ' '.join(formatted_values))
                        out_file.write(output_line)
                progress_retail = (index + 1) / total_fl * 100
                progress_label.config(text=f"Converting DOTA Format Format to YOLO 0BB Format: {progress_retail:.2f}%")
                progress_label.update_idletasks()
                os.replace(output_path, input_path)
        shutil.rmtree(output_folder)

    def xywh2xyxy_tensort(self, origin_h, origin_w, x):
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h
        return y
    
    def bbox_iou_tensort(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        return iou

    @staticmethod
    def tracking_id(a,obj_x,obj_y):
        for id,(x,y) in a.items():
            if (x -TRACK_ID<=obj_x<=x+TRACK_ID) and (y-TRACK_ID<=obj_y<=y+TRACK_ID):
                return id,obj_x,obj_y,x,y
            
    @staticmethod     
    def check_x_y(id,obj_x,obj_y,x,y):
        if obj_x<x-CAL_COOR or obj_x>x+CAL_COOR or obj_y<y-CAL_COOR or obj_y>y+CAL_COOR:
            return id,obj_x,obj_y,x,y,'NG',False
        else :
            return id,obj_x,obj_y,x,y,'OK',True
        
    @staticmethod   
    def track_conn():
        a = {}
        for line in DATA_C.strip().split('\n'):
            values = line.split()
            id = int(values[0])
            x_c = float(values[1])
            y_c = float(values[2])
            a[id] = [x_c, y_c]
        return a

"""
TensorRT
"""



# class YoloV5TRT():
#     EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
#     host_inputs  = []
#     cuda_inputs  = []
#     host_outputs = []
#     cuda_outputs = []
#     bindings = []
#     def __init__(self, library, engine, conf,categories):
#         self.CONF_THRESH = conf 
#         self.IOU_THRESHOLD = 0.4
#         self.LEN_ALL_RESULT = 38001
#         self.LEN_ONE_RESULT = 38
#         self.categories = categories

#         self.cfx = cuda.Device(0).make_context()
#         self.stream = cuda.Stream()

#         TRT_LOGGER = trt.Logger()
        
#         ctypes.CDLL(library)
        
#         with open(engine, 'rb') as f:
#             serialized_engine = f.read()

#         runtime = trt.Runtime(TRT_LOGGER)
        
#         self.engine = runtime.deserialize_cuda_engine(serialized_engine)
#         self.batch_size = self.engine.max_batch_size
#         self.context = self.engine.create_execution_context()
#         for binding in self.engine:
#             size = trt.volume(self.engine.get_binding_shape(binding)) * self.batch_size
#             dtype = trt.nptype(self.engine.get_binding_dtype(binding))
#             host_mem = cuda.pagelocked_empty(size, dtype)
#             cuda_mem = cuda.mem_alloc(host_mem.nbytes)

#             bindings.append(int(cuda_mem))
#             if self.engine.binding_is_input(binding):
#                 self.input_w = self.engine.get_binding_shape(binding)[-1]
#                 self.input_h = self.engine.get_binding_shape(binding)[-2]
#                 host_inputs.append(host_mem)
#                 cuda_inputs.append(cuda_mem)
#             else:
#                 host_outputs.append(host_mem)
#                 cuda_outputs.append(cuda_mem)

#     def PreProcessImg(self, img):
#         image_raw = img
#         h, w, c = image_raw.shape
#         image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         r_w = self.input_w / w
#         r_h = self.input_h / h
#         if r_h > r_w:
#             tw = self.input_w
#             th = int(r_w * h)
#             tx1 = tx2 = 0
#             ty1 = int((self.input_h - th) / 2)
#             ty2 = self.input_h - th - ty1
#         else:
#             tw = int(r_h * w)
#             th = self.input_h
#             tx1 = int((self.input_w - tw) / 2)
#             tx2 = self.input_w - tw - tx1
#             ty1 = ty2 = 0
#         image = cv2.resize(image, (tw, th))
#         image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128))
#         image = image.astype(np.float32)
#         image /= 255.0
#         image = np.transpose(image, [2, 0, 1])
#         image = np.expand_dims(image, axis=0)
#         image = np.ascontiguousarray(image)
#         return image, image_raw, h, w

#     def Inference(self, img):
#         self.cfx.push()
#         input_image, image_raw, origin_h, origin_w = self.PreProcessImg(img)
#         np.copyto(host_inputs[0], input_image.ravel())
        
#         # self.context = self.engine.create_execution_context()
#         cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], self.stream)
        
#         t1 = time.time()
#         self.context.execute_async(self.batch_size, bindings, stream_handle=self.stream.handle)
#         cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], self.stream)
#         self.stream.synchronize()
#         t2 = time.time()
#         output = host_outputs[0]
                
#         for i in range(self.batch_size):
#             result_boxes, result_scores, result_classid = self.PostProcess(output[i * self.LEN_ALL_RESULT: (i + 1) * self.LEN_ALL_RESULT], origin_h, origin_w)
#         self.cfx.pop()
#         det_res = []
#         for j in range(len(result_boxes)):
#             box = result_boxes[j]
#             det = dict()
#             det["class"] = self.categories[int(result_classid[j])]
#             det["conf"] = result_scores[j]
#             det["box"] = box 
#             det_res.append(det)
#             # self.PlotBbox(box, img, label="{}:{:.2f}".format(self.categories[int(result_classid[j])], result_scores[j]),)
#         return self.convert_preds(det_res), t2-t1

#     def __del__(self):

#         del self.engine
#         del self.context

#     def convert_preds(self,preds):
#         arr = []
#         cols = ['xmin','ymin','xmax','ymax','confidence','class','name']

#         for dic in preds :
#             values = list(dic.values())
#             values[2] = list(values[2])
            
#             arr.append([values[2][0],values[2][1],values[2][2],values[2][3],\
#                     values[1],self.categories.index(values[0]),values[0]])

#         df = pd.DataFrame(arr,columns = cols)
#         return df

#     def PostProcess(self, output, origin_h, origin_w):
#         num = int(output[0])
#         pred = np.reshape(output[1:], (-1, self.LEN_ONE_RESULT))[:num, :]
#         pred = pred[:, :6]
#         # Do nms
#         boxes = self.NonMaxSuppression(pred, origin_h, origin_w, conf_thres=self.CONF_THRESH, nms_thres=self.IOU_THRESHOLD)
#         result_boxes = boxes[:, :4] if len(boxes) else np.array([])
#         result_scores = boxes[:, 4] if len(boxes) else np.array([])
#         result_classid = boxes[:, 5] if len(boxes) else np.array([])
#         return result_boxes, result_scores, result_classid
    
#     def NonMaxSuppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
#         boxes = prediction[prediction[:, 4] >= conf_thres]
#         boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
#         boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
#         boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
#         boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
#         boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
#         confs = boxes[:, 4]
#         boxes = boxes[np.argsort(-confs)]
#         keep_boxes = []
#         while boxes.shape[0]:
#             large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
#             label_match = boxes[0, -1] == boxes[:, -1]
#             # Indices of boxes with lower confidence scores, large IOUs and matching labels
#             invalid = large_overlap & label_match
#             keep_boxes += [boxes[0]]
#             boxes = boxes[~invalid]
#         boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
#         return boxes
    
#     def xywh2xyxy(self, origin_h, origin_w, x):
#         y = np.zeros_like(x)
#         r_w = self.input_w / origin_w
#         r_h = self.input_h / origin_h
#         if r_h > r_w:
#             y[:, 0] = x[:, 0] - x[:, 2] / 2
#             y[:, 2] = x[:, 0] + x[:, 2] / 2
#             y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
#             y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
#             y /= r_w
#         else:
#             y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
#             y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
#             y[:, 1] = x[:, 1] - x[:, 3] / 2
#             y[:, 3] = x[:, 1] + x[:, 3] / 2
#             y /= r_h
#         return y
    
#     def bbox_iou(self, box1, box2, x1y1x2y2=True):
#         if not x1y1x2y2:
#             # Transform from center and width to exact coordinates
#             b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
#             b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
#             b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
#             b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
#         else:
#             # Get the coordinates of bounding boxes
#             b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
#             b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

#         inter_rect_x1 = np.maximum(b1_x1, b2_x1)
#         inter_rect_y1 = np.maximum(b1_y1, b2_y1)
#         inter_rect_x2 = np.minimum(b1_x2, b2_x2)
#         inter_rect_y2 = np.minimum(b1_y2, b2_y2)
#         inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
#                      np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
#         b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
#         b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

#         iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

#         return iou
    
#     def PlotBbox(self, x, img, color=None, label=None, line_thickness=None):
#         tl = (line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)  # line/font thickness
#         color = color or [random.randint(0, 255) for _ in range(3)]
#         c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
#         cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#         if label:
#             tf = max(tl - 1, 1)  # font thickness
#             t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
#             c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
#             cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
#             cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA,)


# class KiemGuong:
#     def __init__(self,MIRROR_DATA,radius_ref,delta,pixel_value):
#         self.MIRROR_DATA = MIRROR_DATA
#         self.radius_ref =  radius_ref
#         self.delta = delta
#         self.pixel_value = pixel_value

#     def inSideCircle(self,point,center,radius):
#         distance = (point[0] - center[0]) ** 2 + (point[1] - center[1])**2 - radius**2
#         return distance < 0

#     def measure_light_rectangle(self,img):
#         """
#         Measure light by calculate mean all pixel value
#         """
#         gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         average_light_value = gray_image.mean()
#         return average_light_value

#     def measure_light_circle(self,img,center,radius):
#         # cv2.imshow("IMG",img)
#         gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         roi = np.zeros(gray_image.shape[:2], np.uint8)
#         roi = cv2.circle(roi, center, radius, 255, cv2.FILLED)
#         mask = np.ones_like(gray_image) * 255
#         mask = cv2.bitwise_and(mask, gray_image, mask=roi) + cv2.bitwise_and(mask, mask, mask=~roi)
#         average_light = mask[np.where(mask<255)]
        
#         average_light_value = gray_image.mean()
#         # cv2.waitKey(0)
#         return average_light_value

#     def convert_to_binary(self,img, threshold):
#         _, binary_image = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
#         return binary_image

#     def shift_image(self,img, shift_x, shift_y):
#         # Get the height and width of the image
#         height, width = img.shape[:2]

#         # Define the transformation matrix
#         translation_matrix = np.float32([[1, 0, shift_x],
#                                         [0, 1, shift_y]])

#         # Apply the translation using warpAffine function
#         shifted_image = cv2.warpAffine(img, translation_matrix, (width, height))
#         cv2.imwrite(f"D:\\AI\\LOC\\KIEMGUONG\\images\\train\\{shift_x}_{shift_y}.jpg",shifted_image,[cv2.IMWRITE_JPEG_QUALITY, 100])
#         return shifted_image

#     def location(self,img):
#         # 1. pixel 0.015 mm
#         a,b,r = 0,0,0
#         try :
            
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             height,width = gray.shape
#             # Lọc Gaussian để làm mờ hình ảnh
#             blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#             binary_image = ~self.convert_to_binary(blurred,170)

#             detected_circles = cv2.HoughCircles(binary_image, 
#                             cv2.HOUGH_GRADIENT, 0.3, 1, param1 = 1,
#                         param2 = 50, minRadius = 300, maxRadius = 350)

#             cv2.line(self.img_draw,(width//2,0),(width//2,height),(0, 255, 0), 3)
#             cv2.line(self.img_draw,(0,height//2),(width,height//2),(0, 255, 0), 3)
#             print(detected_circles)
#             if detected_circles is not None:
#                 # Convert the circle parameters a, b and r to integers.
#                 detected_circles = np.uint16(np.around(detected_circles))
            
#                 for pt in detected_circles[0, :]:
#                     a, b, r = pt[0], pt[1], pt[2]
                
#                     # Draw the circumference of the circle.
#                     # cv2.circle(self.img_draw, (a, b), r, (0, 0, 255), 2)
#                     # cv2.circle(self.img_draw, (a, b), 5, (0, 0, 255), -1)
#                     # cv2.putText(self.img_draw,"Radius: " + str(round(r*0.015,2)) + "mm",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#                     # cv2.putText(self.img_draw,"Centroid: (" + str(a) + "," +str(b)+")",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#                     # Draw a small circle (of radius 1) to show the center.
                    
#                     break
            
#         except Exception as error :
#             print("Program Error - Location Test : ",error)
        
#         return a,b,r

        

#     def blur(self,img,blur_amount):

#         blurred_image = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
#         return blurred_image   

#     def shapenessLaplacian(self,img):
#         # pass
#         # [457,245,1200,957]
#         # Load the image
#         gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         # Apply the Laplacian operator
#         laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
#         # Calculate the variance of the Laplacian
#         sharpness = np.var(laplacian)
#         # print(sharpness)
#         return sharpness

#     def sharpnessTenengrad(self,img):
#         """
#         Frequency-based sharpness metrics
#         """
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Calculate the gradient using Sobel operator
#         gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#         gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

#         # Calculate the gradient magnitude squared
#         gradient_magnitude_squared = gradient_x**2 + gradient_y**2

#         # Calculate the sharpness as the sum of gradient magnitude squared
#         sharpness = np.sum(gradient_magnitude_squared)

#         return sharpness

#     def frequency_based_sharpness(self,img):
#         # Chuyển đổi hình ảnh sang không gian tần số
#         image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         image_fft = np.fft.fft2(image_gray)
#         image_fft_shifted = np.fft.fftshift(image_fft)

#         # Tính toán độ sắc nét từ thành phần tần số
#         magnitude_spectrum = np.log(1 + np.abs(image_fft_shifted))
#         sharpness_score = np.mean(magnitude_spectrum)

#         return sharpness_score

#     def select_ROI(self,img,radius,delta):
        
#         height,width,_ = img.shape
#         img_crop = img.copy()

#         try :
#             # ROI 1
#             x1_min,y1_min,x1_max,y1_max = width//2 - delta,height//2 - (radius+delta),width//2 + delta,height//2 - (radius-delta)
#             img_crop_1 = img_crop[y1_min : y1_max, x1_min:x1_max]

#             # ROI 2
#             x2_min,y2_min,x2_max,y2_max =  width//2 + (radius-delta),height//2 - delta,width//2 + (radius+delta),height//2 +delta
#             img_crop_2 = img_crop[y2_min : y2_max, x2_min:x2_max]

#             # ROI 3
#             x3_min,y3_min,x3_max,y3_max = width//2 - delta,height//2 + (radius-delta) ,width//2 + delta,height//2 + (radius+delta)
#             img_crop_3 = img_crop[y3_min : y3_max, x3_min:x3_max]

#             # ROI 4
#             x4_min,y4_min,x4_max,y4_max = width//2 - (radius+delta),height//2 - delta ,width//2 - (radius-delta),height//2 +delta
#             img_crop_4 = img_crop[y4_min : y4_max, x4_min:x4_max]

#             # Draw ROI
#             cv2.putText(self.img_draw,"ROI 1",(x1_min,y1_min-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#             cv2.rectangle(self.img_draw, (x1_min,y1_min), (x1_max,y1_max), (0,255,0), 2) 
#             cv2.putText(self.img_draw,"ROI 2",(x2_min,y2_min-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#             cv2.rectangle(self.img_draw, (x2_min,y2_min), (x2_max,y2_max), (0,255,0), 2) 
#             cv2.putText(self.img_draw,"ROI 3",(x3_min,y3_min-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#             cv2.rectangle(self.img_draw, (x3_min,y3_min), (x3_max,y3_max), (0,255,0), 2) 
#             cv2.putText(self.img_draw,"ROI 4",(x4_min,y4_min-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#             cv2.rectangle(self.img_draw, (x4_min,y4_min), (x4_max,y4_max), (0,255,0), 2) 

#             return img_crop_1, img_crop_2, img_crop_3, img_crop_4

#         except Exception as error : 
#             print('Program Error - Crop Image:',error)

#     def scan_edge_position(self,img,roi = 1):
#         try :
#             img_check = img.copy()
#             edge_position1  = []
#             edge_position2  = []
#             h,w,_ = img_check.shape
#             gray =  cv2.cvtColor(img_check, cv2.COLOR_BGR2GRAY)
#             _, binary =  cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)

#             if roi == 1 :
#                 x1 = w//4
#                 x2 = w - w//4
#                 # cv2.line(binary,(x1,0),(x1,h),(0, 255, 0), 1)
#                 # cv2.line(binary,(w - x1,0),(w - x1,h),(0, 255, 0), 1)
#                 for y in range(h):
#                     if binary[y, x1] == 0: # pixel trắng
#                         # print(f"Pixel giao nhau ở vị trí ({x1}, {y})")
                          
#                         edge_position1.append((x1,y))   
                    
#                 for y in range(h):   
#                     if binary[y, x2] == 0 :
#                         # print(f"Pixel giao nhau ở vị trí ({x2}, {y})")
#                         edge_position2.append((x2,y)) 

#                 # cv2.circle(img_check, edge_position1[0], 2, (0, 0, 255), -1)  
#                 # cv2.circle(img_check, edge_position2[0], 2, (0, 0, 255), -1)  
#             if roi == 2 :
#             # ROI 2
#                 y1 = h//4
#                 y2 = h - h//4
#                 # cv2.line(img_check,(0,y1),(w,y1),(0, 255, 0), 1)
#                 # cv2.line(img_check,(0,h - y1),(w,h - y1),(0, 255, 0), 1)
#                 for x in range(w-1,-1,-1):
#                     if binary[y1, x] == 0: # pixel trắng
#                         # print(f"Pixel giao nhau ở vị trí ({x}, {y1})")    
#                         edge_position1.append((x,y1))      
#                 for x in range(w-1,-1,-1):   
#                     if binary[y2, x] == 0 :
#                         # print(f"Pixel giao nhau ở vị trí ({x}, {y2})")
#                         edge_position2.append((x,y2))   

#                 # cv2.circle(img_check, edge_position1[0], 2, (0, 0, 255), -1)  
#                 # cv2.circle(img_check, edge_position2[0], 2, (0, 0, 255), -1) 
#             # ROI 3
#             if roi == 3 :
#                 x1 = w//4
#                 x2 = w - w//4
#                 # cv2.line(binary,(x1,0),(x1,h),(0, 255, 0), 1)
#                 # cv2.line(binary,(w - x1,0),(w - x1,h),(0, 255, 0), 1)

#                 for y in range(h-1,-1,-1):
#                     if binary[y, x1] == 0: # pixel trắng
#                         edge_position1.append((x1,y))   

#                 for y in range(h-1,-1,-1):   
#                     if binary[y, x2] == 0 :
#                         edge_position2.append((x2,y))   
#                 # cv2.circle(img_check, edge_position1[0], 2, (0, 0, 255), -1)  
#                 # cv2.circle(img_check, edge_position2[0], 2, (0, 0, 255), -1)                        
#             # ROI 4
#             if roi == 4 :
#                 y1 = h//4
#                 y2 = h - h//4
#                 # cv2.line(img_check,(0,y1),(w,y1),(0, 255, 0), 1)
#                 # cv2.line(img_check,(0,h - y1),(w,h -y1),(0, 255, 0), 1)
#                 for x in range(w):
#                     if binary[y1, x] == 0: # pixel trắng
#                         edge_position1.append((x,y1))           
#                 for x in range(w):   
#                     if binary[y2, x] == 0 :
#                         edge_position2.append((x,y2))  

#                 # cv2.circle(img_check, edge_position1[0], 2, (0, 0, 255), -1)  
#                 # cv2.circle(img_check, edge_position2[0], 2, (0, 0, 255), -1)  

#             return img_check,edge_position1,edge_position2
#         except Exception as error :
#             print('Program Error - Scan Edge Position:',error)
#             return img,edge_position1,edge_position2

#     def calculate_distance(self,x1, y1, x2, y2):
#         distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#         return distance

#     def detect_circle(self,img,radius,delta):
#         img_crop = img.copy()
#         height,width,_ = img.shape
#         centroid = (width//2,height//2)
#         radius_filter = []
#         try :
#             # ROI 1
#             x1_min,y1_min,x1_max,y1_max = width//2 - delta,height//2 - (radius+delta),width//2 + delta,height//2 - (radius-delta)
#             img_crop_1 = img_crop[y1_min : y1_max, x1_min:x1_max]
#             img1,ep11,ep12 = self.scan_edge_position(img_crop_1,1)
#             # Distance between 2 point
#             if ep11 :
#                 ep11_default = (ep12[0][0] + x1_min ,ep12[0][1] + y1_min)
#                 radius11 = self.calculate_distance(ep11_default[0],ep11_default[1],centroid[0],centroid[1])
#                 radius_filter.append(radius11)
#             if ep12 :
#                 ep12_default = (ep12[0][0] + x1_min ,ep12[0][1] + y1_min)
#                 radius12 = self.calculate_distance(ep12_default[0],ep12_default[1],centroid[0],centroid[1])
#                 radius_filter.append(radius12)

#             # ROI 2
#             x2_min,y2_min,x2_max,y2_max =  width//2 + (radius-delta),height//2 - delta,width//2 + (radius+delta),height//2 +delta
#             img_crop_2 = img_crop[y2_min : y2_max, x2_min:x2_max]
#             img2,ep21,ep22 = self.scan_edge_position(img_crop_2,2)
#             # Distance between 2 point
#             if ep21 :
#                 ep21_default = (ep21[0][0] + x2_min ,ep21[0][1] + y2_min)
#                 radius21 = self.calculate_distance(ep21_default[0],ep21_default[1],centroid[0],centroid[1])
#                 radius_filter.append(radius21)
#             if ep22 :
#                 ep22_default = (ep22[0][0] + x2_min ,ep22[0][1] + y2_min)
#                 radius22 = self.calculate_distance(ep22_default[0],ep22_default[1],centroid[0],centroid[1])
#                 radius_filter.append(radius22)

#             # ROI 3
#             x3_min,y3_min,x3_max,y3_max = width//2 - delta,height//2 + (radius-delta) ,width//2 + delta,height//2 + (radius+delta)
#             img_crop_3 = img_crop[y3_min : y3_max, x3_min:x3_max]
#             img3,ep31,ep32 = self.scan_edge_position(img_crop_3,3)
#             # Distance between 2 point
#             if ep31 :
#                 ep31_default = (ep31[0][0] + x3_min ,ep31[0][1] + y3_min)
#                 radius31 = self.calculate_distance(ep31_default[0],ep31_default[1],centroid[0],centroid[1])
#                 radius_filter.append(radius31)
#             if ep32 :
#                 ep32_default = (ep32[0][0] + x3_min ,ep32[0][1] + y3_min)
#                 radius32 = self.calculate_distance(ep32_default[0],ep32_default[1],centroid[0],centroid[1])
#                 radius_filter.append(radius32)

#             # ROI 4
#             x4_min,y4_min,x4_max,y4_max = width//2 - (radius+delta),height//2 - delta ,width//2 - (radius-delta),height//2 +delta
#             img_crop_4 = img_crop[y4_min : y4_max, x4_min:x4_max]
#             img4,ep41,ep42 = self.scan_edge_position(img_crop_4,4)
#             if ep41 :
#                 ep41_default = (ep41[0][0] + x4_min ,ep41[0][1] + y4_min)
#                 radius41 =self.calculate_distance(ep41_default[0],ep41_default[1],centroid[0],centroid[1])
#                 radius_filter.append(radius41)
#             if ep42 :
#                 ep42_default = (ep42[0][0] + x4_min ,ep42[0][1] + y4_min)
#                 radius42 = self.calculate_distance(ep42_default[0],ep42_default[1],centroid[0],centroid[1])
#                 radius_filter.append(radius42)

#             # Hori = np.concatenate((img1, img2,img3,img4), axis=1)
#             # cv2.namedWindow("Detect Cirle Image", cv2.WINDOW_NORMAL) 
#             # cv2.imshow("Detect Cirle Image",Hori)
#             # cv2.waitKey(0)
#             return centroid[0],centroid[1],radius_filter
        
#         except Exception as error : 
#             print('Program Error - Detect circle:',error)
#             return centroid[0],centroid[1],radius_filter


#     def sharpnessSobel(self,img):
#         """
#         Gradient-based sharpness
#         """
#         # Convert the image to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Calculate the gradients using Sobel operator
#         gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#         gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

#         # Calculate the absolute gradients
#         abs_gradient_x = cv2.convertScaleAbs(gradient_x)
#         abs_gradient_y = cv2.convertScaleAbs(gradient_y)

#         # Calculate the sharpness as the mean of absolute gradients
#         sharpness = np.mean(abs_gradient_x) + np.mean(abs_gradient_y)
#         return sharpness

#     def run(self,img):
#         """
#         MIRROR TEST
#         1. Check location of circle (centroid and radius)
#         2. Check brightness of image 
#         3. Check shapness of image
#         """
        
#         results = True
#         self.img = img
#         self.img_draw = self.img.copy()
#         # Declear criterion
#         CENTER_X_MIN = self.MIRROR_DATA['Center X'][0]
#         CENTER_X_MAX = self.MIRROR_DATA['Center X'][1]
#         CENTER_Y_MIN = self.MIRROR_DATA['Center Y'][0]
#         CENTER_Y_MAX = self.MIRROR_DATA['Center Y'][1]
#         RADIUS_MIN = self.MIRROR_DATA['Circle Radius'][0]
#         RADIUS_REF = self.radius_ref
#         delta = self.delta
#         PX_VALUE = self.pixel_value
#         # delta = 40
#         RADIUS_MAX = self.MIRROR_DATA['Circle Radius'][1]
#         BRIGHTNESS_MIN = self.MIRROR_DATA['Iris 1'][0]
#         BRINGHTNESS_MAX = self.MIRROR_DATA['Iris 1'][1]
#         SHARPNESS_MIN = self.MIRROR_DATA['Focus 1'][0]
#         # 1. Check location
#         x,y,radius = self.detect_circle(img,RADIUS_REF,delta)
#         # print(radius.sort(reverse = True))
#         radius = sorted(radius,reverse=True)
#         print(radius)
#         if len(radius) > 0 :
#             if len(radius) != 8 :
#                 print('Mirror Test - Location NG - Radius')
#                 radius = radius[0]
#                 print("A")
#                 results = False
#             else :
#                 radius = radius[-3]
#                 if radius < RADIUS_MIN or radius > RADIUS_MAX :
#                     print('Mirror Test - Location NG - Radius')
#                     print("B")
#                     results = False

#             print(radius)
#             if (x < CENTER_X_MIN or x > CENTER_X_MAX ) or (y < CENTER_Y_MIN or y > CENTER_Y_MAX) :
#                 print('Mirror Test - Location NG - Centroid')
#                 results = False
            
#             cv2.circle(self.img_draw, (x, y), 3, (0, 0, 255), -1)
#             cv2.circle(self.img_draw, (x, y), int(radius), (0, 255, 0), 2)
#             cv2.putText(self.img_draw,"Ban kinh: " + str(round(radius*PX_VALUE,2)) + "mm",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#             cv2.putText(self.img_draw,"Tam guong: (" + str(x) + "," +str(y)+")",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#         else :
#             results = False
#             cv2.putText(self.img_draw,"No Circle",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
#         img_crop_1, img_crop_2, img_crop_3, img_crop_4 = self.select_ROI(self.img,RADIUS_REF,delta)
#         # 2. Check brightness of image
#         # ROI 1
#         brightness_roi1 = self.measure_light_rectangle(img_crop_1)
#         # ROI 2
#         brightness_roi2 = self.measure_light_rectangle(img_crop_2)
#         # ROI 3
#         brightness_roi3 = self.measure_light_rectangle(img_crop_3)
#         # ROI 4
#         brightness_roi4 = self.measure_light_rectangle(img_crop_4)
#         # print(brightness_roi1,brightness_roi2,brightness_roi3,brightness_roi4)
#         if brightness_roi1 < BRIGHTNESS_MIN or brightness_roi1 > BRINGHTNESS_MAX :
#             # print("'Mirror Test - ROI 1 NG Brightness")
#             results = False
#         if brightness_roi2 < BRIGHTNESS_MIN or brightness_roi1 > BRINGHTNESS_MAX :
#             # print("'Mirror Test - ROI 2 NG Brightness")
#             results = False
#         if brightness_roi3 < BRIGHTNESS_MIN or brightness_roi1 > BRINGHTNESS_MAX :
#             # print("'Mirror Test - ROI 3 NG Brightness")
#             results = False
#         if brightness_roi4 < BRIGHTNESS_MIN or brightness_roi1 > BRINGHTNESS_MAX :
#             # print("'Mirror Test - ROI 4 NG Brightness")
#             results = False
#         cv2.putText(self.img_draw,"Iris 1: " + str(round(brightness_roi1,0)),(50,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#         cv2.putText(self.img_draw,"Iris 2: " + str(round(brightness_roi2,0)),(50,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#         cv2.putText(self.img_draw,"Iris 3: " + str(round(brightness_roi3,0)),(50,250),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#         cv2.putText(self.img_draw,"Iris 4: " + str(round(brightness_roi4,0)),(50,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#         # print(results)
#         # print(brightness_roi1, brightness_roi2, brightness_roi3, brightness_roi4)
#         # 3. Check sharpness of image
#         # ROI 1
#         sharpness_roi1 = self.shapenessLaplacian(img_crop_1)
#         # ROI 2
#         sharpness_roi2 = self.shapenessLaplacian(img_crop_2)
#         # ROI 3
#         sharpness_roi3 = self.shapenessLaplacian(img_crop_3)
#         # ROI 4
#         sharpness_roi4 = self.shapenessLaplacian(img_crop_4)

#         if sharpness_roi1 < SHARPNESS_MIN :
#             # print("'Mirror Test - ROI 1 NG Sharpness")
#             results = False
#         if sharpness_roi2 < SHARPNESS_MIN :
#             # print("'Mirror Test - ROI 2 NG Sharpness")
#             results = False
#         if sharpness_roi3 < SHARPNESS_MIN :
#             # print("'Mirror Test - ROI 3 NG Sharpness")
#             results = False
#         if sharpness_roi4 < SHARPNESS_MIN :
#             # print("'Mirror Test - ROI 4 NG Sharpness")
#             results = False
#         cv2.putText(self.img_draw,"Focus 1: " + str(round(sharpness_roi1,0)),(50,350),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#         cv2.putText(self.img_draw,"Focus 2: " + str(round(sharpness_roi2,0)),(50,400),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#         cv2.putText(self.img_draw,"Focus 3: " + str(round(sharpness_roi3,0)),(50,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#         cv2.putText(self.img_draw,"Focus 4: " + str(round(sharpness_roi4,0)),(50,500),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#         print(results)
#         return self.img_draw,results


#     def display_images(image1, image2):
#         # Resize images to have the same height (assuming they have the same width)
#         height = 700
#         image1 = cv2.resize(image1, (int(image1.shape[1] * height / image1.shape[0]), height))
#         image2 = cv2.resize(image2, (int(image2.shape[1] * height / image2.shape[0]), height))

#         # Combine images side by side
#         combined_image = cv2.hconcat([image1, image2])

#         # Display the combined image
#         cv2.namedWindow("Images", cv2.WINDOW_NORMAL) 
#         cv2.imshow('Images', combined_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     def adjust_brightness(image_path, brightness_factor):
#         # Đọc bức ảnh
#         image = Image.open(image_path)

#         # Tạo một đối tượng Enhancer để điều chỉnh độ sáng
#         enhancer = ImageEnhance.Brightness(image)

#         # Điều chỉnh độ sáng sử dụng brightness_factor
#         adjusted_image = enhancer.enhance(brightness_factor)

#         # Lưu ảnh đã điều chỉnh
#         adjusted_image.save(f"..\\images\\brightness\\{brightness_factor}.jpg",quality = 100)