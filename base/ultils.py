import sys
from pathlib import Path
import root_path
from ultralytics import YOLO
from tkinter import filedialog
from PIL import Image, ImageTk
import glob
import mysql.connector
from tkinter import messagebox,simpledialog
import threading
import numpy as np
import time
import socket
from udp import UDPFinsConnection
from initialization import FinsPLCMemoryAreas
import tkinter as tk
import shutil
import os
import cv2
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator
import torch
import math
from collections import Counter
from ultralytics.utils import ops
from base.constants import *
from base.extention import *
soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def removefile():
    directory1 = 'C:/Users/CCSX009/Documents/yolov5/test_image/camera1/*.jpg'
    directory2 = 'C:/Users/CCSX009/Documents/yolov5/test_image/camera2/*.jpg'
    chk1 = glob.glob(directory1)
    for f1 in chk1:
        os.remove(f1)
        print('already delete folder 1')
    chk2 = glob.glob(directory2)
    for f2 in chk2:
        os.remove(f2)
        print('already delete folder 2')

class MySQL_Connection():

    def __init__(self,host,user,passwd,database):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.database = database

    def Connect_MySQLServer_root(self):
            db_connection = mysql.connector.connect(
            host= self.host,
            user=self.user, 
            passwd=self.passwd,
            database=self.database)     
            cursor = db_connection.cursor()
            return cursor,db_connection

    def check_connection(self):
        _,db_connection = self.Connect_MySQLServer()
        try:
            db_connection.ping(reconnect=True, attempts=3, delay=5)
            return True
        except mysql.connector.Error as err:
            messagebox.showinfo("Notification", f"Error connecting to the database: {str(err)}")
            return False

    def reconnect(self):
        _,db_connection = self.Connect_MySQLServer()
        try:
            db_connection.reconnect(attempts=3, delay=5)
            cursor = db_connection.cursor()  
            return True
        except mysql.connector.Error as err:
            messagebox.showinfo("Notification", f"Failed to reconnect to the database: {str(err)}")
            return False
        
    @staticmethod    
    def Connect_to_MySQLServer(host,user,passwd,database):
            db_connection = mysql.connector.connect(
            host= host,
            user=user, 
            passwd=passwd,
            database=database)                    
            cursor = db_connection.cursor()
            return cursor,db_connection
    
    def Connect_MySQLServer(self):
        try:
            db_connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.passwd
            )
            if db_connection.is_connected():
                cursor = db_connection.cursor(dictionary=True)
                return cursor, db_connection
        except Exception as e:
            print(f"Error: {e}")
            return None, None
      

class PLC_Connection():

    def __init__(self,host,port):
        self.host = host
        self.port = port
      
    def connect_plc_keyence(self):
        try:
            soc.connect((self.host, self.port))
            return True
        except OSError:
            print("Can't connect to PLC")
            time.sleep(3)
            print("Reconnecting....")
            return False

    def run_plc_keyence(self):
        connected = False
        while connected == False:
            connected = self.connect_plc_keyence(self.host,self.port)
        print("connected") 

    def read_plc_keyence(self,data):
        a = 'RD '
        c = '\x0D'
        d = a+ data +c
        datasend = d.encode("UTF-8")
        soc.sendall(datasend)
        data = soc.recv(1024)
        datadeco = data.decode("UTF-8")
        data1 = int(datadeco)
        return data1

    def socket_connect(self,soc,host,port):
        """
        Thực hiện kết nối với PLC
        host : địa chỉ IP của PLC
        port : port sử dụng bên PLC
        """
        try:
            soc.connect((host,port))
            return True
        except :
            return False

    def readdata(self,soc,data):
        """
        # Thực hiện đọc dữ liệu từ PLC 
        data : Thanh ghi bên PLC. Vd : DM1
        """
        a = 'RD '
        c = '\x0D'
        d = a + data + c
        datasend = d.encode("UTF-8")
        soc.sendall(datasend)
        response = soc.recv(1024)
        dataFromPLC = response.decode("UTF-8")
        return int(dataFromPLC)


    def writedata(self,soc,register,data):
        """
        Ghi dữ liệu vào PLC 
        register : Thanh ghi cần ghi dữ liệu bên PLC
        data : Dữ liệu cần truyền là
        """
        a = 'WR '
        b = ' '
        c = '\x0D'
        d = a+ register + b + str(data) + c
        datasend  = d.encode("UTF-8")
        soc.sendall(datasend)
        response = soc.recv(1024)


    def write_plc_keyence(self,register,data):
        a = 'WR '
        b = ' '
        c = '\x0D'
        d = a+register+b+str(data)+c
        datasend  = d.encode("UTF-8")
        soc.sendall(datasend)
        datares = soc.recv(1024)

    def connect_plc_omron(self):
        global fins_instance
        try:
            fins_instance = UDPFinsConnection()
            fins_instance.connect(self.host)
            fins_instance.dest_node_add=1
            fins_instance.srce_node_add=25
            return True
        except:
            print("Can't connect to PLC")
            for i in range(100000000):
                pass
            print("Reconnecting....")
            return False

    def run_plc_omron(self):
        connected = False
        while connected == False:
            connected = self.connect_plc_omron(self.host)
            print('connecting ....')
        print("connected plc") 

    def read_plc_omron(self,register):
        register = (register).to_bytes(2, byteorder='big') + b'\x00'
        read_var = fins_instance.memory_area_read(FinsPLCMemoryAreas().DATA_MEMORY_WORD,register)
        read_var = int.from_bytes(read_var[-2:], byteorder='big')  
        return read_var

    def write_plc_omron(self,register,data):
        register = (register).to_bytes(2, byteorder='big') + b'\x00'
        fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,register,b'\x00\x00',data)



class Base(PLC_Connection):

    def __init__(self):
        self.database = MySQL_Connection(None,None,None,None) 
        self.name_table = None
        self.item_code_cfg = None
        self.image_files = []
        self.current_image_index = 0
        self.state = 0
        self.right_angle = 180
        self._right_angle = 90
        self.password = " "
        self.lockable_widgets = [] 
        self.lock_params = []
        self.model_name_labels = []
        self.join = []
        self.ok_vars = []
        self.ng_vars = []
        self.num_inputs = []
        self.wn_inputs = []
        self.wx_inputs = []
        self.hn_inputs = []
        self.hx_inputs = []
        self.plc_inputs = []
        self.conf_scales = []
        self.rn_inputs = []
        self.rx_inputs = []
        self.rotage_join = []
        self.widgets_option_layout_parameters = []
        self.row_widgets = []
        self.weights = []
        self.datasets_format_model = []
        self.scale_conf_all = None
        self.size_model = None
        self.item_code = []
        self.make_cls_var = False
        self.permisson_btn = []
        self.model = None
        self.time_processing_output = None
        self.result_detection = None
        self.datasets_format_model = None
        self.process_image_func = None
        self.processing_functions = {
            'HBB': self.run_func_hbb,
            'OBB': self.run_func_obb
        }
        self.tuple = setupTools.track_conn()
        self.img_buffer = [] 
        self.counter = 0
        self.socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.complete = 'DM4006'

    def connect_database(self):
        cursor, db_connection  = self.database.Connect_MySQLServer()
        check_connection = self.database.check_connection()
        reconnect = self.database.reconnect()
        return cursor,db_connection,check_connection,reconnect
    
    def check_connect_database(self):
        cursor, db_connection = self.database.Connect_MySQLServer()
        if cursor is not None and db_connection is not None:
            pass
        else:
            messagebox.showwarning('Warning','Connection to database failed!')
        return cursor, db_connection
    
    def save_params_model(self):
        confirm_save_data = messagebox.askokcancel("Confirm", "Are you sure ?")
        cursor, db_connection, _, _ = self.connect_database()
        if confirm_save_data:
            try:
                item_code_value = str(self.item_code.get())
                dataset_format = self.datasets_format_model.get()
                weight = self.weights.get()
                confidence_all = int(self.scale_conf_all.get())
                size_model = int(self.size_model.get())
                cursor.execute(f"DELETE FROM {self.name_table} WHERE item_code = %s", (item_code_value,))

                for index in range(len(self.model_name_labels)):
                    values = self.get_values_for_insert(index, item_code_value, weight, confidence_all, size_model, dataset_format)
                    query_sql = self.build_insert_query(dataset_format)
                    cursor.execute(query_sql, values)

                db_connection.commit()
                messagebox.showinfo("Notification", "Saved parameters successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Data saved failed! Error: {str(e)}")
            
            finally:
                cursor.close()
                db_connection.close()

    def get_values_for_insert(self, index, item_code_value, weight, confidence_all, size_model, dataset_format):

        label_name = self.model_name_labels[index].cget("text")
        join_detect = self.join[index].get()
        OK_jont = self.ok_vars[index].get()
        NG_jont = self.ng_vars[index].get()
        num_labels = int(self.num_inputs[index].get())
        width_min = int(self.wn_inputs[index].get())
        width_max = int(self.wx_inputs[index].get())
        height_min = int(self.hn_inputs[index].get())
        height_max = int(self.hx_inputs[index].get())
        PLC_value = int(self.plc_inputs[index].get())
        cmpnt_conf = int(self.conf_scales[index].get())

        if dataset_format == 'OBB':
            rotage_min = float(self.rn_inputs[index].get())
            rotage_max = float(self.rx_inputs[index].get())
            return (item_code_value, weight, confidence_all, label_name, join_detect, OK_jont, NG_jont, num_labels,
                    width_min, width_max, height_min, height_max, PLC_value, cmpnt_conf, size_model, rotage_min, rotage_max, dataset_format)
        else: 
            return (item_code_value, weight, confidence_all, label_name, join_detect, OK_jont, NG_jont, num_labels,
                    width_min, width_max, height_min, height_max, PLC_value, cmpnt_conf, size_model, dataset_format)

    def build_insert_query(self, dataset_format):

        if dataset_format == 'OBB':
            return f"""
                INSERT INTO {self.name_table} 
                (item_code, weight, confidence_all, label_name, join_detect, OK, NG, num_labels, width_min, width_max, 
                height_min, height_max, PLC_value, cmpnt_conf, size_detection, rotage_min, rotage_max, dataset_format)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        else:
            return f"""
                INSERT INTO {self.name_table} 
                (item_code, weight, confidence_all, label_name, join_detect, OK, NG, num_labels, width_min, width_max, 
                height_min, height_max, PLC_value, cmpnt_conf, size_detection, dataset_format)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
    def torch_load_nodemap(self,source=None,task=None,device=None):
        return YOLO(model=source,task=task).to(device=device)

    def process_func_local(self,selected_format):
        self.process_image_func = self.processing_functions.get(selected_format, None)

    def preprocess(self,img):
        if isinstance(img, str):
            image_array = cv2.imread(img)
            fh = True
            return image_array,fh
        elif isinstance(img, np.ndarray):
            fh =False
            image_array = img
            imgRGB = cv2.cvtColor(image_array,cv2.COLOR_BayerGB2RGB)
            imgRGB = cv2.flip(imgRGB,0)
            return cv2.cvtColor(imgRGB, cv2.COLOR_BGR2RGB),fh
        
    def run_func_hbb(self, input_image, width, height):
        current_time = time.time()
        size_model_all = int(self.size_model.get())
        conf_all = int(self.scale_conf_all.get()) / 100 
        results = self.model(input_image,imgsz=size_model_all,conf=conf_all)
        settings_dict = {setting['label_name']: setting for setting in self.default_model_params}
        boxes_dict = results[0].boxes.cpu().numpy()
        xywh_list = boxes_dict.xywh.tolist()
        cls_list = boxes_dict.cls.tolist()
        conf_list = boxes_dict.conf.tolist()
        _valid_idex,_invalid_idex,list_cls_ng,_flag,lst_result= [],[],[],False,'ERROR'
        for index, (xywh, cls, conf) in enumerate(reversed(list(zip(xywh_list,cls_list,conf_list)))):
            setting = settings_dict[results[0].names[int(cls)]]
            if setting:
                if setting['join_detect']:
                    if xywh[2] < setting['width_min'] or xywh[2] > setting['width_max'] \
                            or xywh[3] < setting['height_min'] or xywh[3] > setting['height_max'] \
                            or int(conf*100) < setting['cmpnt_conf']:
                        _invalid_idex.append(int(index))
                        continue
                    _valid_idex.append(results[0].names[int(cls)])
                else:
                    _invalid_idex.append(int(index))           
        for model_name,setting in settings_dict.items():
            if setting['join_detect'] and setting['OK_jont']:
                if _valid_idex.count(setting['label_name']) != setting['num_labels']:
                    lst_result,_flag = 'NG',True
                    list_cls_ng.append(model_name)
            if setting['join_detect'] and setting['NG_jont']:
                if model_name in _valid_idex:
                    lst_result,_flag = 'NG',True
                    list_cls_ng.append(setting['label_name'])
        if not _flag:
            lst_result = 'OK'
        if self.make_cls_var.get():
            self._make_cls(input_image,results,self.default_model_params)
        show_img = np.squeeze(results[0].extract_npy(_invalid_idex=_invalid_idex))
        show_img = cv2.resize(show_img, (width, height), interpolation=cv2.INTER_AREA)
        output_image = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        lst_check_location,time_processing = None,f'{str(int((time.time()-current_time)*1000))}ms'
        return output_image,lst_result,list_cls_ng,lst_check_location,time_processing
    
    def run_func_obb(self,source,width,height):
        current_time = time.time()
        self.counter +=1
        size_model_all = int(self.size_model.get())
        conf_all = int(self.scale_conf_all.get())/100
        source,fh = self.preprocess(source)
        results = self.model(source,imgsz=size_model_all,conf=conf_all)
        settings_dict = {setting['label_name']: setting for setting in self.default_model_params}
        _xywhr,_cls,_conf=results[0].obb.cpu().numpy().xywhr.tolist(),results[0].obb.cpu().numpy().cls.tolist(),results[0].obb.cpu().numpy().conf.tolist()
        _valid_idex,_invalid_idex,list_cls_ng,_flag,lst_result,lst_check_location= [],[],[],False,'ERROR',[]
        for index,(xywhr,cls,conf) in enumerate(reversed(list(zip(_xywhr,_cls,_conf)))):
            setting = settings_dict[results[0].names[int(cls)]]
            if setting:
                if setting['join_detect']:
                    if (xywhr[2] < setting['width_min'] or xywhr[2] > setting['width_max']) \
                        or (xywhr[3] < setting['height_min'] or xywhr[3] > setting['height_max']) \
                        or (int(conf*100) < setting['cmpnt_conf']):
                        _invalid_idex.append(int(index))
                    try:
                        if LOCALTION_OBJS:
                            list_cls_ng,_invalid_idex,lst_check_location,_flag = \
                        self._bbox_localtion_direction_objs(
                            _flag, index, setting, xywhr, list_cls_ng, _invalid_idex, lst_check_location)
                    except:
                        pass
                    _valid_idex.append(results[0].names[int(cls)])
                else:
                    _invalid_idex.append(int(index))  
        for index,(xywhr,cls,conf) in enumerate(reversed(list(zip(_xywhr,_cls,_conf)))):  
                setting = settings_dict[results[0].names[int(cls)]]
                if setting:
                    if setting['join_detect'] and setting['OK_jont']: 
                        if Counter(_valid_idex)[setting['label_name']] != setting['num_labels']:
                            _flag = True
                            list_cls_ng.append(setting['label_name'])
                    if setting['join_detect'] and setting['NG_jont']:
                        if setting['label_name'] or results[0].names[int(cls)] in _valid_idex:
                            _flag = True
                            list_cls_ng.append(setting['label_name'])
                            _invalid_idex.append(int(index))  
        lst_result = 'OK' if not _flag else 'NG'
        if not fh:
            if self.counter == 6: 
                self.writedata(self.socket,self.complete,1)
                self.counter = 0
        if self.make_cls_var.get():       
            self.xyxyxyxy2xywhr_indirect(source,results[0],_xywhr,_cls,_conf,self.default_model_params)
        show_img = np.squeeze(results[0].extract_npy(list_remove=_invalid_idex))
        show_img = cv2.resize(show_img, (width, height), interpolation=cv2.INTER_AREA)
        output_image = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        lst_check_location = sorted(lst_check_location, key=lambda item: item[0])
        lst_check_location = lst_check_location if lst_check_location != [] else []
        time_processing = f'{str(int((time.time()-current_time)*1000))}ms'
        return output_image,lst_result,list_cls_ng,lst_check_location,time_processing
    
    def _bbox_localtion_direction_objs(self,_flag,index,setting,xywhr,
        list_cls_ng,_invalid_idex,lst_check_location):
        if OBJECTS_ANGLE:
            if setting['label_name'] in ITEM:
                radian = (float(round(math.degrees(xywhr[4]),1)))
                if (radian < setting['rotage_min']) or \
                    (radian > setting['rotage_max']):
                    _flag = True
                    list_cls_ng.append(setting['label_name']) 
                    _invalid_idex.append(int(index))  
        if OBJECTS_COORDINATES:
            if setting['label_name'] == ITEM[0]:
                if xywhr[0] and xywhr[1] :
                    id,obj_x,obj_y,x,y = setupTools.tracking_id(self.tuple,xywhr[0],xywhr[1])
                    id,obj_x,obj_y,x,y,conn,val = setupTools.func_localtion(id,obj_x,obj_y,x,y)
                    lst_check_location.append((id, obj_x, obj_y, x, y,round(math.degrees(xywhr[4]),1),conn))
                    if not val:
                        _flag = True 
                else: 
                    _flag = True 
        return list_cls_ng,_invalid_idex,lst_check_location,_flag
    
    def _default_settings(self):
        self.default_model_params = [
            {
                'label_name':  label.cget("text"),
                'join_detect': self.join[index].get(),
                'OK_jont': self.ok_vars[index].get(),
                'NG_jont': self.ng_vars[index].get(),
                'num_labels': int(self.num_inputs[index].get()),
                'width_min': int(self.wn_inputs[index].get()),
                'width_max': int(self.wx_inputs[index].get()),
                'height_min': int(self.hn_inputs[index].get()),
                'height_max': int(self.hx_inputs[index].get()),
                'PLC_value': int(self.plc_inputs[index].get()),
                'cmpnt_conf': int(self.conf_scales[index].get()),
                'rotage_min': float(self.rn_inputs[index].get()),
                'rotage_max': float(self.rx_inputs[index].get()),
            }
            for index, label in enumerate(self.model_name_labels)
        ]
    
    def _make_cls(self,image_path_mks_cls,results,model_settings):  
        ims = cv2.imread(image_path_mks_cls)
        w,h,_ = ims.shape
        with open(image_path_mks_cls[:-3] + 'txt', "a") as file:
            for params in results.xywhn:
                params = params.tolist()
                for item in range(len(params)):
                    param = params[item]
                    param = [round(i,6) for i in param]
                    number_label = int(param[5])
                    conf_result = float(param[4])
                    width_result = float(param[2])*w
                    height_result = float(param[3])*h
                    for setting in model_settings:
                        if results.names[int(number_label)] == setting['label_name']:
                            if setting['join_detect']: 
                                if width_result < setting['width_min'] or width_result > setting['width_max'] \
                                        or height_result < setting['height_min'] or height_result > setting['height_max'] \
                                        or conf_result < setting['cmpnt_conf']: 
                                    formatted_values = ["{:.6f}".format(value) for value in param[:4]]
                                    output_line = "{} {}\n".format(str(number_label),' '.join(formatted_values))
                                    file.write(output_line)
        path = Path(image_path_mks_cls).parent
        path = os.path.join(path,'classes.txt')
        with open(path, "w") as file:
            for index in range(len(results.names)):
                file.write(str(results.names[index])+'\n')

    def xywhr2xyxyxyxy(self,class_id,x_center,y_center,width,height,angle,im_height,im_width):
        half_width = width / 2
        half_height = height / 2
        angle_rad = np.deg2rad(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        corners = np.array([
            [-half_width, -half_height],  
            [half_width, -half_height], 
            [half_width, half_height],   
            [-half_width, half_height]
        ])
        rotated_corners = np.dot(corners, rotation_matrix)
        final_corners = rotated_corners + np.array([x_center, y_center])
        normalized_corners = final_corners / np.array([im_width,im_height])
        return [int(class_id)] + normalized_corners.flatten().tolist()

    def format_params_xywhr2xyxyxyxy(self,des_path,progress_label):
        input_folder = des_path
        os.makedirs(os.path.join(input_folder,'instance'),exist_ok=True)
        output_folder = (os.path.join(input_folder,'instance'))
        total_fl = len(des_path) 
        for index,txt_file in enumerate(os.listdir(input_folder)):
            if txt_file.endswith('.txt'):
                if txt_file == 'classes.txt':
                    continue
                input_path = os.path.join(input_folder, txt_file)
                im = cv2.imread(f'{input_path[:-3]}jpg')
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
                        class_id,x_center,y_center,width,height,angle = params
                        converted_label = self.xywhr2xyxyxyxy(class_id,x_center,y_center,width,height,angle,im_height,im_width)
                        out_file.write(" ".join(map(str, converted_label)) + '\n')
                progress_retail = (index + 1) / total_fl * 100
                progress_label.config(text=f"Converting YOLO OBB Dataset Format to DOTA Format: {progress_retail:.2f}%")
                progress_label.update_idletasks()
                os.replace(output_path, input_path)
        shutil.rmtree(output_folder)

    def xyxyxyxy2xywhr_indirect(self,input_image,results,xywhr_list,cls_list,conf_list,model_settings):
        settings_dict = {setting['label_name']: setting for setting in model_settings}
        with open(input_image[:-3] + 'txt', 'a') as out_file:
            out_file.write('YOLO_OBB\n')
            for index, (xywhr, cls, conf) in enumerate(reversed(list(zip(xywhr_list, cls_list, conf_list)))):
                setting = settings_dict[results.names[int(cls)]]
                xywhr_list[index][-1] = math.degrees(xywhr_list[index][-1])
                if xywhr_list[index][-1] > self._right_angle:
                    xywhr_list[index][-1] = self.right_angle - xywhr_list[index][-1]
                else:
                    xywhr_list[index][-1] = -(xywhr_list[index][-1])
                line = [int(cls_list[index])] + xywhr_list[index]  
                formatted_line = " ".join(["{:.6f}".format(x) if isinstance(x, float) else str(x) for x in line])
                if setting:
                    if setting['join_detect']:
                        if xywhr[2] < setting['width_min'] or xywhr[2] > setting['width_max'] \
                                or xywhr[3] < setting['height_min'] or xywhr[3] > setting['height_max'] \
                                or int(conf*100) < setting['cmpnt_conf']:
                                continue                           
                        out_file.write(f'{formatted_line}\n')
        path = Path(input_image).parent
        path = os.path.join(path,'classes.txt')
        with open(path, "w") as file:
            for i in range(len(results.names)):
                file.write(f'{str(results.names[i])}\n')
               
    def load_data_model(self):
        cursor, db_connection,_,_ = self.connect_database()
        cursor.execute(f"SELECT * FROM {self.name_table} WHERE item_code = %s", (self.item_code_cfg,))
        records = cursor.fetchall()
        cursor.close()
        db_connection.close()
        if records:
            first_record = records[0]
            load_item_code = first_record['item_code']
            load_path_weight = first_record['weight']
            load_confidence_all_scale = first_record['confidence_all']
            load_dataset_format = first_record['dataset_format']
            size_model = first_record['size_detection']
        return records,load_path_weight,load_item_code,load_confidence_all_scale,load_dataset_format,size_model

    def load_parameters_model(self,initial_model,load_path_weight,load_item_code,
        load_confidence_all_scale,records,load_dataset_format,size_model,Frame):
        self._set_widget(self.datasets_format_model,load_dataset_format)
        self._set_widget(self.weights,load_path_weight)
        self._set_widget(self.item_code,load_item_code)
        self._set_intvalue(self.size_model,size_model)
        self._set_intvalue(self.scale_conf_all,load_confidence_all_scale)
        try:
            if load_dataset_format == 'HBB':
                self.process_func_local(load_dataset_format)
                self._clear_widget(Frame)
                self.option_layout_parameters(Frame,self.model)
                for index,_ in enumerate(initial_model.names):
                    for record in records:                
                        if record['label_name'] == initial_model.names[index]:
                            self._set_intvalue(self.join[index],bool(record['join_detect']))
                            self._set_intvalue(self.ok_vars[index],bool(record['OK']))
                            self._set_intvalue(self.ng_vars[index],bool(record['NG']))
                            self._set_widget(self.num_inputs[index],record['num_labels'])
                            self._set_widget(self.wn_inputs[index],record['width_min'])
                            self._set_widget(self.wx_inputs[index],record['width_max'])
                            self._set_widget(self.hn_inputs[index],record['height_min'])
                            self._set_widget(self.hx_inputs[index],record['height_max'])
                            self._set_widget(self.plc_inputs[index],record['PLC_value'])
                            self._set_intvalue(self.conf_scales[index],record['cmpnt_conf'])
                
            elif load_dataset_format == 'OBB':
                self.process_func_local(load_dataset_format)
                self._clear_widget(Frame)
                self.option_layout_parameters_orient_bounding_box(Frame,self.model)
                for index,_ in enumerate(initial_model.names):          
                    for record in records:                
                        if record['label_name'] == initial_model.names[index]:
                            self._set_intvalue(self.join[index],bool(record['join_detect']))
                            self._set_intvalue(self.ok_vars[index],bool(record['OK']))
                            self._set_intvalue(self.ng_vars[index],bool(record['NG']))
                            self._set_widget(self.num_inputs[index], record['num_labels'])
                            self._set_widget(self.wn_inputs[index], record['width_min'])
                            self._set_widget(self.wx_inputs[index], record['width_max'])
                            self._set_widget(self.hn_inputs[index], record['height_min'])
                            self._set_widget(self.hx_inputs[index], record['height_max'])
                            self._set_widget(self.plc_inputs[index], record['PLC_value'])
                            self._set_intvalue(self.conf_scales[index],record['cmpnt_conf'])
                            self._set_widget(self.rn_inputs[index], record['rotage_min'])
                            self._set_widget(self.rx_inputs[index], record['rotage_max'])
                self._default_settings()
        except IndexError as e:
            messagebox.showerror("Error", f"Load parameters failed! Error: {str(e)}")

    def change_model(self,Frame):
        selected_file = filedialog.askopenfilename(title="Choose a file", filetypes=[("Model Files", "*.pt")])
        if selected_file:
            self._set_widget(self.weights,selected_file)
            self.model = self.torch_load_nodemap(source=selected_file)
            self.confirm_dataset_format(Frame)
        else:
            messagebox.showinfo("Notification","Please select the correct training file!")
            pass

    def confirm_dataset_format(self,Frame):
        if self.datasets_format_model.get() == 'OBB':
            for widget in Frame.grid_slaves():
                widget.grid_forget()
            self.option_layout_parameters_orient_bounding_box(Frame,self.model)
        elif self.datasets_format_model.get() =='HBB':
            for widget in Frame.grid_slaves():
                widget.grid_forget()
            self.option_layout_parameters(Frame,self.model)

    def _set_widget(self,widget,value):
        widget.delete(0, tk.END)
        widget.insert(0, value)

    def _set_intvalue(self,int_widget,value):
        int_widget.set(value)

    def _clear_widget(self,Frame):
        for widget in Frame.grid_slaves():
            widget.grid_forget()

    def load_params_child(self):
        cursor, db_connection,_,_ = self.connect_database()
        try:
            cursor.execute(f"SELECT * FROM {self.name_table} WHERE item_code = %s", (self.item_code.get().__str__(),))
        except Exception as e:
            messagebox.showwarning('Warning',f'{e}: Item Code does not exist')
        records = cursor.fetchall()
        model = self.torch_load_nodemap(source=self.weights.get())
        cursor.close()
        db_connection.close()
        return records,model

    def load_parameters_from_weight(self, records):
        confirm_load_parameters = messagebox.askokcancel("Confirm", "Are you sure you want to load the parameters?")
        if confirm_load_parameters:
            records, initial_model = self.load_params_child()
            try:
                if self.datasets_format_model.get() == 'HBB':  
                    for index,_ in enumerate(initial_model.names):    
                        for record in records:                
                            if record['label_name'] == initial_model.names[index]:
                                self._set_intvalue(self.join[index],bool(record['join_detect']))
                                self._set_intvalue(self.ok_vars[index],bool(record['OK']))
                                self._set_intvalue(self.ng_vars[index],bool(record['NG']))
                                self._set_widget(self.num_inputs[index],record['num_labels'])
                                self._set_widget(self.wn_inputs[index],record['width_min'])
                                self._set_widget(self.wx_inputs[index],record['width_max'])
                                self._set_widget(self.hn_inputs[index],record['height_min'])
                                self._set_widget(self.hx_inputs[index],record['height_max'])
                                self._set_widget(self.plc_inputs[index],record['PLC_value'])
                                self._set_intvalue(self.conf_scales[index],record['cmpnt_conf'])
                elif self.datasets_format_model.get() == 'OBB':
                    for index,_ in enumerate(initial_model.names):          
                        for record in records:                
                            if record['label_name'] == initial_model.names[index]:
                                self._set_intvalue(self.join[index],bool(record['join_detect']))
                                self._set_intvalue(self.ok_vars[index],bool(record['OK']))
                                self._set_intvalue(self.ng_vars[index],bool(record['NG']))
                                self._set_widget(self.num_inputs[index], record['num_labels'])
                                self._set_widget(self.wn_inputs[index], record['width_min'])
                                self._set_widget(self.wx_inputs[index], record['width_max'])
                                self._set_widget(self.hn_inputs[index], record['height_min'])
                                self._set_widget(self.hx_inputs[index], record['height_max'])
                                self._set_widget(self.plc_inputs[index], record['PLC_value'])
                                self._set_intvalue(self.conf_scales[index],record['cmpnt_conf'])
                                self._set_widget(self.rn_inputs[index], record['rotage_min'])
                                self._set_widget(self.rx_inputs[index], record['rotage_max'])
            except IndexError as e:
                messagebox.showerror("Error", f"Load parameters failed! Error: {str(e)}")

    def handle_image(self,img1_orgin, width, height,camera_frame):
        for widget in camera_frame.winfo_children():
            widget.destroy()
        image_result,lst_result,label_ng,_,time_processing = self.process_image_func(img1_orgin, width, height)
        img_pil = Image.fromarray(image_result)
        photo = ImageTk.PhotoImage(img_pil)
        canvas = tk.Canvas(camera_frame, width=width, height=height)
        canvas.grid(row=2, column=0, padx=10, pady=10, sticky='ew')
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo
        canvas.create_text(10, 10, anchor=tk.NW, text=f'Time: {time_processing}', fill='black', font=('Segoe UI', 20))
        canvas.create_text(10, 40, anchor=tk.NW, text=f'Result: {lst_result}', fill='green' if lst_result == 'OK' else 'red', font=('Segoe UI', 20))
        if not label_ng:
            canvas.create_text(10, 70, anchor=tk.NW, text=f' ', fill='green', font=('Segoe UI', 20))
        else:
            label_ng = ','.join(label_ng)
            canvas.create_text(10, 70, anchor=tk.NW, text=f'NG: {label_ng}', fill='red', font=('Segoe UI', 20))
        return lst_result  
    
    def detect_single_img(self, camera_frame):
        selected_file = filedialog.askopenfilename(title="Choose a file", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if selected_file:
            for widget in camera_frame.winfo_children():
                widget.destroy()
            width = 480
            height = 450
            self.handle_image(selected_file, width, height,camera_frame)
        else: 
            pass
           
    def detect_multi_img(self,camera_frame):
        selected_folder = filedialog.askdirectory(title="Choose a folder")
        if selected_folder:
            self.image_files = [os.path.join(selected_folder, f) for f in os.listdir(selected_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.current_image_index = 0
            if self.image_files:
                for widget in camera_frame.winfo_children():
                    widget.destroy()
                self.show_image(self.current_image_index,camera_frame)
            else:
                messagebox.showinfo("No Images", "The selected folder contains no images.")
        else:
            pass

    def show_image(self,index,camera_frame):  
        width = 480
        height = 450
        image_path = self.image_files[index]
        self.handle_image(image_path, width, height,camera_frame)

    def detect_next_img(self,camera_frame):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_image(self.current_image_index,camera_frame)
        else:
            messagebox.showinfo("End of Images", "No more images in the folder.")

    def detect_previos_img(self,camera_frame):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.current_image_index,camera_frame)
        else:
            messagebox.showinfo("Start of Images", "This is the first image in the folder.")

    def detect_auto(self, camera_frame):
        selected_folder_original = filedialog.askdirectory(title="Choose a folder")
        if selected_folder_original: 
            selected_folder = glob.glob(selected_folder_original + '/*.jpg')
            if not selected_folder:
                pass
            self.image_index = 0
            self.selected_folder_detect_auto = selected_folder
            self.camera_frame = camera_frame
            self.image_path_mks_cls = []
            self.lst_result = None
            def process_next_image():
                if self.image_index < len(self.selected_folder_detect_auto):
                    self.image_path_mks_cls = self.selected_folder_detect_auto[self.image_index]
                    width = 480
                    height = 450
                    self.lst_result = self.handle_image(self.image_path_mks_cls, width, height,camera_frame)
                    self.image_index += 1
                    self.camera_frame.after(500, process_next_image)
            process_next_image()
        else:
            pass

    
    def logging(self, folder_ok, folder_ng, logging_ok_checkbox_var, logging_ng_checkbox_var, camera_frame, percent_entry,logging_frame):
        selected_folder_original = filedialog.askdirectory(title="Choose a folder")
        selected_folder = glob.glob(os.path.join(selected_folder_original,'*.jpg'))
        width = 480
        height = 450
        self.image_index = 0
        self.selected_folder_logging = selected_folder
        self.logging_frame = logging_frame
        self.percent_entry = percent_entry
        total_images = len(self.selected_folder_logging)
        if not self.selected_folder_logging:
            return
        
        def update_progress(current_index, total_images):
            percent_value = int((current_index + 1) / total_images * 100)
            self.percent_entry.delete(0, tk.END)
            self.percent_entry.insert(0, f"{percent_value}%")

        def process_images():
            for img in self.selected_folder_logging:
                basename = os.path.basename(img)
                if self.image_index < total_images:
                    lst_result = self.handle_image(img, width, height, camera_frame)
                    if lst_result == 'OK':
                        if logging_ok_checkbox_var.get():
                            shutil.move(img, os.path.join(folder_ok.get(), basename))
                    else:
                        if logging_ng_checkbox_var.get():
                            shutil.move(img, os.path.join(folder_ng.get(), basename))
                    self.logging_frame.after(10, update_progress, self.image_index, total_images)
                    self.image_index += 1
                else:
                    messagebox.showinfo("End of Images", "No more images in the folder.")
                    break
            self.percent_entry.delete(0, tk.END)
            self.percent_entry.insert(0, "0%")
        threading.Thread(target=process_images).start()
                
    def toggle_state_layout_model(self): 
        if self.state == 1:
            password = simpledialog.askstring("Administrator", "Enter password:", show="*")
            if password == self.password:
                self.state = 0
                self.permisson_btn.config(text="Lock")
                self.toggle_widgets_state("normal")
            else:
                messagebox.showerror("Error", "Incorrect password!")
        else:
            self.state = 1
            self.permisson_btn.config(text="Unlock")
            self.toggle_widgets_state("disabled")

    def toggle_widgets_state(self,state):
        for widget in self.lockable_widgets:
            widget.config(state=state)

    def toggle_state_option_layout_parameters(self):
        for widget in self.lock_params:
            widget.config(state='disabled')

    def pick_folder_ok(self,folder_ok):
        file_path = filedialog.askdirectory(title="Choose a folder")
        if file_path:
            folder_ok.delete(0,tk.END)
            folder_ok.insert(0,file_path)
           
    def pick_folder_ng(self,folder_ng):
        file_path = filedialog.askdirectory(title="Choose a folder")
        if file_path:
            folder_ng.delete(0,tk.END)
            folder_ng.insert(0,file_path)

    def load_first_img(self):
        filename = r"C:\Users\CCSX009\Documents\ultralytics-main\2024-03-05_00-01-31-398585-C1.jpg"
        self.model(filename, imgsz=608, conf=0.2)
        print('Load model 1 successfully')
    
class base_handle_video(PLC_Connection,MySQL_Connection):

    def __init__(self):
        self.database = MySQL_Connection(None,None,None,None) 
        self.name_table = None
        self.item_code_cfg = None
        self.image_files = []
        self.current_image_index = 0
        self.state = 0
        self.password = " "
        self.lockable_widgets = [] 
        self.lock_params = []
        self.model_name_labels = []
        self.join = []
        self.ok_vars = []
        self.ng_vars = []
        self.num_inputs = []
        self.wn_inputs = []
        self.wx_inputs = []
        self.hn_inputs = []
        self.hx_inputs = []
        self.plc_inputs = []
        self.conf_scales = []
        self.widgets_option_layout_parameters = []
        self.row_widgets = []
        self.weights = None
        self.scale_conf_all = None
        self.size_model = None
        self.item_code = []
        self.make_cls_var = False
        self.permisson_btn = []
        self.model = None
        self.time_processing_output = None
        self.result_detection = None

    def connect_database(self):
        cursor, db_connection  = self.database.Connect_MySQLServer()
        check_connection = self.database.check_connection()
        reconnect = self.database.reconnect()
        return cursor,db_connection,check_connection,reconnect
    
    def load_data_model_vid(self):
        cursor, db_connection,_,_ = self.connect_database()
        cursor.execute(f"SELECT * FROM {self.name_table} WHERE item_code = %s", (self.item_code_cfg,))
        records = cursor.fetchall()
        cursor.close()
        db_connection.close()
        if records:
            first_record = records[0]
            load_item_code = first_record['item_code']
            load_path_weight = first_record['weight']
            load_confidence_all_scale = first_record['confidence_all']
        else: 
            first_record = None
            load_item_code = None
            load_path_weight = None
            load_confidence_all_scale = None

        return records,load_path_weight,load_item_code,load_confidence_all_scale

    def load_parameters_model_vid(self,initial_model,load_path_weight,load_item_code,load_confidence_all_scale,records):
        self.weights.delete(0, tk.END)
        self.weights.insert(0, load_path_weight)
        try:
            self.item_code.delete(0, tk.END)
            self.item_code.insert(0, load_item_code)
            self.scale_conf_all.set(load_confidence_all_scale)
        except: 
            pass
        try:
            for index in range(len(initial_model.names)):          
                for record in records:                
                    if record['label_name'] == initial_model.names[index]:
                        self.join[index].set(bool(record['join_detect']))
                        self.ok_vars[index].set(bool(record['OK']))
                        self.ng_vars[index].set(bool(record['NG']))
                        self.num_inputs[index].delete(0, tk.END)
                        self.num_inputs[index].insert(0, record['num_labels'])
                        self.wn_inputs[index].delete(0, tk.END)
                        self.wn_inputs[index].insert(0, record['width_min'])
                        self.wx_inputs[index].delete(0, tk.END)
                        self.wx_inputs[index].insert(0, record['width_max'])
                        self.hn_inputs[index].delete(0, tk.END)
                        self.hn_inputs[index].insert(0, record['height_min'])                
                        self.hx_inputs[index].delete(0, tk.END)
                        self.hx_inputs[index].insert(0, record['height_max'])
                        self.plc_inputs[index].delete(0, tk.END)
                        self.plc_inputs[index].insert(0, record['PLC_value'])
                        self.conf_scales[index].set(record['cmpnt_conf'])
        except IndexError as e:
            messagebox.showerror("Error", f"Load parameters failed! Error: {str(e)}")

    def select_video(self,path_video):
        selected_file = filedialog.askopenfilename(title="Choose a file", filetypes=[("Video Files Type", "*.mp4")])
        if selected_file:
            path_video.delete(0,tk.END)
            path_video.insert(0,selected_file)

    def render(self,
        progress_label,
        video_canvas,
        source = None,
        device="cpu",
        save_img=True,
        exist_ok=False,
        classes=None,
        line_thickness=2,
    ):
        if source == None : 
            messagebox.showwarning("Warning", "You have to select a valid file")
        vid_frame_count = 0
        device = '0' if torch.cuda.is_available() else 'cpu'
        self.model.to("cuda") if device == "0" else self.model.to("cpu")
        names = self.model.model.names
        videocapture = cv2.VideoCapture(source)
        frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
        fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")
        total_frames = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
        save_dir = increment_path(Path("ultralytics_source_output") / "exp", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))
        while videocapture.isOpened():
            success, frame = videocapture.read()
            if not success:
                break
            vid_frame_count += 1
            self.handle_params_video(frame,classes,line_thickness,names)
            if save_img:
                video_writer.write(frame)
                self.update_progress(vid_frame_count, total_frames, progress_label)
        video_writer.release()
        videocapture.release()
        cv2.destroyAllWindows()
        self.show(save_dir,video_canvas)

    def update_progress(self, current_frame, total_frames, progress_label):
        device = 'GPU' if torch.cuda.is_available() else 'CPU'
        progress = (current_frame / total_frames) * 100
        progress_label.config(text=f"Rendering on {device} in progress: {progress:.2f}%")
        progress_label.update_idletasks()

    def show(self, save_dir,video_canvas):
        video_path = list(save_dir.glob("*.mp4"))[0]
        cap = cv2.VideoCapture(str(video_path))
        self.update_frame(cap, video_canvas)

    def update_frame(self, cap, video_canvas):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (600, 550))
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_canvas.imgtk = imgtk
            video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        video_canvas.after(20, self.update_frame, cap, video_canvas)

    def start_rendering(self,path_video,progress_label,video_canvas):
        kwargs = {
            "progress_label": progress_label,
            "video_canvas": video_canvas,
            "source": path_video,
            "device": "cpu",
            "save_img": True,
            "exist_ok": False,
            "classes": None,
            "line_thickness": 2
        }
        processing_thread = threading.Thread(target=self.render, kwargs=kwargs)
        processing_thread.start()

    def handle_params_video(self,frame,classes,line_thickness,names): 
             
        results = self.model.track(frame, persist=True, classes=classes)
        model_settings = [
            {
                'label_name':  self.model_name_labels[index].cget("text"),
                'join_detect': self.join[index].get(),
                'OK_jont': self.ok_vars[index].get(),
                'NG_jont': self.ng_vars[index].get(),
                'num_labels': int(self.num_inputs[index].get()),
                'width_min': int(self.wn_inputs[index].get()),
                'width_max': int(self.wx_inputs[index].get()),
                'height_min': int(self.hn_inputs[index].get()),
                'height_max': int(self.hx_inputs[index].get()),
                'PLC_value': int(self.plc_inputs[index].get()),
                'cmpnt_conf': int(self.conf_scales[index].get()),
            }
            for index in range(len(self.model_name_labels))
        ]
        settings_dict = {setting['label_name']: setting for setting in model_settings}
        if results[0].boxes.id is not None:
            coun_id = []
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy().tolist()
            clss = results[0].boxes.cls.cpu().numpy().tolist()
            xywhs = results[0].boxes.xywh.cpu().numpy().tolist()
            conf_list = results[0].boxes.conf.cpu().numpy().tolist()
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))
            for box, track_id, cls, xywh, conf in zip(boxes, track_ids, clss, xywhs,conf_list):
                label = f'{str(track_id)} {names[cls]} {conf:.2f} x:{str(int(xywh[0]))} y:{str(int(xywh[1]))}'
                if settings_dict[results[0].names[int(cls)]]:
                    if settings_dict[results[0].names[int(cls)]]['join_detect']:                      
                        annotator.box_label(box, label, color=(255,0,0))                         
                a_point = (200, 1500)
                d_point = (3700, 2000)
                color = (255, 255, 255)
                thickness = 2
                cv2.rectangle(frame, a_point, d_point, color, thickness)

                if a_point[0] <= xywh[0] <= d_point[0] and a_point[1] <= xywh[1] <= d_point[1]:
                    if track_id in coun_id:
                        continue
                    coun_id.append(track_id)
            text_position = (a_point[0] + 10, a_point[1] + 30)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            text_color = (0, 0, 255)
            text_thickness = 2
            cv2.putText(frame, str(len(coun_id)), text_position, font, font_scale, text_color, text_thickness)
            if len(coun_id)>2:
                cv2.putText(frame, 'NG',(200,200),cv2.FONT_HERSHEY_COMPLEX, 3,(0,0,255),5)
            else:
                cv2.putText(frame, 'OK',(200,200),cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),5)  

        else: 
            pass