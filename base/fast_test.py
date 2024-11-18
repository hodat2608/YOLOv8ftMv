import tkinter as tk
import root_path
from tkinter import ttk
from PIL import Image, ImageTk  # Sử dụng để hiển thị dấu tích và dấu X
from base.device_config import *
from base.ultils import *
from IOConnection.basler_pylon.PylonExportImgBuffer import *
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class ConnectionStatusGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trạng thái kết nối")
        self.geometry("400x150")

        self.check_img = ImageTk.PhotoImage(Image.open(r"base\Asserts\check.png").resize((20, 20)))
        self.cross_img = ImageTk.PhotoImage(Image.open(r"base\Asserts\delete.png").resize((20, 20)))
        self.reconnect_img = ImageTk.PhotoImage(Image.open(r"base\Asserts\link.png").resize((15, 15)))

        self.mysql_label, self.mysql_status, self.mysql_reconnect = self.create_connection_row("Kết nối MySQL", self.reconnect_mysql, row=0)
        self.camera_label, self.camera_status, self.camera_reconnect = self.create_connection_row("Kết nối Camera", self.reconnect_camera, row=1)
        self.plc_label, self.plc_status, self.plc_reconnect = self.create_connection_row("Kết nối PLC", self.reconnect_plc, row=2)

        self.check_connections()

    def create_connection_row(self, label_text, reconnect_command, row):
    
        label = ttk.Label(self, text=label_text, font=("Arial", 12))
        label.grid(row=row, column=0, sticky="w", padx=10, pady=5)

        status = ttk.Label(self)
        status.grid(row=row, column=1, padx=10, pady=5)

        reconnect_button = tk.Button(self, image=self.reconnect_img, command=reconnect_command, borderwidth=0)
        reconnect_button.grid(row=row, column=2, padx=10, pady=5)

        return label, status, reconnect_button

    def check_connections(self):
        """Kiểm tra trạng thái các kết nối và cập nhật giao diện."""
        self.update_status(self.mysql_status, self.check_mysql_connection())
        self.update_status(self.camera_status, self.check_camera_connection())
        self.update_status(self.plc_status, self.check_plc_connection())

    def update_status(self, status_label, is_connected):
        """Cập nhật hình ảnh trạng thái."""
        status_label.config(image=self.check_img if is_connected else self.cross_img)

    def check_mysql_connection(self):
        try:
            self.database = MySQL_Connection(
                MYSQL_CONNECTION['HOST'],
                MYSQL_CONNECTION['ROOT'],
                MYSQL_CONNECTION['PASSWORD'],
                MYSQL_CONNECTION['DATABASE']
            )
            return self.database.check_connection()
        except:
            return False

    def check_camera_connection(self):
        try:
            self.request_pylon = Basler_Pylon_xFunc(
                BASLER_UNITS_CAMERA_1['Serial number'],
                BASLER_UNITS_CAMERA_1['User Set Default']
                ,host='192.168.0.10',port=8501
            )
            print(self.request_pylon.check_connect())
            return self.request_pylon.check_connect()
        except:
            return False

    def check_plc_connection(self):
        # try:
        #     self.check = MyPLC()
        #     return self.check.connect()
        # except:
            return False

    def reconnect_mysql(self):
        """Thử kết nối lại MySQL."""
        print("Đang thử kết nối lại MySQL...")
        self.update_status(self.mysql_status, self.check_mysql_connection())

    def reconnect_camera(self):
        """Thử kết nối lại Camera."""
        print("Đang thử kết nối lại Camera...")
        self.update_status(self.camera_status, self.check_camera_connection())

    def reconnect_plc(self):
        """Thử kết nối lại PLC."""
        print("Đang thử kết nối lại PLC...")
        self.update_status(self.plc_status, self.check_plc_connection())

if __name__ == "__main__":
    app = ConnectionStatusGUI()
    app.mainloop()
# import keyboard
# import time

# Khởi tạo trạng thái phím ban đầu
# key_pressed = False

# while True: 
#     # Kiểm tra nếu phím 'shift+a' được nhấn và trạng thái là chưa in
#     if keyboard.is_pressed('shift+a') and not key_pressed:
#         print('aaaaa')
#         key_pressed = True  # Cập nhật trạng thái đã in ra

#     # Kiểm tra nếu phím 'shift+a' đã được thả ra để sẵn sàng in lại
#     if not keyboard.is_pressed('shift+a') and key_pressed:
#         key_pressed = False  # Cập nhật trạng thái để có thể in ra lại lần sau
     
#     # Tạm nghỉ ngắn để tránh sử dụng CPU cao
#     time.sleep(0.01)





# def handle_camera(window,values,model,plc_name,num_model,n_name_cam,value_plc_done,imgbuf1,imgbuf2,imgbuf3,imgbuf4,busy_c2,reset_counter):
#     global bien_ket_qua_1
#     global bien_ket_qua_2
#     global bien_ket_qua_3
#     global dem_cam_1
#     global dem_cam_2
#     global dem_lan 
#     global position_duplicate_2

#     # if read_value_from_file() == 1:
#     #     dem_cam_1 = 0
#     #     dem_cam_2 = 0
#     #     write_value_to_file(0) 
#     #     print('reset camera 2')
        

#     if n_name_cam == 1:
#         dem_cam_1 +=1
#     elif n_name_cam == 2:
#         dem_cam_2 +=1

#     # size1 = values[f'choose_size{num_model-1}']
#     # conf1 = values[f'conf_thres{num_model-1}']/100   
#     # size2 = values[f'choose_size{num_model}']
#     # conf2 = values[f'conf_thres{num_model}']/100   

#     size1 = values[f'choose_size{num_model-1}']
#     conf1 = values[f'conf_thres{num_model-1}']/100  

#     t1 = time.time()
#     img_orgin = imgbuf1.get()
#     img_orgin = preprocess(img_orgin)
#     # if (dem_cam_2 - 1) % 4 == 0: # 1 5 9
#     if dem_cam_2 == 1 or dem_cam_2 == 5  or dem_cam_2 == 8: 
#         names, show = handle_image(img_orgin,model[0],size1,conf1,num_model-1,values)
#         myresult = handle_result(window, model[0],names,show,num_model-1,plc_name,img_orgin,values)
#         t2 = time.time() - t1
#         time_cam = str(int(t2*1000)) + 'ms'
#         imgbytes = cv2.imencode('.png',show)[1].tobytes()
#         imgbuf2.put((time_cam,imgbytes,myresult,img_orgin,show,dem_cam_1,dem_cam_2))

#     # elif (dem_cam_2 - 3) % 4 == 0 : # 3 7 11
#     if dem_cam_2 == 3 or dem_cam_2 == 7  or dem_cam_2 == 9: 
#         names, show = handle_image(img_orgin,model[1],size1,conf1,num_model,values)
#         myresult = handle_result(window, model[1],names,show,num_model,plc_name,img_orgin, values)
        
#         if dem_cam_2 == 9:  
#             if bien_ket_qua_1 == [] and bien_ket_qua_2 == [] and bien_ket_qua_3 == []:
#                 register_ok1 = int(values[f'PLC_OK_{num_model-1}'])
#                 register_ok2 = int(values[f'PLC_OK_{num_model}'])
#                 register_ok3 = int(values[f'PLC_OK_{num_model+1}'])
#                 write_plc(plc_name,register_ok1,1)
#                 write_plc(plc_name,register_ok2,1)
#                 write_plc(plc_name,register_ok3,1)
#             else:
#                 if bien_ket_qua_1 != [] :
#                     for register_ng in bien_ket_qua_1:
#                         write_plc(plc_name,register_ng,1)
#                     bien_ket_qua_1 = []
#                 if bien_ket_qua_2 != [] :
#                     for register_ng in bien_ket_qua_2:
#                         write_plc(plc_name,register_ng,1)
#                     bien_ket_qua_2 = []
#                 if bien_ket_qua_3 != [] :
#                     for register_ng in bien_ket_qua_3:
#                         write_plc(plc_name,register_ng,1)
#                     bien_ket_qua_3 = []
            
#             write_plc(plc_name, value_plc_done, 1) 

#             print('hoan tat cam 2 lan: ',dem_lan)
#             dem_lan +=1
#             dem_cam_1 = 0
#             dem_cam_2 = 0

#         t2 = time.time() - t1
#         time_cam = str(int(t2*1000)) + 'ms'
#         imgbytes = cv2.imencode('.png',show)[1].tobytes()
#         imgbuf3.put((time_cam,imgbytes,myresult,img_orgin,show,dem_cam_1,dem_cam_2))

#     # elif dem_cam_2 % 2 == 0:
#     elif dem_cam_2 == 2 or dem_cam_2 == 4  or dem_cam_2 == 6: 

#         names, show = handle_image(img_orgin,model[2],size1,conf1,num_model+1,values)
#         myresult = handle_result(window, model[2],names,show,num_model+1,plc_name,img_orgin, values)

#         t2 = time.time() - t1
#         time_cam = str(int(t2*1000)) + 'ms'
#         imgbytes = cv2.imencode('.png',show)[1].tobytes()
#         imgbuf4.put((time_cam,imgbytes,myresult,img_orgin,show,dem_cam_1,dem_cam_2))
        