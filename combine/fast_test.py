import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk  # Sử dụng để hiển thị dấu tích và dấu X
from base.device_config import *

class ConnectionStatusGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trạng thái kết nối")
        self.geometry("300x200")

        # Tạo các label để hiển thị trạng thái kết nối
        self.mysql_label = ttk.Label(self, text="Kết nối MySQL:")
        self.mysql_label.grid(row=0, column=0, sticky="w", padx=10, pady=10)

        self.camera_label = ttk.Label(self, text="Kết nối Camera:")
        self.camera_label.grid(row=1, column=0, sticky="w", padx=10, pady=10)

        self.plc_label = ttk.Label(self, text="Kết nối PLC:")
        self.plc_label.grid(row=2, column=0, sticky="w", padx=10, pady=10)

        # Load hình ảnh cho dấu tích xanh và dấu X đỏ
        self.check_img = ImageTk.PhotoImage(Image.open("check_green.png").resize((20, 20)))
        self.cross_img = ImageTk.PhotoImage(Image.open("cross_red.png").resize((20, 20)))

        # Label để hiển thị kết quả cho từng kết nối
        self.mysql_status = ttk.Label(self)
        self.mysql_status.grid(row=0, column=1)

        self.camera_status = ttk.Label(self)
        self.camera_status.grid(row=1, column=1)

        self.plc_status = ttk.Label(self)
        self.plc_status.grid(row=2, column=1)

        # Kiểm tra kết nối và cập nhật trạng thái
        self.check_connections()

    def check_connections(self):
        # Kiểm tra kết nối MySQL
        if self.check_mysql_connection():
            self.mysql_status.config(image=self.check_img)
        else:
            self.mysql_status.config(image=self.cross_img)

        # Kiểm tra kết nối Camera
        if self.check_camera_connection():
            self.camera_status.config(image=self.check_img)
        else:
            self.camera_status.config(image=self.cross_img)

        if self.check_plc_connection():
            self.plc_status.config(image=self.check_img)
        else:
            self.plc_status.config(image=self.cross_img)

    def check_mysql_connection(self):
        try:
            self.database = MySQL_Connection(MYSQL_CONNECTION['HOST'], MYSQL_CONNECTION['ROOT'], MYSQL_CONNECTION['PASSWORD'], MYSQL_CONNECTION['DATABASE'])
            return self.database.check_connect
        except:
            return False

    def check_camera_connection(self):
        try:
            self.request_pylon = Basler_Pylon_xFunc(BASLER_UNITS_CAMERA_1['Serial number'], BASLER_UNITS_CAMERA_1['User Set Default'])
            return True  
        except:
            return False

    def check_plc_connection(self):
        try:
            self.check = MyPLC()
            return True 
        except:
            return False

if __name__ == "__main__":
    app = ConnectionStatusGUI()
    app.mainloop()
