import tkinter as tk
import root_path
from tkinter import ttk
from PIL import Image, ImageTk  # Sử dụng để hiển thị dấu tích và dấu X
from base.ultilss.device_config import *
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

        self.check_img = ImageTk.PhotoImage(
            Image.open(r"base\Asserts\check.png").resize((20, 20))
        )
        self.cross_img = ImageTk.PhotoImage(
            Image.open(r"base\Asserts\delete.png").resize((20, 20))
        )
        self.reconnect_img = ImageTk.PhotoImage(
            Image.open(r"base\Asserts\link.png").resize((15, 15))
        )

        self.mysql_label, self.mysql_status, self.mysql_reconnect = (
            self.create_connection_row("Kết nối MySQL", self.reconnect_mysql, row=0)
        )
        self.camera_label, self.camera_status, self.camera_reconnect = (
            self.create_connection_row("Kết nối Camera", self.reconnect_camera, row=1)
        )
        self.plc_label, self.plc_status, self.plc_reconnect = (
            self.create_connection_row("Kết nối PLC", self.reconnect_plc, row=2)
        )

        self.check_connections()

    def create_connection_row(self, label_text, reconnect_command, row):

        label = ttk.Label(self, text=label_text, font=("Arial", 12))
        label.grid(row=row, column=0, sticky="w", padx=10, pady=5)

        status = ttk.Label(self)
        status.grid(row=row, column=1, padx=10, pady=5)

        reconnect_button = tk.Button(
            self, image=self.reconnect_img, command=reconnect_command, borderwidth=0
        )
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
                MYSQL_CONNECTION["HOST"],
                MYSQL_CONNECTION["ROOT"],
                MYSQL_CONNECTION["PASSWORD"],
                MYSQL_CONNECTION["DATABASE"],
            )
            return self.database.check_connection()
        except:
            return False

    def check_camera_connection(self):
        try:
            self.request_pylon = Basler_Pylon_xFunc(
                BASLER_UNITS_CAMERA_1["Serial number"],
                BASLER_UNITS_CAMERA_1["User Set Default"],
                host="192.168.0.10",
                port=8501,
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
