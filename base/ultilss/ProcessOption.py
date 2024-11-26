import sys
from pathlib import Path
import root_path
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, filedialog
import glob
from base.ultilss.base_init import *
from base.ultilss.LoadSQL import *
from base.ultilss.ProcessModel import *
import threading


class CustomProcessModel(ProcessingModelType):
    def __init__(self, *args, **kwargs):
        super(CustomProcessModel, self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)

    def handle_image(self, img1_orgin, width, height, camera_frame):
        for widget in camera_frame.winfo_children():
            widget.destroy()
        image_result, lst_result, label_ng, _, time_processing = (
            self.process_image_func(img1_orgin, width, height)
        )
        img_pil = Image.fromarray(image_result)
        photo = ImageTk.PhotoImage(img_pil)
        canvas = tk.Canvas(camera_frame, width=width, height=height)
        canvas.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo
        canvas.create_text(
            10,
            10,
            anchor=tk.NW,
            text=f"Time: {time_processing}",
            fill="black",
            font=("Segoe UI", 20),
        )
        canvas.create_text(
            10,
            40,
            anchor=tk.NW,
            text=f"Result: {lst_result}",
            fill="green" if lst_result == "OK" else "red",
            font=("Segoe UI", 20),
        )
        if not label_ng:
            canvas.create_text(
                10, 70, anchor=tk.NW, text=f" ", fill="green", font=("Segoe UI", 20)
            )
        else:
            label_ng = ",".join(label_ng)
            canvas.create_text(
                10,
                70,
                anchor=tk.NW,
                text=f"NG: {label_ng}",
                fill="red",
                font=("Segoe UI", 20),
            )
        return lst_result

    def detect_single_img(self, camera_frame):
        selected_file = filedialog.askopenfilename(
            title="Choose a file", filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if selected_file:
            for widget in camera_frame.winfo_children():
                widget.destroy()
            width = 480
            height = 450
            self.handle_image(selected_file, width, height, camera_frame)
        else:
            pass

    def detect_multi_img(self, camera_frame):
        selected_folder = filedialog.askdirectory(title="Choose a folder")
        if selected_folder:
            self.image_files = [
                os.path.join(selected_folder, f)
                for f in os.listdir(selected_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            self.current_image_index = 0
            if self.image_files:
                for widget in camera_frame.winfo_children():
                    widget.destroy()
                self.show_image(self.current_image_index, camera_frame)
            else:
                messagebox.showinfo(
                    "No Images", "The selected folder contains no images."
                )
        else:
            pass

    def show_image(self, index, camera_frame):
        width = 480
        height = 450
        image_path = self.image_files[index]
        self.handle_image(image_path, width, height, camera_frame)

    def detect_next_img(self, camera_frame):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_image(self.current_image_index, camera_frame)
        else:
            messagebox.showinfo("End of Images", "No more images in the folder.")

    def detect_previos_img(self, camera_frame):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.current_image_index, camera_frame)
        else:
            messagebox.showinfo(
                "Start of Images", "This is the first image in the folder."
            )

    def detect_auto(self, camera_frame):
        selected_folder_original = filedialog.askdirectory(title="Choose a folder")
        if selected_folder_original:
            selected_folder = glob.glob(selected_folder_original + "/*.jpg")
            if not selected_folder:
                pass
            self.image_index = 0
            self.selected_folder_detect_auto = selected_folder
            self.camera_frame = camera_frame
            self.image_path_mks_cls = []
            self.lst_result = None

            def process_next_image():
                if self.image_index < len(self.selected_folder_detect_auto):
                    self.image_path_mks_cls = self.selected_folder_detect_auto[
                        self.image_index
                    ]
                    width = 480
                    height = 450
                    self.lst_result = self.handle_image(
                        self.image_path_mks_cls, width, height, camera_frame
                    )
                    self.image_index += 1
                    self.camera_frame.after(500, process_next_image)

            process_next_image()
        else:
            pass

    def logging(
        self,
        folder_ok,
        folder_ng,
        logging_ok_checkbox_var,
        logging_ng_checkbox_var,
        camera_frame,
        percent_entry,
        logging_frame,
    ):
        selected_folder_original = filedialog.askdirectory(title="Choose a folder")
        selected_folder = glob.glob(os.path.join(selected_folder_original, "*.jpg"))
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
                    if lst_result == "OK":
                        if logging_ok_checkbox_var.get():
                            shutil.move(img, os.path.join(folder_ok.get(), basename))
                    else:
                        if logging_ng_checkbox_var.get():
                            shutil.move(img, os.path.join(folder_ng.get(), basename))
                    self.logging_frame.after(
                        10, update_progress, self.image_index, total_images
                    )
                    self.image_index += 1
                else:
                    messagebox.showinfo(
                        "End of Images", "No more images in the folder."
                    )
                    break
            self.percent_entry.delete(0, tk.END)
            self.percent_entry.insert(0, "0%")

        threading.Thread(target=process_images).start()

    def toggle_state_layout_model(self):
        if self.state == 1:
            password = simpledialog.askstring(
                "Administrator", "Enter password:", show="*"
            )
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

    def toggle_widgets_state(self, state):
        for widget in self.lockable_widgets:
            widget.config(state=state)

    def toggle_state_option_layout_parameters(self):
        for widget in self.lock_params:
            widget.config(state="disabled")

    def pick_folder_ok(self, folder_ok):
        file_path = filedialog.askdirectory(title="Choose a folder")
        if file_path:
            folder_ok.delete(0, tk.END)
            folder_ok.insert(0, file_path)

    def pick_folder_ng(self, folder_ng):
        file_path = filedialog.askdirectory(title="Choose a folder")
        if file_path:
            folder_ng.delete(0, tk.END)
            folder_ng.insert(0, file_path)

    def load_first_img(self):
        filename = r"C:\ultralytics-main\2024-03-05_00-01-31-398585-C1.jpg"
        self.model(filename, imgsz=608, conf=0.2)
        print("Load model successfully")
