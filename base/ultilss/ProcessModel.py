import sys
from pathlib import Path
import root_path
from ultralytics import YOLO
from base.ultilss.base_init import *
from base.ultilss.LoadSQL import *
import math
from pathlib import Path
from collections import Counter


class ProcessingModelType(LoadDatabase):

    def __init__(self, *args, **kwargs):
        super(ProcessingModelType, self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)

    def torch_load_nodemap(self, source=None, task=None, device=None):
        return YOLO(model=source, task=task).to(device=device)

    def process_func_local(self, selected_format):
        self.process_image_func = self.processing_functions.get(selected_format, None)

    def preprocess(self, img):
        if isinstance(img, str):
            image_array = cv2.imread(img)
            fh = True
            return image_array, fh
        elif isinstance(img, np.ndarray):
            fh = False
            image_array = img
            imgRGB = cv2.cvtColor(image_array, cv2.COLOR_BayerGB2RGB)
            imgRGB = cv2.flip(imgRGB, 0)
            return cv2.cvtColor(imgRGB, cv2.COLOR_BGR2RGB), fh

    def run_func_hbb(self, input_image, width, height):
        current_time = time.time()
        size_model_all = int(self.size_model.get())
        conf_all = int(self.scale_conf_all.get()) / 100
        results = self.model(input_image, imgsz=size_model_all, conf=conf_all)
        settings_dict = {
            setting["label_name"]: setting for setting in self.default_model_params
        }
        boxes_dict = results[0].boxes.cpu().numpy()
        xywh_list = boxes_dict.xywh.tolist()
        cls_list = boxes_dict.cls.tolist()
        conf_list = boxes_dict.conf.tolist()
        _valid_idex, _invalid_idex, list_cls_ng, _flag, lst_result = (
            [],
            [],
            [],
            False,
            "ERROR",
        )
        for index, (xywh, cls, conf) in enumerate(
            reversed(list(zip(xywh_list, cls_list, conf_list)))
        ):
            setting = settings_dict[results[0].names[int(cls)]]
            if setting:
                if setting["join_detect"]:
                    if (
                        xywh[2] < setting["width_min"]
                        or xywh[2] > setting["width_max"]
                        or xywh[3] < setting["height_min"]
                        or xywh[3] > setting["height_max"]
                        or int(conf * 100) < setting["cmpnt_conf"]
                    ):
                        _invalid_idex.append(int(index))
                        continue
                    _valid_idex.append(results[0].names[int(cls)])
                else:
                    _invalid_idex.append(int(index))
        for model_name, setting in settings_dict.items():
            if setting["join_detect"] and setting["OK_jont"]:
                if _valid_idex.count(setting["label_name"]) != setting["num_labels"]:
                    lst_result, _flag = "NG", True
                    list_cls_ng.append(model_name)
            if setting["join_detect"] and setting["NG_jont"]:
                if model_name in _valid_idex:
                    lst_result, _flag = "NG", True
                    list_cls_ng.append(setting["label_name"])
        if not _flag:
            lst_result = "OK"
        if self.make_cls_var.get():
            self._make_cls(input_image, results, self.default_model_params)
        show_img = np.squeeze(results[0].extract_npy(_invalid_idex=_invalid_idex))
        show_img = cv2.resize(show_img, (width, height), interpolation=cv2.INTER_AREA)
        output_image = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        lst_check_location, time_processing = (
            None,
            f"{str(int((time.time()-current_time)*1000))}ms",
        )
        return (
            output_image,
            lst_result,
            list_cls_ng,
            lst_check_location,
            time_processing,
        )

    def run_func_obb(self, source, width, height):
        current_time = time.time()
        self.counter += 1
        size_model_all = int(self.size_model.get())
        conf_all = int(self.scale_conf_all.get()) / 100
        '''
        source, fh = self.preprocess(source)
        '''
        results = self.model(source, imgsz=size_model_all, conf=conf_all)
        self._default_settings()
        settings_dict = {
            setting["label_name"]: setting for setting in self.default_model_params
        }
        _xywhr, _cls, _conf = (
            results[0].obb.cpu().numpy().xywhr.tolist(),
            results[0].obb.cpu().numpy().cls.tolist(),
            results[0].obb.cpu().numpy().conf.tolist(),
        )
        (
            _valid_idex,
            _invalid_idex,
            list_cls_ng,
            _flag,
            lst_result,
            lst_check_location,
            dictionary,
        ) = ([], [], [], False, "ERROR", [],[])
        for index, (xywhr, cls, conf) in enumerate(
            reversed(list(zip(_xywhr, _cls, _conf)))
        ):
            setting = settings_dict[results[0].names[int(cls)]]
            if setting:
                if setting["join_detect"]:
                    dictionary.append(
                        (xywhr[0], xywhr[1], xywhr[2], xywhr[3], int(cls))
                    )
                    if (
                        (
                            xywhr[2] < setting["width_min"]
                            or xywhr[2] > setting["width_max"]
                        )
                        or (
                            xywhr[3] < setting["height_min"]
                            or xywhr[3] > setting["height_max"]
                        )
                        or (int(conf * 100) < setting["cmpnt_conf"])
                    ):
                        _invalid_idex.append(int(index))
                    try:
                        if LOCALTION_OBJS:
                            list_cls_ng, _invalid_idex, lst_check_location, _flag = (
                                self._bbox_localtion_direction_objs(
                                    _flag,
                                    index,
                                    setting,
                                    xywhr,
                                    list_cls_ng,
                                    _invalid_idex,
                                    lst_check_location,
                                )
                            )
                    except:
                        pass
                    _valid_idex.append(results[0].names[int(cls)])
                else:
                    _invalid_idex.append(int(index))
        for index, (xywhr, cls, conf) in enumerate(
            reversed(list(zip(_xywhr, _cls, _conf)))
        ):
            setting = settings_dict[results[0].names[int(cls)]]
            if setting:
                if setting["join_detect"] and setting["OK_jont"]:
                    if (
                        Counter(_valid_idex)[setting["label_name"]]
                        != setting["num_labels"]
                    ):
                        _flag = True
                        list_cls_ng.append(setting["label_name"])
                if setting["join_detect"] and setting["NG_jont"]:
                    if (
                        setting["label_name"]
                        or results[0].names[int(cls)] in _valid_idex
                    ):
                        _flag = True
                        list_cls_ng.append(setting["label_name"])
                        _invalid_idex.append(int(index))
        lst_result = "OK" if not _flag else "NG"
        '''
        if not fh:
            if self.counter == 6:
                self.writedata(self.socket, self.complete, 1)
                self.counter = 0
        '''
        if self.make_cls_var.get():
            self.xyxyxyxy2xywhr_indirect(
                source, results[0], _xywhr, _cls, _conf, self.default_model_params
            )
        show_img = np.squeeze(results[0]._plot(_invalid_idex=_invalid_idex))
        show_img = np.squeeze(results[0]._export(dictionary=dictionary))
        show_img = cv2.resize(show_img, (width, height), interpolation=cv2.INTER_AREA)
        output_image = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        lst_check_location = sorted(lst_check_location, key=lambda item: item[0])
        lst_check_location = lst_check_location if lst_check_location != [] else []
        time_processing = f"{str(int((time.time()-current_time)*1000))}ms"
        return (
            output_image,
            lst_result,
            list_cls_ng,
            lst_check_location,
            time_processing,
        )

    def _bbox_localtion_direction_objs(
        self,
        _flag,
        index,
        setting,
        xywhr,
        list_cls_ng,
        _invalid_idex,
        lst_check_location,
    ):
        if OBJECTS_ANGLE:
            if setting["label_name"] in ITEM:
                radian = float(round(math.degrees(xywhr[4]), 1))
                if (radian < setting["rotage_min"]) or (radian > setting["rotage_max"]):
                    _flag = True
                    list_cls_ng.append(setting["label_name"])
                    _invalid_idex.append(int(index))
        if OBJECTS_COORDINATES:
            if setting["label_name"] == ITEM[0]:
                if xywhr[0] and xywhr[1]:
                    c = self._get_lc_object(
                        self.tuple, xywhr[0], xywhr[1]
                    )
                    d = self._get_lc_state(
                        c[0],c[1][0],c[1][1],c[2][0],c[2][1]
                    )
                    lst_check_location.append(
                        (d[0],d[1][0],d[1][1],d[2][0],d[2][1], 
                        round(math.degrees(xywhr[4]),1),
                        'OK' if d[3][0] else 'NG',c[3][0],ITEM[0])
                    )
                    if not d[3][0]:
                        _flag = True
                else:
                    _flag = True
        return list_cls_ng, _invalid_idex, lst_check_location, _flag

    def _default_settings(self):
        self.default_model_params = [
            {
                "label_name": label.cget("text"),
                "join_detect": self.join[index].get(),
                "OK_jont": self.ok_vars[index].get(),
                "NG_jont": self.ng_vars[index].get(),
                "num_labels": int(self.num_inputs[index].get()),
                "width_min": int(self.wn_inputs[index].get()),
                "width_max": int(self.wx_inputs[index].get()),
                "height_min": int(self.hn_inputs[index].get()),
                "height_max": int(self.hx_inputs[index].get()),
                "PLC_value": int(self.plc_inputs[index].get()),
                "cmpnt_conf": int(self.conf_scales[index].get()),
                "rotage_min": float(self.rn_inputs[index].get()),
                "rotage_max": float(self.rx_inputs[index].get()),
            }
            for index, label in enumerate(self.model_name_labels)
        ]

    def _default_settings_h(self):
        self.default_model_params_h = [
            {
                "label_name": label.cget("text"),
                "join_detect": self.join[index].get(),
                "OK_jont": self.ok_vars[index].get(),
                "NG_jont": self.ng_vars[index].get(),
                "num_labels": int(self.num_inputs[index].get()),
                "width_min": int(self.wn_inputs[index].get()),
                "width_max": int(self.wx_inputs[index].get()),
                "height_min": int(self.hn_inputs[index].get()),
                "height_max": int(self.hx_inputs[index].get()),
                "PLC_value": int(self.plc_inputs[index].get()),
                "cmpnt_conf": int(self.conf_scales[index].get()),
            }
            for index, label in enumerate(self.model_name_labels)
        ]

    def _make_cls(self, image_path_mks_cls, results, model_settings):
        ims = cv2.imread(image_path_mks_cls)
        w, h, _ = ims.shape
        with open(image_path_mks_cls[:-3] + "txt", "a") as file:
            for params in results.xywhn:
                params = params.tolist()
                for item in range(len(params)):
                    param = params[item]
                    param = [round(i, 6) for i in param]
                    number_label = int(param[5])
                    conf_result = float(param[4])
                    width_result = float(param[2]) * w
                    height_result = float(param[3]) * h
                    for setting in model_settings:
                        if results.names[int(number_label)] == setting["label_name"]:
                            if setting["join_detect"]:
                                if (
                                    width_result < setting["width_min"]
                                    or width_result > setting["width_max"]
                                    or height_result < setting["height_min"]
                                    or height_result > setting["height_max"]
                                    or conf_result < setting["cmpnt_conf"]
                                ):
                                    formatted_values = [
                                        "{:.6f}".format(value) for value in param[:4]
                                    ]
                                    output_line = "{} {}\n".format(
                                        str(number_label), " ".join(formatted_values)
                                    )
                                    file.write(output_line)
        path = Path(image_path_mks_cls).parent
        path = os.path.join(path, "classes.txt")
        with open(path, "w") as file:
            for index in range(len(results.names)):
                file.write(str(results.names[index]) + "\n")

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
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:

            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * np.clip(
            inter_rect_y2 - inter_rect_y1 + 1, 0, None
        )
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        return iou
    
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
        dx = width / 2
        dy = height / 2

        angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float64))

        rotation_matrix = torch.tensor(
            [
                [torch.cos(angle_rad), -torch.sin(angle_rad)],
                [torch.sin(angle_rad),  torch.cos(angle_rad)],
            ],
            dtype=torch.float64,
        )
        corners = torch.tensor([
                [-dx, -dy],
                [ dx, -dy],
                [ dx,  dy],
                [-dx,  dy],
        ], dtype=torch.float64)

        rotated_corners = torch.matmul(corners, rotation_matrix)

        final_corners = rotated_corners + torch.tensor([x_center, y_center], dtype=torch.float64)

        normalized_corners = final_corners / torch.tensor([im_width, im_height], dtype=torch.float64)
        
        return [int(class_id)] + normalized_corners.flatten().tolist()

    def transpose_matrix(
        self,
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
        angle_rad = np.deg2rad(angle)
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
        corners = np.array(
            [
                [-half_width, -half_height],
                [half_width, -half_height],
                [half_width, half_height],
                [-half_width, half_height],
            ]
        )
        rotated_corners = np.dot(corners, rotation_matrix)
        final_corners = rotated_corners + np.array([x_center, y_center])
        normalized_corners = final_corners / np.array([im_width, im_height])
        return [int(class_id)] + normalized_corners.flatten().tolist()

    def transfer_parameter(self, des_path, progress_label):
        input_folder = des_path
        os.makedirs(os.path.join(input_folder, "instance"), exist_ok=True)
        output_folder = os.path.join(input_folder, "instance")
        total_fl = len(des_path)
        for index, txt_file in enumerate(os.listdir(input_folder)):
            if txt_file.endswith(".txt"):
                if txt_file == "classes.txt":
                    continue
                input_path = os.path.join(input_folder, txt_file)
                im = cv2.imread(f"{input_path[:-3]}jpg")
                im_height, im_width, _ = im.shape
                output_path = os.path.join(output_folder, txt_file)
                with open(input_path, "r") as file:
                    lines = file.readlines()
                with open(output_path, "w") as out_file:
                    for line in lines:
                        line = line.strip()
                        if "YOLO_OBB" in line:
                            continue
                        params = list(map(float, line.split()))
                        class_id, x_center, y_center, width, height, angle = params
                        converted_label = self.transpose_matrix(
                            class_id=class_id,
                            x_center=x_center,
                            y_center=y_center,
                            width=width,
                            height=height,
                            angle=angle,
                            im_height=im_height,
                            im_width=im_width,
                        )
                        out_file.write(" ".join(map(str, converted_label)) + "\n")
                progress_retail = (index + 1) / total_fl * 100
                progress_label.config(
                    text=f"Converting Format...{progress_retail:.2f}%"
                )
                progress_label.update_idletasks()
                os.replace(output_path, input_path)
        shutil.rmtree(output_folder)

    def transfer_parameter_dataset(self, des_path, progress_label):
        input_folder = des_path
        os.makedirs(os.path.join(input_folder, "instance"), exist_ok=True)
        output_folder = os.path.join(input_folder, "instance")
        
        txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt") and f != "classes.txt"]
        total_fl = len(txt_files)
        
        for index, txt_file in enumerate(txt_files):
            input_path = os.path.join(input_folder, txt_file)
            im = cv2.imread(f"{input_path[:-3]}jpg")
            im_height, im_width, _ = im.shape
            output_path = os.path.join(output_folder, txt_file)
            
            with open(input_path, "r") as file:
                lines = file.readlines()
            
            with open(output_path, "w") as out_file:
                for line in lines:
                    line = line.strip()
                    if "YOLO_OBB" in line:
                        continue
                    params = list(map(float, line.split()))
                    class_id, x_center, y_center, width, height, angle = params
                    converted_label = self.transpose_matrix(
                            class_id=class_id,
                            x_center=x_center,
                            y_center=y_center,
                            width=width,
                            height=height,
                            angle=angle,
                            im_height=im_height,
                            im_width=im_width,
                        )
                    out_file.write(" ".join(map(str, converted_label)) + "\n")
            
            progress_retail = (index + 1) / total_fl * 100
            progress_label.config(
                text=f"Converting Format : {progress_retail:.2f}%"
            )
            progress_label.update_idletasks()
            os.replace(output_path, input_path)

        shutil.rmtree(output_folder)

    def xyxyxyxy2xywhr_indirect(
        self, input_image, results, xywhr_list, cls_list, conf_list, model_settings):
        settings_dict = {setting["label_name"]: setting for setting in model_settings}
        with open(input_image[:-3] + "txt", "a") as out_file:
            out_file.write("YOLO_OBB\n")
            for index, (xywhr, cls, conf) in enumerate(
                reversed(list(zip(xywhr_list, cls_list, conf_list)))
            ):
                setting = settings_dict[results.names[int(cls)]]
                xywhr_list[index][-1] = math.degrees(xywhr_list[index][-1])
                if xywhr_list[index][-1] > self._right_angle:
                    xywhr_list[index][-1] = self.right_angle - xywhr_list[index][-1]
                else:
                    xywhr_list[index][-1] = -(xywhr_list[index][-1])
                line = [int(cls_list[index])] + xywhr_list[index]
                formatted_line = " ".join(
                    [
                        "{:.6f}".format(x) if isinstance(x, float) else str(x)
                        for x in line
                    ]
                )
                if setting:
                    if setting["join_detect"]:
                        if (
                            xywhr[2] < setting["width_min"]
                            or xywhr[2] > setting["width_max"]
                            or xywhr[3] < setting["height_min"]
                            or xywhr[3] > setting["height_max"]
                            or int(conf * 100) < setting["cmpnt_conf"]
                        ):
                            continue
                        out_file.write(f"{formatted_line}\n")
        path = Path(input_image).parent
        path = os.path.join(path, "classes.txt")
        with open(path, "w") as file:
            for i in range(len(results.names)):
                file.write(f"{str(results.names[i])}\n")
