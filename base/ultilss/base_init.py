import sys
from pathlib import Path
import root_path
from base.ultilss.MySQL import *
from base.ultilss.plc_connect import *
from base.ultilss.constants import *
from typing import Any, Dict, List, Optional, Tuple, Union


class Base(PLC_Connection, setupTools, MySQL_Connection):

    def __init__(self, *args, **kwargs):
        self.database = MySQL_Connection(None, None, None, None)
        self.name_table: Optional[str] = None
        self.item_code_cfg: Optional[str] = None
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
        self.scale_conf_all: int = None
        self.size_model: int = None
        self.item_code = []
        self.make_cls_var: bool = False
        self.permisson_btn = []
        self.model = None
        self.time_processing_output = None
        self.result_detection = None
        self.datasets_format_model = None
        self.process_image_func = None
        self.processing_functions = {"HBB": self.run_func_hbb, "OBB": self.run_func_obb}
        self.tuple = self._get_lc_item()
        self.img_buffer = []
        self.counter = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.complete = "DM4006"
