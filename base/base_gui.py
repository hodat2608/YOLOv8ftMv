
class Base_GUI: 
    def __init__(self,*args, **kwargs): 
        super(Base_GUI, self).__init__(*args, **kwargs)
        self.image_files = []
        self.current_image_index = -1
        self.state = 1
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
        self.cls = False
        self.img_frame = None


# self.image_files = []
# self.current_image_index = -1
# self.state = 1
# self.lockable_widgets = []
# self.lock_params = []
# self.model_name_labels = []
# self.join = []
# self.ok_vars = []
# self.ng_vars = []
# self.num_inputs = []
# self.wn_inputs = []
# self.wx_inputs = []
# self.hn_inputs = []
# self.hx_inputs = []
# self.plc_inputs = []
# self.conf_scales = []
# self.rn_inputs = []
# self.rx_inputs = []
# self.rotage_join = []
# self.widgets_option_layout_parameters = []
# self.row_widgets = []
# self.weights = []
# self.datasets_format_model = []
# self.scale_conf_all = None
# self.size_model = None
# self.item_code = []
# self.make_cls_var = False
# self.permisson_btn = []
# self.model = None
# self.time_processing_output = None
# self.result_detection = None
# self.cls = False
# self.img_frame = None