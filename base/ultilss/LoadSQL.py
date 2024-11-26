import sys
from pathlib import Path
import root_path
from base.ultilss.base_init import Base
from tkinter import messagebox, filedialog
import tkinter as tk


class LoadDatabase(Base):

    def __init__(self, *args, **kwargs):
        super(LoadDatabase, self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)

    def connect_database(self):
        cursor, db_connection = self.database.Connect_MySQLServer()
        check_connection = self.database.check_connection()
        reconnect = self.database.reconnect()
        return cursor, db_connection, check_connection, reconnect

    def check_connect_database(self):
        cursor, db_connection = self.database.Connect_MySQLServer()
        if cursor is not None and db_connection is not None:
            pass
        else:
            messagebox.showwarning("Warning", "Connection to database failed!")
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
                cursor.execute(
                    f"DELETE FROM {self.name_table} WHERE item_code = %s",
                    (item_code_value,),
                )
                for index in range(len(self.model_name_labels)):
                    values = self.get_values_for_insert(
                        index,
                        item_code_value,
                        weight,
                        confidence_all,
                        size_model,
                        dataset_format,
                    )
                    query_sql = self.build_insert_query(dataset_format)
                    cursor.execute(query_sql, values)

                db_connection.commit()
                messagebox.showinfo("Notification", "Saved parameters successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Data saved failed! Error: {str(e)}")

            finally:
                cursor.close()
                db_connection.close()

    def get_values_for_insert(
        self, index, item_code_value, weight, confidence_all, size_model, dataset_format
    ):
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

        if dataset_format == "OBB":
            rotage_min = float(self.rn_inputs[index].get())
            rotage_max = float(self.rx_inputs[index].get())
            return (
                item_code_value,
                weight,
                confidence_all,
                label_name,
                join_detect,
                OK_jont,
                NG_jont,
                num_labels,
                width_min,
                width_max,
                height_min,
                height_max,
                PLC_value,
                cmpnt_conf,
                size_model,
                rotage_min,
                rotage_max,
                dataset_format,
            )
        else:
            return (
                item_code_value,
                weight,
                confidence_all,
                label_name,
                join_detect,
                OK_jont,
                NG_jont,
                num_labels,
                width_min,
                width_max,
                height_min,
                height_max,
                PLC_value,
                cmpnt_conf,
                size_model,
                dataset_format,
            )

    def build_insert_query(self, dataset_format):

        if dataset_format == "OBB":
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

    def load_data_model(self):
        cursor, db_connection, _, _ = self.connect_database()
        cursor.execute(
            f"SELECT * FROM {self.name_table} WHERE item_code = %s",
            (self.item_code_cfg,),
        )
        records = cursor.fetchall()
        cursor.close()
        db_connection.close()
        if records:
            first_record = records[0]
            load_item_code = first_record["item_code"]
            load_path_weight = first_record["weight"]
            load_confidence_all_scale = first_record["confidence_all"]
            load_dataset_format = first_record["dataset_format"]
            size_model = first_record["size_detection"]
        return (
            records,
            load_path_weight,
            load_item_code,
            load_confidence_all_scale,
            load_dataset_format,
            size_model,
        )

    def load_parameters_model(
        self,
        initial_model,
        load_path_weight,
        load_item_code,
        load_confidence_all_scale,
        records,
        load_dataset_format,
        size_model,
        Frame,
    ):
        self._set_widget(self.datasets_format_model, load_dataset_format)
        self._set_widget(self.weights, load_path_weight)
        self._set_widget(self.item_code, load_item_code)
        self._set_intvalue(self.size_model, size_model)
        self._set_intvalue(self.scale_conf_all, load_confidence_all_scale)
        try:
            if load_dataset_format == "HBB":
                self.process_func_local(load_dataset_format)
                self._clear_widget(Frame)
                self.option_layout_parameters(Frame, self.model)
                for index, _ in enumerate(initial_model.names):
                    for record in records:
                        if record["label_name"] == initial_model.names[index]:
                            self._set_intvalue(
                                self.join[index], bool(record["join_detect"])
                            )
                            self._set_intvalue(self.ok_vars[index], bool(record["OK"]))
                            self._set_intvalue(self.ng_vars[index], bool(record["NG"]))
                            self._set_widget(
                                self.num_inputs[index], record["num_labels"]
                            )
                            self._set_widget(self.wn_inputs[index], record["width_min"])
                            self._set_widget(self.wx_inputs[index], record["width_max"])
                            self._set_widget(
                                self.hn_inputs[index], record["height_min"]
                            )
                            self._set_widget(
                                self.hx_inputs[index], record["height_max"]
                            )
                            self._set_widget(
                                self.plc_inputs[index], record["PLC_value"]
                            )
                            self._set_intvalue(
                                self.conf_scales[index], record["cmpnt_conf"]
                            )

            elif load_dataset_format == "OBB":
                self.process_func_local(load_dataset_format)
                self._clear_widget(Frame)
                self.option_layout_parameters_orient_bounding_box(Frame, self.model)
                for index, _ in enumerate(initial_model.names):
                    for record in records:
                        if record["label_name"] == initial_model.names[index]:
                            self._set_intvalue(
                                self.join[index], bool(record["join_detect"])
                            )
                            self._set_intvalue(self.ok_vars[index], bool(record["OK"]))
                            self._set_intvalue(self.ng_vars[index], bool(record["NG"]))
                            self._set_widget(
                                self.num_inputs[index], record["num_labels"]
                            )
                            self._set_widget(self.wn_inputs[index], record["width_min"])
                            self._set_widget(self.wx_inputs[index], record["width_max"])
                            self._set_widget(
                                self.hn_inputs[index], record["height_min"]
                            )
                            self._set_widget(
                                self.hx_inputs[index], record["height_max"]
                            )
                            self._set_widget(
                                self.plc_inputs[index], record["PLC_value"]
                            )
                            self._set_intvalue(
                                self.conf_scales[index], record["cmpnt_conf"]
                            )
                            self._set_widget(
                                self.rn_inputs[index], record["rotage_min"]
                            )
                            self._set_widget(
                                self.rx_inputs[index], record["rotage_max"]
                            )
                self._default_settings()
        except IndexError as e:
            messagebox.showerror("Error", f"Load parameters failed! Error: {str(e)}")

    def change_model(self, Frame):
        selected_file = filedialog.askopenfilename(
            title="Choose a file", filetypes=[("Model Files", "*.pt")]
        )
        if selected_file:
            self._set_widget(self.weights, selected_file)
            self.model = self.torch_load_nodemap(source=selected_file)
            self.confirm_dataset_format(Frame)
        else:
            messagebox.showinfo(
                "Notification", "Please select the correct training file!"
            )
            pass

    def confirm_dataset_format(self, Frame):
        if self.datasets_format_model.get() == "OBB":
            for widget in Frame.grid_slaves():
                widget.grid_forget()
            self.option_layout_parameters_orient_bounding_box(Frame, self.model)
        elif self.datasets_format_model.get() == "HBB":
            for widget in Frame.grid_slaves():
                widget.grid_forget()
            self.option_layout_parameters(Frame, self.model)

    def _set_widget(self, widget, value):
        widget.delete(0, tk.END)
        widget.insert(0, value)

    def _set_intvalue(self, int_widget, value):
        int_widget.set(value)

    def _clear_widget(self, Frame):
        for widget in Frame.grid_slaves():
            widget.grid_forget()

    def load_params_child(self):
        cursor, db_connection, _, _ = self.connect_database()
        try:
            cursor.execute(
                f"SELECT * FROM {self.name_table} WHERE item_code = %s",
                (self.item_code.get().__str__(),),
            )
        except Exception as e:
            messagebox.showwarning("Warning", f"{e}: Item Code does not exist")
        records = cursor.fetchall()
        model = self.torch_load_nodemap(source=self.weights.get())
        cursor.close()
        db_connection.close()
        return records, model

    def load_parameters_from_weight(self, records):
        confirm_load_parameters = messagebox.askokcancel(
            "Confirm", "Are you sure you want to load the parameters?"
        )
        if confirm_load_parameters:
            records, initial_model = self.load_params_child()
            try:
                if self.datasets_format_model.get() == "HBB":
                    for index, _ in enumerate(initial_model.names):
                        for record in records:
                            if record["label_name"] == initial_model.names[index]:
                                self._set_intvalue(
                                    self.join[index], bool(record["join_detect"])
                                )
                                self._set_intvalue(
                                    self.ok_vars[index], bool(record["OK"])
                                )
                                self._set_intvalue(
                                    self.ng_vars[index], bool(record["NG"])
                                )
                                self._set_widget(
                                    self.num_inputs[index], record["num_labels"]
                                )
                                self._set_widget(
                                    self.wn_inputs[index], record["width_min"]
                                )
                                self._set_widget(
                                    self.wx_inputs[index], record["width_max"]
                                )
                                self._set_widget(
                                    self.hn_inputs[index], record["height_min"]
                                )
                                self._set_widget(
                                    self.hx_inputs[index], record["height_max"]
                                )
                                self._set_widget(
                                    self.plc_inputs[index], record["PLC_value"]
                                )
                                self._set_intvalue(
                                    self.conf_scales[index], record["cmpnt_conf"]
                                )
                elif self.datasets_format_model.get() == "OBB":
                    for index, _ in enumerate(initial_model.names):
                        for record in records:
                            if record["label_name"] == initial_model.names[index]:
                                self._set_intvalue(
                                    self.join[index], bool(record["join_detect"])
                                )
                                self._set_intvalue(
                                    self.ok_vars[index], bool(record["OK"])
                                )
                                self._set_intvalue(
                                    self.ng_vars[index], bool(record["NG"])
                                )
                                self._set_widget(
                                    self.num_inputs[index], record["num_labels"]
                                )
                                self._set_widget(
                                    self.wn_inputs[index], record["width_min"]
                                )
                                self._set_widget(
                                    self.wx_inputs[index], record["width_max"]
                                )
                                self._set_widget(
                                    self.hn_inputs[index], record["height_min"]
                                )
                                self._set_widget(
                                    self.hx_inputs[index], record["height_max"]
                                )
                                self._set_widget(
                                    self.plc_inputs[index], record["PLC_value"]
                                )
                                self._set_intvalue(
                                    self.conf_scales[index], record["cmpnt_conf"]
                                )
                                self._set_widget(
                                    self.rn_inputs[index], record["rotage_min"]
                                )
                                self._set_widget(
                                    self.rx_inputs[index], record["rotage_max"]
                                )
            except IndexError as e:
                messagebox.showerror(
                    "Error", f"Load parameters failed! Error: {str(e)}"
                )
