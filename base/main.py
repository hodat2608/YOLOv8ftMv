import root_path
import tkinter as tk
from base.train import *
from base.extract_vid import *
from tkinter import ttk
from base.ultils import removefile
from tkinter import *
from tkinter import *
from base.labling import *
from base.menu_cfg import *
from base.model_1_config import *
from base.ultilss.config import *

__appname__ = "YOLOv8.2.0 by Utralytics"


class main:
    def __init__(self):
        disable_ctrl_c()
        asserts = Asserts()
        global menubar
        window = tk.Tk()
        window.title(__appname__)
        window.state("zoomed")
        style = ttk.Style()
        style.configure(
            "TNotebook",
            background="white",
            foreground="black",
            font=("Arial", 10, "bold"),
        )
        style.configure("TFrame", background="white", foreground="black")
        notebook = ttk.Notebook(window)
        notebook.pack(fill="both", expand=True)
        menubar = Menu(window)
        filemenu = Menu(menubar, tearoff=0)
        Open = asserts.icon_open_display_cam
        exit = asserts.exit
        filemenu.add_command(
            label=" Open Camera Display",
            image=Open,
            compound=tk.LEFT,
            command=lambda: display_layout(notebook, window),
        )
        filemenu.add_separator()
        filemenu.add_command(
            label=" Exit",
            image=exit,
            compound=tk.LEFT,
            command=lambda: confirm_exit(window),
        )
        menubar.add_cascade(label="Open", menu=filemenu)
        View = Menu(menubar, tearoff=0)
        View.add_command(label="Real-Time Integration", command=donothing)
        View.add_command(
            label="Extract Output", command=lambda: video(notebook, window)
        )
        menubar.add_cascade(label="View", menu=View)
        Configuration = Menu(menubar, tearoff=0)
        IO_connection(Configuration)
        menubar.add_cascade(label="I/O Configuration", menu=Configuration)
        train = Menu(menubar, tearoff=0)
        open_label_img_menu(train)
        train.add_command(
            label="Train Dataset", command=lambda: training_data(notebook, window)
        )
        menubar.add_cascade(label="Tool", menu=train)
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About...", command=donothing)
        menubar.add_cascade(label="Help", menu=helpmenu)
        filemenu.image = Open
        filemenu.image = exit
        window.config(menu=menubar)
        create_context_menu(notebook)
        window.protocol("WM_DELETE_WINDOW", lambda: confirm_exit(window))
        window.mainloop()


if __name__ == "__main__":
    main()
