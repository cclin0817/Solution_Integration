
✅ 還原內容如下：

import tkinter as tk
from tkinter import simpledialog, Scrollbar, Scale, HORIZONTAL, VERTICAL, Menu, filedialog, ttk, font
from tkinter import Toplevel, StringVar, OptionMenu, Button, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import networkx as nx
import tkinter.font as tkFont
from .canvas import ZoomableCanvas
from .block import *
from .obj import *
from .functions import *
import colorsys
from .qor_checker import QoRChecker
from .base_utility import *
from .pclamp_utility import *
from .sensor_utility import *
from .bus_utility import *
from .gpio_utility import *
from .clock_utility import ClockUtility
from PIL import Image, ImageOps, ImageDraw, ImageFont

class Application(tk.Frame):
    def __init__(self, master=None, input_json=None, output_dir=None, args=None):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        self.output_dir = output_dir
        self.create_widgets()
        self.import_json(input_json)
        self.canvas_setup()
        self.args = args
        self.current_utility = None

    def switch_to_pclamp(self):
        self.load_utility(PclampUtility)

    def switch_to_sensor(self):
        self.load_utility(SensorUtility)

    def switch_to_bus(self):
        self.load_utility(BusUtility)

    def switch_to_clock(self):
        self.load_utility(ClockUtility)

    def switch_to_gpio(self):
        self.load_utility(GPIOUtility)

    def switch_to_qor(self):
        # Create QoRChecker instance
        self.load_utility(QoRChecker)

    def load_utility(self, UtilityClass):
        for widget in self.utility_frame.winfo_children():
            widget.destroy()
        self.current_utility = UtilityClass(self, self.canvas, self.args)
        self.current_utility.create_widgets(self.utility_frame)
        self.canvas.fit()

    def update_utility(self, chiplet):
        if self.current_utility is None:
            return
        self.current_utility.update_widgets(self.utility_frame, chiplet=chiplet)

    def show_selection_screen(self):
        for widget in self.utility_frame.winfo_children():
            widget.destroy()
        self.current_utility = None
        self.switch_pclamp_button = tk.Button(self.utility_frame, text="Switch to pclamp planning utility", command=self.switch_to_pclamp)
        self.switch_pclamp_button.pack(fill="x")

        self.switch_sensor_button = tk.Button(self.utility_frame, text="Switch to sensor planning utility", command=self.switch_to_sensor)
        self.switch_sensor_button.pack(fill="x")

        self.switch_bus_button = tk.Button(self.utility_frame, text="Switch to bus planning utility", command=self.switch_to_bus)
        self.switch_bus_button.pack(fill="x")

        self.switch_clock_button = tk.Button(self.utility_frame, text="Switch to clock planning utility", command=self.switch_to_clock)
        self.switch_clock_button.pack(fill="x")

        self.switch_gpio_button = tk.Button(self.utility_frame, text="Switch to GPIO planning utility", command=self.switch_to_gpio)
        self.switch_gpio_button.pack(fill="x")

        self.switch_qor_button = tk.Button(self.utility_frame, text="Switch to QoR utility", command=self.switch_to_qor)
        self.switch_qor_button.pack(fill="x")

    def save_snapshot(self):
        idx = 0
        f_name = self.output_dir + '/screenshot_' + str(idx) + '.png'
        while (os.path.exists(f_name)):
            idx += 1
            f_name = self.output_dir + '/screenshot_' + str(idx) + '.png'
        self.canvas.postscript(file=f"{f_name}.ps", colormode='color')

        image = Image.open(f"{f_name}.ps")
        image.save(f"{f_name}.png")


    def save_full_view(self):
        canvas = self.canvas
        bbox = canvas.bbox("all")
        if bbox:
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, x2, y2 = 0, 0, canvas.winfo_width(), canvas.winfo_height()

        image_width = x2 - x1
        image_height = y2 - y1
        image = Image.new("RGB", (image_width, image_height), "white")
        draw = ImageDraw.Draw(image)
        for item in canvas.find_all():
            coords = canvas.coords(item)
            item_type = canvas.type(item)
            fill_color = canvas.itemcget(item, "fill")
            if fill_color == "":
                fill_color = None

            outline_color = None
            if item_type in ["rectangle", "oval", "polygon"]:
                outline_color = canvas.itemcget(item, "outline")
                if outline_color == "":
                    outline_color = None

            if item_type == "rectangle":
                draw.rectangle(coords, fill=fill_color, outline=outline_color)
            elif item_type == "oval":
                draw.ellipse(coords, fill=fill_color, outline=outline_color)
            elif item_type == "line":
                draw.line(coords, fill=fill_color)
            elif item_type == "polygon":
                draw.polygon(coords, fill=fill_color, outline=outline_color)

        for item in canvas.find_all():
            coords = canvas.coords(item)
            item_type = canvas.type(item)
            fill_color = canvas.itemcget(item, "fill")
            if fill_color == "":
                fill_color = None
            if item_type == "text":
                text = canvas.itemcget(item, "text")
                font_options = canvas.itemconfig(item)["font"]
                #font_family, font_size = font_options.split(" ")
                for option in font_options:
                    if isinstance(option, str) and " " in option:  # find string like "Arial 7"
                        font_family, font_size = option.split(" ")
                        break

                tk_font = tkFont.Font(family=font_family, size=int(font_size))

                pillow_font = ImageFont.truetype("DejaVuSans.ttf", size=int(font_size))
                #pillow_font = ImageFont.truetype("arial.ttf", tk_font.cget("size"))

                text_width = tk_font.measure(text)
                text_height = tk_font.metrics("linespace")

                adjusted_coords = (coords[0] - text_width / 2, coords[1] - text_height / 2)

                draw.text(adjusted_coords, text, fill=fill_color, font=pillow_font)
        idx = 0
        f_name = self.output_dir + '/full_view_' + str(idx) + '.png'
        while (os.path.exists(f_name)):
            idx += 1
            f_name = self.output_dir + '/full_view_' + str(idx) + '.png'
        self.canvas.postscript(file=f"{f_name}.ps", colormode='color', x=x1, y=y1, width=x2-x1, height=y2-y1)
        image.save(f_name)

    def create_widgets(self):
        self.master.bind_all("<MouseWheel>", lambda event: "break")
        # default style for all custom frame

        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(side="left", fill="both", expand=True)

        self.canvas = ZoomableCanvas(self.canvas_frame, app=self, bg="white", width=850, height=850)
        self.hbar = Scrollbar(self.canvas_frame, orient=HORIZONTAL, command=self.canvas.xview)

        self.hbar.pack(side="bottom", fill="x")
        self.vbar = Scrollbar(self.canvas_frame, orient=VERTICAL, command=self.canvas.yview)
        self.vbar.pack(side="right", fill="y")
        self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)


        self.utility_frame = ttk.Frame(self)
        self.utility_frame.pack(side="right", fill="y")
        self.show_selection_screen()

        self.right_frame = ttk.Frame(self, width=350)
        self.right_frame.pack(side="right", fill="both")

        # zoomable
        self.scrollable_canvas = tk.Canvas(self.right_frame, width=450)
        self.scrollable_canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = tk.Scrollbar(self.right_frame, orient="vertical", command=self.scrollable_canvas.yview)
        self.scrollbar.pack(side="right", fill="y")

        self.scrollable_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollable_canvas.bind('<Configure>', lambda e: self.scrollable_canvas.configure(scrollregion=self.scrollable_canvas.bbox("all")))
        self.scrollable_canvas.bind('<Button-4>', self._on_mouse_wheel_linux)
        self.scrollable_canvas.bind('<Button-5>', self._on_mouse_wheel_linux)
        self.inner_frame = tk.Frame(self.scrollable_canvas, width=450)
        self.scrollable_canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        #self.controls_frame = tk.Frame(self.inner_frame)
        #self.controls_frame.pack(side="top", fill="x")

        #self.load_button = tk.Button(self.controls_frame, text="Import block from TVC", command=self.import_json, width=1, height=1, wraplength=450)
        #self.load_button.pack(fill="x")

        self.buttons_frame = tk.Frame(self.inner_frame)
        self.buttons_frame.pack(side="top", fill="both", expand=True)


        # A virtual frame for legend
        self.legend_frame = tk.Frame(self.inner_frame, width=450)
        self.legend_frame.pack(side="top", fill="both", expand=True)


        # A legend frame for block
        self.block_legend_frame = tk.LabelFrame(self.legend_frame, text='Cell name', width=450)
        self.block_legend_frame.pack(side="top", fill="both", expand=True)

        var = tk.BooleanVar(value=False)
        checkbutton = ttk.Checkbutton(self.legend_frame, text="Show labels", variable=var, command=self.canvas.update_canvas_w_label)
        checkbutton.pack(side="top", fill="x")
        self.canvas.checkbuttons['Show labels'] = var
        var = tk.BooleanVar(value=False)
        checkbutton = ttk.Checkbutton(self.legend_frame, text="Show text", variable=var, command=self.canvas.update_canvas_w_label)
        checkbutton.pack(side="top", fill="x")
        self.canvas.checkbuttons['Show text'] = var
        var = tk.BooleanVar(value=False)
        checkbutton = ttk.Checkbutton(self.legend_frame, text="Show connections", variable=var, command=self.canvas.update_canvas_w_connections)
        checkbutton.pack(side="top", fill="x")
        self.canvas.checkbuttons['Show connections'] = var

        self.inner_frame.update_idletasks()
        self.scrollable_canvas.config(scrollregion=self.scrollable_canvas.bbox("all"))

        self.block_dict = {}

        self._bind_mouse_wheel(self.inner_frame, scrollable=True)

        self.master.bind("<space>", self.canvas.fit)
        self.master.bind("<F12>", self.canvas.change_color_mode)

    def _bind_mouse_wheel(self, widget, scrollable=False):
        """Bind mouse wheel events to all child widgets"""
        if scrollable == False:
            return
        widget.bind('<Button-4>', self._on_mouse_wheel_linux)
        widget.bind('<Button-5>', self._on_mouse_wheel_linux)

        for child in widget.winfo_children():
            self._bind_mouse_wheel(child, scrollable=True)

    def _on_mouse_wheel_linux(self, event):
        widget = self.scrollable_canvas.winfo_containing(event.x_root, event.y_root)
        ancestor = self.scrollable_canvas
        #print(widget)
        #print(ancestor)
        enter_scrollable = False
        while widget is not None:
            if widget == ancestor:
                enter_scrollable = True
                break
            widget = widget.master
        #print(enter_scrollable)
        if enter_scrollable:
            if event.num == 4:
                self.scrollable_canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                self.scrollable_canvas.yview_scroll(3, "units")
            return "break"
        return "break"

    #def on_pointView_select(self, event):
    #    selection = self.pointView.selection()
    #    if selection:
    #        item = self.pointView.item(selection[0])
    #        point_str = item["values"][1]
    #        coord_part = point_str[1:-1]
    #        col, row = map(int, coord_part.split(", "))
    #        self.canvas.center_on_point(col, row, item["values"][0])

    def on_right_click(self, event):
        selection = self.pointView.selection()
        if selection:
            self.context_menu.tk_popup(event.x_root, event.y_root)



    def create_dynamic_buttons(self, data):
        # clear the button
        for widget in self.buttons_frame.winfo_children():
            widget.destroy()

        # create new buttons
        button_list = []
        for option in data:
            button = tk.Button(self.buttons_frame, text=option, command=lambda r=option: self.on_fp_click(r, button_list))
            button_list.append(button)
            button.pack(side="top", fill="x", padx=0, pady=0, ipadx=0, ipady=0)

    def canvas_setup(self):
        def parse_blocks(data, record):
            die_name = record
            die_data = data['S1_output'][die_name]
            all_block_name = list(set([block_info['DesignName'] for block_info in die_data['BlockShapes']]))
            random_block_color = generate_colors(len(all_block_name))
            block_color_dict = {}

            for idx in range(0, len(all_block_name)):
                block_color_dict[all_block_name[idx]] = random_block_color[idx]
            #print(block_color_dict)
            for block_info in die_data['BlockShapes']:
                if die_name not in self.canvas.marked_blocks:
                    self.canvas.marked_blocks[die_name] = {}
                if block_info['DesignName'] not in self.canvas.marked_blocks[die_name]:
                    PinCoords = block_info['PinCoords'] if 'PinCoords' in block_info.keys() else None
                    if 'Color' not in block_info:
                        color = block_color_dict[block_info['DesignName']]
                    else:
                        color = block_info['Color']
                    self.canvas.marked_blocks[die_name][block_info['BlockName']] = MarkedBlock(block_info['Coords'], block_info['DesignName'], block_info['BlockName'], color, block_info['PinCoords'], block_info['ThermalSensor'], is_sensor_assigned=False, top_power_nets=block_info['TopPowerNets'], top_ground_nets=block_info['TopGroundNets'])
                    #print(self.canvas.marked_blocks[die_name][block_info['BlockName']])
                    x, y = float(block_info['Coords'][2][0]), float(block_info['Coords'][2][1])

        for chiplet in self.canvas.data['S1_output'].keys():
            parse_blocks(self.canvas.data, chiplet)
            data = self.canvas.data
            self.canvas.core_region[chiplet] = data['S1_output'][chiplet]['TOPShape']['CoreCoords']
            self.canvas.die_region[chiplet] = data['S1_output'][chiplet]['TOPShape']['DieCoords']

            if chiplet not in self.canvas.marked_blockage:
                self.canvas.marked_blockage[chiplet] = []
            self.canvas.marked_blockage[chiplet].extend(data['S1_output'][chiplet]['TOPShape']['Blockages'])
            self.canvas.die_max_x[chiplet] = int( data['S1_output'][chiplet]['TOPShape']['DieCoords'][2][0] / self.canvas.cell_size) + 1
            self.canvas.die_max_y[chiplet] = int( data['S1_output'][chiplet]['TOPShape']['DieCoords'][2][1] / self.canvas.cell_size) + 1

            #self.update_block_list(self.canvas.marked_blocks[chiplet])
        self.load_ip_dict()
        self.load_solution_info()

    def swap_fp(self):
        die_names = self.canvas.data['S1_output'].keys()
        for widget in self.buttons_frame.winfo_children():
            widget.destroy()
        #print('Swap fp clicked')
        self.create_dynamic_buttons(die_names)

    def on_fp_click(self, record, button_list):

        for button in button_list:
            button.destroy()
        button_list.clear()
        button = tk.Button(self.buttons_frame, text='Swap fp', command=self.swap_fp)
        button.pack(side="top", fill="x", padx=0, pady=0, ipadx=0, ipady=0)
        # screenshot button
        self.screenshot_button = tk.Button(self.buttons_frame, text="Take a snapshot", command=self.save_snapshot)
        self.screenshot_button.pack(fill="x")
        self.full_view_button = tk.Button(self.buttons_frame, text="Take a full view screenshot", command=self.save_full_view)
        self.full_view_button.pack(fill="x")
        self.master.title("TVC - "+record)
        self.canvas.die_name = record

        self.canvas.reset_canvas_data()
        self.canvas.fit()
        self.update_block_legend()
        self.update_solution_legend(record)
        self.build_connections(record)
        self.update_utility(record)


    def update_solution_legend(self, chiplet):
        reset_used_color()
        for widget in self.legend_frame.winfo_children():
            #print('Widget',widget._name)
            if 'checkbutton' not in widget._name and 'labelframe' not in widget._name :
                widget.destroy()
        if chiplet not in self.canvas.solution_classes:
            return
        all_solutions = self.canvas.solution_classes[chiplet]
        max_width = 450  # Define the maximum width for wrapping
        current_width = 0

        current_row_frame = tk.Frame(self.legend_frame)
        current_row_frame.pack(fill="x", pady=5)

        for solution_name, solution_obj_list in all_solutions.items():
            obj = solution_obj_list[0]
            all_attr = obj.__dict__.keys()

            if 'Legend_key' in all_attr:
                legend_key = getattr(obj, 'Legend_key', '')
                print(f"Sub-class {solution_name} has legend_key")
            else:
                print(f"No legend_key in {solution_name} {all_attr}")
                cur_legend_frame = tk.LabelFrame(current_row_frame, text=solution_name)
                cur_legend_frame.pack(side="left", padx=5, pady=5)
                var = tk.BooleanVar(value=False)
                checkbutton = ttk.Checkbutton(cur_legend_frame, text=solution_name, variable=var, command=self.canvas.update_canvas_w_label)
                checkbutton.pack(side="left")
                self.canvas.checkbuttons[solution_name] = var

                # Update the current width and check if a new row is needed
                current_width += cur_legend_frame.winfo_reqwidth() + 225
                #print(solution_name, current_width)
                if current_width > max_width:
                    current_row_frame = tk.Frame(self.legend_frame)
                    current_row_frame.pack(fill="x", pady=5)
                    current_width = 0
                continue

            legend_keys = set()
            cur_obj_legend = {}
            num_columns = 0
            for obj in solution_obj_list:
                key = getattr(obj, 'Legend_key', 'No this key')
                if key != 'No this key':
                    cur_obj_legend.setdefault(key, []).append(obj)
                    legend_keys.add(key)
            self.canvas.solution_legend_objs[solution_name] = cur_obj_legend
            color_list = generate_colors(len(legend_keys))
            legend_key_color = {}
            idx = 0
            for element in legend_keys:
                legend_key_color[element] = (color_list[idx], 'square', 'False')
                idx += 1
            self.canvas.legend_items[solution_name] = legend_key_color

            cur_legend_frame = tk.LabelFrame(current_row_frame, text=solution_name)
            cur_legend_frame.pack(side="left", padx=5, pady=5)
            self.create_legend_ui(self.canvas.legend_items[solution_name], cur_legend_frame)

            # Update the current width and check if a new row is needed
            current_width += cur_legend_frame.winfo_reqwidth() + 225
            #print(solution_name, current_width)
            if current_width > max_width:
                current_row_frame = tk.Frame(self.legend_frame)
                current_row_frame.pack(fill="x", pady=5)
                current_width = 0
        #print("INFO1", self.canvas.solution_legend_objs)
        #print("INFO2", self.canvas.solution_legend_objs.keys())
        self.legend_frame.update_idletasks()
        self._bind_mouse_wheel(self.inner_frame, scrollable=True)
        self.scrollable_canvas.configure(scrollregion=self.scrollable_canvas.bbox("all"))
        self.canvas.update_canvas_w_label()

    def create_legend_ui(self, legend, frame, title='', check_button=True):
        if title:
            tk.Label(frame, text=title).pack(side="top", fill="y", anchor='w')

        shape_draw_methods = {
            'circle': lambda canvas, color: canvas.create_oval(5, 5, 15, 15, fill=color),
            'square': lambda canvas, color: canvas.create_rectangle(5, 5, 15, 15, fill=color),
            'triangle': lambda canvas, color: canvas.create_polygon(10, 5, 5, 15, 15, 15, fill=color),
            'diamond': lambda canvas, color: canvas.create_polygon(10, 5, 5, 10, 10, 15, 15, 10, fill=color)
        }

        cur_frame = frame
        for item, (color, shape, visible) in legend.items():
            var = tk.BooleanVar(value=visible)
            frame_item = ttk.Frame(cur_frame)
            frame_item.pack(anchor='w', pady=2)

            color_canvas = tk.Canvas(frame_item, width=20, height=20)
            color_canvas.pack(side="left", padx=5)
            shape_draw_methods[shape](color_canvas, color)
            if check_button:
                checkbutton = ttk.Checkbutton(frame_item, text=item, variable=var, command=self.canvas.update_canvas_w_label)
                checkbutton.pack(side="left")
                self.canvas.checkbuttons[item] = var
            else:
                text_label = tk.Label(frame_item, text=item)
                text_label.pack(side='left')
            self.scrollable_canvas.update_idletasks()
            self.scrollable_canvas.configure(scrollregion=self.scrollable_canvas.bbox("all"))

        self._bind_mouse_wheel(frame, scrollable=True)

    def update_block_legend(self):
        for widget in self.block_legend_frame.winfo_children():
            widget.destroy()
        for inst_name, value in self.canvas.marked_blocks[self.canvas.die_name].items():
            self.canvas.block_legend[value.CellName] = (value.Color, 'square', True)

        self.create_legend_ui(self.canvas.block_legend, self.block_legend_frame, check_button=False)

    def load_ip_dict(self):
        #print(f"All IPS: {self.canvas.data['IPS'].keys()}")

        for ip in self.canvas.data['IPS'].keys():
            ip_width = round(self.canvas.data['IPS'][ip]['SIZE_X'], 4)
            ip_height = round(self.canvas.data['IPS'][ip]['SIZE_Y'], 4)

            self.canvas.ip_size.update({ip: {'width': ip_width, 'height': ip_height}})
        #print(f"{self.canvas.ip_size}")

    def load_solution_info(self):
        all_solution_data = {}
        #print(self.canvas.data.keys())
        for key, value in self.canvas.data.items():
            if 'Solution_' in key:
                for target, solution_value in value.items():
                    if target in all_solution_data:
                        all_solution_data[target].update(solution_value)
                    else:
                        all_solution_data[target] = solution_value
        self.canvas.solution_data = all_solution_data
        self.canvas.solution_classes = create_classes_from_dict(self.canvas.solution_data)

        for major_class, subclasses in self.canvas.solution_classes.items():
            #print(f"Built up major class {major_class} and its sub-classes ")
            self.canvas.solution_obj[major_class] = {}
            for class_name, obj_list in subclasses.items():
                #print(f"Sub-class:{class_name}, Attribute:{vars(type(obj_list[0]))}")
                for obj in obj_list:
                    name = getattr(obj, 'Name', '')
                    if name != '' and name not in self.canvas.solution_obj[major_class]:
                        self.canvas.solution_obj[major_class][name] = obj
                    elif self.canvas.solution_obj[major_class][name] == obj:
                        pass
                    else:
                        raise ValueError(f"In chiplet: {major_class}, name of obj is empty:{name}, or duplicate name obj detected in json {obj}, {self.canvas.solution_obj[major_class][name]}")


    def build_connections(self, chiplet):
        attribute = 'Connection'
        if chiplet not in self.canvas.solution_classes:
            return
        all_solutions = self.canvas.solution_classes[chiplet]
        self.canvas.solution_connection = {}
        for solution_name, obj_list in all_solutions.items():
            obj = obj_list[0]
            if hasattr(obj, attribute):
                connection_pair = []
                for obj in obj_list:
                    connection = getattr(obj, attribute, [])
                    inst_name = getattr(obj, 'Name')
                    if len(connection) != 0:
                        for name in connection:
                            if name in self.canvas.solution_obj[chiplet]:
                                conn_obj = self.canvas.solution_obj[chiplet][name]
                            else:
                                conn_obj = self.canvas.marked_blocks[chiplet][name]
                            connection_pair.append((obj, conn_obj))
                self.canvas.solution_connection[solution_name] = connection_pair
        #print(self.canvas.solution_connection)

    def import_json(self, input_json=None):
        #file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not input_json: return

        with open(input_json, "r") as json_file:
            self.canvas.data = json.load(json_file)
            data = self.canvas.data
            die_names = data['S1_output'].keys()
            self.create_dynamic_buttons(die_names)



