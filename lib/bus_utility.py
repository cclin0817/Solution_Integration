
✅ 還原內容如下：

import tkinter as tk
from tkinter import Menu, Toplevel, StringVar, Button, ttk, filedialog
from .base_utility import *
from .functions import *
from .block import *
from . import log
import json
import re

class Bus:
    def __init__(self, asso_points, asso_blocks, direction):
        self.asso_points = asso_points
        self.asso_blocks = asso_blocks
        self.direction = direction
        self.path_segment = []

class BusUtility(BaseUtility):
    def __init__(self, app, canvas, args):
        super().__init__(app, canvas, args)
        self.marked_points = {}
        self.buses = []
        self.output_dir = self.app.output_dir + '/bus_utility/'
        os.makedirs(self.output_dir, exist_ok=True)

    def create_widgets(self, parent_frame):
        label = tk.Label(parent_frame, text="Bus planning")
        label.pack()

        self.load_button = tk.Button(parent_frame, text="Load location from JSON", command=self.load_from_json, width = 1, height = 1)
        self.load_button.pack(fill="x")

        self.save_button = tk.Button(parent_frame, text="Save location to JSON", command=self.save_to_json, width = 1, height = 1)
        self.save_button.pack(fill="x")

        self.bus_button = tk.Button(parent_frame, text="Create bus", command=self.add_bus)
        self.bus_button.pack(fill="x")

        self.pointView = ttk.Treeview(parent_frame, columns=("name", "coordinates", "layout_coor"), show="headings")
        self.pointView.column("name", width = 50)
        self.pointView.column("coordinates", width = 150)
        self.pointView.column("layout_coor", width = 150)
        self.pointView.heading("name", text="Name")
        self.pointView.heading("coordinates", text="Coordinates in GUI")
        self.pointView.heading("layout_coor", text="Real Coordinates")
        self.pointView.pack(side="top", fill="both", expand=True)
        self.pointView.bind("<<TreeviewSelect>>", self.on_pointView_select)
        self.pointView.bind("<Button-3>", self.on_click_point_view)
        self.pointView.bind("<w>", lambda event: self.move_selected_point(0, -1))
        self.pointView.bind("<s>", lambda event: self.move_selected_point(0, 1))
        self.pointView.bind("<a>", lambda event: self.move_selected_point(-1, 0))
        self.pointView.bind("<d>", lambda event: self.move_selected_point(1, 0))

        self.busView = ttk.Treeview(parent_frame, columns=("name", "direction", "asso_point", "asso_block"), show="headings")
        self.busView.column("name", width = 50)
        self.busView.column("direction", width = 100)
        self.busView.column("asso_point", width = 100)
        self.busView.column("asso_block", width = 350)
        self.busView.heading("name", text="Name")
        self.busView.heading("direction", text="Direction")
        self.busView.heading("asso_point", text="Associated points")
        self.busView.heading("asso_block", text="Associated blocks")
        self.busView.pack(side="top", fill="both", expand=True)

        self.return_button = tk.Button(parent_frame, text="Return to utility selection", command=self.app.show_selection_screen)
        self.return_button.pack(fill="x")

        self.context_menu = Menu(parent_frame, tearoff=0)
        self.context_menu.add_command(label="Delete", command=self.delete_point)

        self.bus_context_menu = Menu(parent_frame, tearoff=0)
        self.bus_context_menu.add_command(label="Delete Bus", command=self.delete_bus)
        self.busView.bind("<Button-3>", self.on_bus_right_click)

    def move_selected_point(self, dx, dy):
        if self.is_point_moved(dx, dy, False):
            self.draw()

    def delete_point(self):
        selection = self.pointView.selection()
        if selection:
            item = self.pointView.item(selection[0])
            point_str = item["values"][1]
            coord_part = point_str[1:-1]
            col, row = map(int, coord_part.split(", "))
            if (col, row) in self.marked_points:
                del self.marked_points[(col, row)]
                self.rename_points()
                self.draw_points()

    def load_from_json(self):
        file_path = filedialog.askopenfilename(initialdir=self.output_dir, filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)

                self.marked_points.clear()
                self.buses.clear()

                for coord, info in data.get("points", {}).items():
                    col, row = map(int, coord.split(","))
                    self.create_point(col, row, name=info['name'])

                for bus_info in data.get("buses", []):
                    bus = Bus(bus_info["asso_points"], bus_info["asso_blocks"], bus_info["direction"],)
                    self.buses.append(bus)
                    self.build_path(bus)

                self.update_point_list(self.marked_points)
                self.update_bus_list()
                self.draw()

    def save_to_json(self):
        file_path = filedialog.asksaveasfilename(initialdir=self.output_dir, defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            data = {
                "points": {
                    f"{col},{row}": {"name": marked_point.name}
                    for (col, row), marked_point in self.marked_points.items()
                },
                "buses": [
                    {"index": index, "asso_points": bus.asso_points, "asso_blocks": bus.asso_blocks, "direction": bus.direction}
                    for index, bus in enumerate(self.buses)
                ]
            }
            with open(file_path, "w") as json_file:
                json.dump(data, json_file)

    def add_bus(self):
        dialog = CustomDirectionDialog(self.app)
        direction = dialog.result

        bus_window = Toplevel(self.app)
        bus_window.title("Select Asso Points for Bus")
        bus_window.geometry("600x400")

        asso_point_var = StringVar(bus_window)
        sorted_marked_points = sorted(self.marked_points.items(), key=lambda item: item[1].name)
        point_names = [f"{marked_point.name} ({c}, {r})" for (c, r), marked_point in sorted_marked_points]
        point_names_var = StringVar(value=point_names)

        regex = re.compile(r'u_sub_block_(\d+)')
        valid_blocks = {}
        for key, block in self.canvas.marked_blocks[self.canvas.die_name].items():
            name = block.Name
            match = regex.match(name)
            if match:
                valid_blocks[key] = name
        asso_block_var = StringVar(bus_window)
        block_names = [f"{regex.match(valid_blocks[key]).group(0)}" for key in valid_blocks]
        block_names = sorted(block_names, key=lambda x: int(re.search(r'\d+', x).group()))
        block_names_var = StringVar(value=block_names)

        listbox = tk.Listbox(bus_window, listvariable=point_names_var, selectmode="multiple", height=15, exportselection=False)
        listbox.pack(side="left", fill="both", expand=True)

        listbox2 = tk.Listbox(bus_window, listvariable=block_names_var, selectmode="multiple", height=15, exportselection=False)
        listbox2.pack(side="left", fill="both", expand=True)

        def create_bus():
            selected_indices = listbox.curselection()
            selected_indices2 = listbox2.curselection()
            if selected_indices and selected_indices2:
                asso_points = [point_names[i].split(" ")[0] for i in selected_indices]
                asso_blocks = []
                if direction == "V":
                    for i in selected_indices2:
                        pre_name = re.match(r"([a-zA-Z_]+)(\d+)", block_names[i])
                        asso_blocks.append(pre_name.group(2))
                else:
                    pre_name = re.match(r"([a-zA-Z_]+)(\d+)", block_names[ selected_indices2[0] ])
                    pre_name2 = re.match(r"([a-zA-Z_]+)(\d+)", block_names[ selected_indices2[-1] ])
                    for i in range(int(pre_name.group(2)), int(pre_name2.group(2))+1):
                        #pre_name = re.match(r"([a-zA-Z_]+)(\d+)", block_names[i])
                        asso_blocks.append(f"{i}")
                bus = Bus(asso_points, asso_blocks, direction)
                self.buses.append(bus)
                self.update_bus_list()

                self.build_path(bus)

            bus_window.destroy()

        select_button = Button(bus_window, text="Add Bus", command=create_bus)
        select_button.pack(pady=10)

        bus_window.mainloop()

    def build_path(self, bus):
        routing_path = [int(s) for s in bus.asso_points]
        point_index = 0;
        maze  = self.canvas.create_maze()
        while point_index < len(routing_path) - 1:
            for (col, row), point in self.marked_points.items():
                if int(point.name) == routing_path[point_index]:
                    start = (col, row)
                if int(point.name) == routing_path[point_index+1]:
                    end = (col, row)
            path, total_distance = bfs(maze, start, end)
            bus.path_segment.append(path)
            point_index += 1
        self.draw_path()

    def draw_path(self):
        self.canvas.delete("connection")
        for bus in self.buses:
            for path in bus.path_segment:
                coords = []
                for cur_pt, next_pt in zip(path, path[1:]):
                    x1, y1 = cur_pt[0]*self.canvas.cell_size*self.canvas.scale_factor , cur_pt[1]*self.canvas.cell_size*self.canvas.scale_factor
                    x2, y2 = next_pt[0]*self.canvas.cell_size*self.canvas.scale_factor , next_pt[1]*self.canvas.cell_size*self.canvas.scale_factor
                    coords.extend([x1, y1, x2, y2])
                self.canvas.create_line(coords, tags='connection', fill='darkgreen', width=3)

    def update_bus_list(self):
        for i in self.busView.get_children():
            self.busView.delete(i)
        for index, bus in enumerate(self.buses):
            asso_points_str = ", ".join(bus.asso_points)
            asso_blocks_str = ", ".join(bus.asso_blocks)
            self.busView.insert("", "end", values=(index,bus.direction , asso_points_str, asso_blocks_str))

    def delete_bus(self):
        selection = self.busView.selection()
        if selection:
            item = self.busView.item(selection[0])
            bus_index = int(item["values"][0])

            if 0 <= bus_index < len(self.buses):
                del self.buses[bus_index]
                self.update_bus_list()
        self.draw_path()

    def on_bus_right_click(self, event):
        self.pointView.selection_remove(self.pointView.selection())
        selection = self.busView.identify_row(event.y)
        if selection:
            self.busView.selection_set(selection)
            self.bus_context_menu.tk_popup(event.x_root, event.y_root)

    def update_point_list(self, points):
        for i in self.pointView.get_children():
            self.pointView.delete(i)
        sorted_points = sorted(points.items(), key=lambda item: int(item[1].name))
        for (col, row), point in sorted_points:
            self.pointView.insert("", "end", values=(point.name, f"({col}, {row})", f"({col*self.canvas.cell_size}, {(self.canvas.die_max_y[self.canvas.die_name]-row)*self.canvas.cell_size})"))

    def on_click_canvas(self, col, row):
        if (col, row) in self.marked_points: return
        if self.canvas.in_block_region(col, row) or self.canvas.in_blockage_region(col, row): return
        if not self.canvas.in_core_region(col, row): return
        self.create_point(col, row)

    def create_point(self, col, row, asso_block="", name = None):
        if name == None:
            name = str(len(self.marked_points))
        self.marked_points[(col, row)] = BusPoint(name)
        self.fill_cell(col, row, name, "point")
        self.update_point_list(self.marked_points)

    def draw(self):
        self.draw_points()
        self.draw_path()



