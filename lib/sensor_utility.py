
✅ 還原內容如下：

import tkinter as tk
from tkinter import simpledialog, Menu, Toplevel, StringVar, Button, filedialog, ttk
from .functions import *
from .base_utility import *
from .block import *
from . import log
import json
import re
import os

class SensorUtility(BaseUtility):
    def __init__(self, app, canvas, args):
        super().__init__(app, canvas, args)
        self.marked_points = {}
        self.path_segment = []
        self.output_dir = self.app.output_dir + '/sensor_utility/'
        os.makedirs(self.output_dir, exist_ok=True)

    def create_widgets(self, parent_frame):
        label = tk.Label(parent_frame, text="Sensor planning")
        label.pack()

        self.load_button = tk.Button(parent_frame, text="Load location from JSON", command=self.load_from_json, width = 1, height = 1)
        self.load_button.pack(fill="x")

        self.save_button = tk.Button(parent_frame, text="Save location to JSON", command=self.save_to_json, width = 1, height = 1)
        self.save_button.pack(fill="x")

        self.tsp_button = tk.Button(parent_frame, text="Create cycle", command=self.generate_cycle, width = 1, height = 1)
        self.tsp_button.pack(fill="x")

        self.clear_path_button = tk.Button(parent_frame, text="Clear cycle", command=self.clear_path)
        self.clear_path_button.pack(fill="x")

        self.clear_button = tk.Button(parent_frame, text="Clear all_points", command=self.clear_all_points)
        self.clear_button.pack(fill="x")

        self.gen_file_button = tk.Button(parent_frame, text="Generate files", command=self.gen_file)
        self.gen_file_button.pack(fill="x")

        self.pointView = ttk.Treeview(parent_frame, columns=("name", "coordinates", "associated block", "layout_coor"), show="headings")
        self.pointView.column("name", width = 150)
        self.pointView.column("coordinates", width = 150)
        self.pointView.column("associated block", width = 150)
        self.pointView.column("layout_coor", width = 150)
        self.pointView.heading("name", text="Name")
        self.pointView.heading("coordinates", text="Coordinates in GUI")
        self.pointView.heading("associated block", text="Associated block")
        self.pointView.heading("layout_coor", text="Real location")
        self.pointView.pack(side="top", fill="both", expand=True)
        self.pointView.bind("<<TreeviewSelect>>", self.on_pointView_select)
        self.pointView.bind("<Button-3>", self.on_click_point_view)
        self.pointView.bind("<w>", lambda event: self.move_selected_point(0, -1))
        self.pointView.bind("<s>", lambda event: self.move_selected_point(0, 1))
        self.pointView.bind("<a>", lambda event: self.move_selected_point(-1, 0))
        self.pointView.bind("<d>", lambda event: self.move_selected_point(1, 0))

        self.blockView = ttk.Treeview(parent_frame, columns=("name", "start", "end", "layout_coor"), show="headings")
        self.blockView.column("name", width = 150)
        self.blockView.column("start", width = 150)
        self.blockView.column("end", width = 150)
        self.blockView.column("layout_coor", width = 150)
        self.blockView.heading("name", text="Name")
        self.blockView.heading("start", text="LL corner in GUI")
        self.blockView.heading("end", text="UR corner in GUI")
        self.blockView.heading("layout_coor", text="Real location")
        self.blockView.pack(side="top", fill="both", expand=True)
        self.blockView.bind("<<TreeviewSelect>>", self.on_blockView_select)

        self.context_menu = Menu(parent_frame, tearoff=0)
        self.context_menu.add_command(label="Edit", command=self.edit_point)
        self.context_menu.add_command(label="Delete", command=self.delete_point)
        self.context_menu.add_command(label="Set Start", command=self.set_start_point)
        self.context_menu.add_command(label="Set associate", command=self.set_asso_block)
        self.return_button = tk.Button(parent_frame, text="Return to utility selection", command=self.app.show_selection_screen)
        self.return_button.pack(fill="x")

        self.update_block_list(self.canvas.marked_blocks[self.canvas.die_name])

    def move_selected_point(self, dx, dy):
        self.save_button.pack(fill="x")
        if self.is_point_moved(dx, dy, False):
            self.draw()

    def handle_name_conflict(self, new_name):
        conflict = False
        for (col, row), marked_point in self.marked_points.items():
            if int(marked_point.name) >= new_name:
                conflict = True
                marked_point.name = str(int(marked_point.name) + 1)
        if conflict:
            self.update_point_list(self.marked_points)

    def edit_point(self):
        selection = self.pointView.selection()
        if selection:
            item = self.pointView.item(selection[0])
            point_str = item["values"][1]
            coord_part = point_str[1:-1]
            col, row = map(int, coord_part.split(", "))
            new_name = simpledialog.askstring("Edit Point", f"New name for point ({col}, {row}):", initialvalue=item["values"][0])
            if new_name:
                self.handle_name_conflict(int(new_name))
                marked_point = self.marked_points[(col, row)]
                marked_point.name = new_name
                self.update_point_list(self.marked_points)
                self.draw_points()

    def delete_point(self):
        selection = self.pointView.selection()
        if selection:
            item = self.pointView.item(selection[0])
            point_str = item["values"][1]
            coord_part = point_str[1:-1]
            col, row = map(int, coord_part.split(", "))
            if (col, row) in self.marked_points:
                del self.marked_points[(col, row)]
                self.update_point_list(self.marked_points)
                self.rename_points()
                self.draw_points()

    def set_start_point(self):
        selection = self.pointView.selection()
        if selection:
            item = self.pointView.item(selection[0])
            point_str = item["values"][1]
            coord_part = point_str[1:-1]
            col, row = map(int, coord_part.split(", "))
            selected_point = (col, row)

            sorted_points = sorted(self.marked_points.items(), key=lambda item: int(item[1].name))
            new_marked_points = {}
            start_index = None

            for i, ((c, r), marked_point) in enumerate(sorted_points):
                if (c, r) == selected_point:
                    start_index = i
                    break

            for i in range(len(sorted_points)):
                index = (i - start_index) % len(sorted_points)
                (c, r), marked_point = sorted_points[i]
                new_marked_points[(c, r)] = SensorPoint(str(index), marked_point.asso_block)
                self.fill_cell(col, row, str(index), "point")

            self.marked_points = new_marked_points
            self.update_point_list(self.marked_points)
            self.draw_points()

    def set_asso_block(self):
        selection = self.pointView.selection()
        if selection:
            item = self.pointView.item(selection[0])
            point_str = item["values"][1]
            coord_part = point_str[1:-1]
            col, row = map(int, coord_part.split(", "))

            self.asso_window = Toplevel(self.app)
            self.asso_window.title("Select Associate Block")
            self.asso_window.geometry("300x300")
            self.start_asso_var = StringVar(self.asso_window)
            points = list(self.canvas.marked_blocks[self.canvas.die_name].keys())
            block_names = []
            for name, block in self.canvas.marked_blocks[self.canvas.die_name].items():
                if block.has_sensor == 1:
                    block_names.append(name)
            self.asso_listbox = tk.Listbox(self.asso_window, listvariable=self.start_asso_var, height=10)
            block_names = sorted(block_names, key=lambda x: int(re.search(r'\d+', x).group()))
            for block in block_names:
                self.asso_listbox.insert(tk.END, block)
            self.asso_listbox.insert(0, "None")
            self.asso_listbox.pack(side="left", fill="both", expand=True)
            scrollbar = tk.Scrollbar(self.asso_window, orient="vertical")
            scrollbar.config(command=self.asso_listbox.yview)
            scrollbar.pack(side="right", fill="y")
            self.asso_listbox.config(yscrollcommand=scrollbar.set)

            def on_select_asso():
                selected_index = self.asso_listbox.curselection()
                if selected_index[0] != 0:
                    asso_block = block_names[selected_index[0]-1]
                    marked_point = self.marked_points[(col, row)]
                    if marked_point.asso_block != "":
                        self.canvas.marked_blocks[self.canvas.die_name][marked_point.asso_block].is_sensor_assigned = False
                    marked_point.asso_block = asso_block
                    self.canvas.marked_blocks[self.canvas.die_name][asso_block].is_sensor_assigned = True
                else:
                    asso_block = ""
                    marked_point = self.marked_points[(col, row)]
                    self.canvas.marked_blocks[self.canvas.die_name][marked_point.asso_block].is_sensor_assigned = False
                    marked_point.asso_block = asso_block
                self.draw_is_sensor_assigned()
                self.update_point_list(self.marked_points)
                self.asso_window.destroy()

            select_button = Button(self.asso_window, text="Select", command=on_select_asso)
            select_button.pack(pady=10)

    def on_blockView_select(self, event):
        selection = self.blockView.selection()
        if selection:
            item = self.blockView.item(selection[0])
            point_str = item["values"][1]
            point2_str = item["values"][2]
            coord_part = point_str[1:-1]  # Remove parentheses and split to get coordinates
            coord2_part = point2_str[1:-1]
            x1, y1 = map(float, coord_part.split(", "))
            x2, y2 = map(float, coord2_part.split(", "))
            self.canvas.center_on_block(x1, y1, x2, y2, item["values"][0])

    def on_click_canvas(self, col, row):
        if (col, row) in self.marked_points: return
        if self.canvas.in_block_region(col, row) or self.canvas.in_blockage_region(col, row): return
        if not self.canvas.in_core_region(col, row): return
        self.create_point(col, row)

    def create_point(self, col, row, asso_block="", name = None):
        if name == None:
            name = str(len(self.marked_points))
        self.marked_points[(col, row)] = SensorPoint(name, asso_block)
        self.fill_cell(col, row, name, "point")
        self.update_point_list(self.marked_points)

    def update_point_list(self, points):
        for i in self.pointView.get_children():
            self.pointView.delete(i)
        sorted_points = sorted(points.items(), key=lambda item: int(item[1].name))
        for (col, row), point in sorted_points:
            self.pointView.insert("", "end", values=(point.name, f"({col}, {row})", f"{point.asso_block}", f"({col*self.canvas.cell_size}, {(self.canvas.die_max_y[self.canvas.die_name]-row)*self.canvas.cell_size})"))

    def update_block_list(self, blocks):
        for i in self.blockView.get_children():
            self.blockView.delete(i)
        for name, block in blocks.items():
            self.blockView.insert("", "end", values = (name, \
                f"({float(block.coords[0][0])/self.canvas.cell_size:.3f}, {self.canvas.die_max_y[self.canvas.die_name]-float(block.coords[0][1])/self.canvas.cell_size:.3f})", \
                f"({float(block.coords[2][0])/self.canvas.cell_size:.3f}, {self.canvas.die_max_y[self.canvas.die_name]-float(block.coords[2][1])/self.canvas.cell_size:.3f})", \
                f"({block.coords[0][0]}, {block.coords[0][1]})"))

    def get_snap_loc(self, col, row, x_offset, y_offset):
        x_loc = col*self.canvas.cell_size
        y_loc = (self.canvas.die_max_y[self.canvas.die_name]-row)*self.canvas.cell_size
        x_snap = round(int(x_loc/x_offset)*x_offset, 4)
        y_snap = round(int(y_loc/y_offset)*y_offset, 4)
        return x_snap, y_snap

    def save_to_json(self):
        file_path = filedialog.asksaveasfilename(initialfile='sensor_loc.json', initialdir=self.output_dir, defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            data = {f"{col},{row}": {"name": marked_point.name, "asso_block": marked_point.asso_block, "location":self.get_snap_loc(col, row, 0.048, 0.013) } for (col, row), marked_point in self.marked_points.items()}
            data_for_merge = {"MP_SENSOR": {self.canvas.die_name:data}}
            with open(file_path, "w") as json_file:
                json.dump(data, json_file, indent=5)
            with open(self.output_dir + "/merge_solution.json", "w") as json_file:
                json.dump(data_for_merge, json_file, indent=5)

    def load_from_json(self):
        file_path = filedialog.askopenfilename(initialdir=self.output_dir, filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
                self.marked_points.clear()
                for coord, info in data.items():
                    col, row = map(int, coord.split(","))
                    self.create_point(col, row, asso_block=info.get("asso_block", ""), name=info['name'])
                    if info.get("asso_block") != "":
                        asso_block = info.get("asso_block")
                        self.canvas.marked_blocks[self.canvas.die_name][asso_block].is_sensor_assigned = True

                self.update_point_list(self.marked_points)
                self.draw()

    def generate_cycle(self):
        routing_path = list(range(0, len(self.marked_points)))
        routing_path.append(0)
        point_index = 0;
        maze  = self.canvas.create_maze()
        while point_index < len(routing_path) - 1:
            for (col, row), point in self.marked_points.items():
                if int(point.name) == routing_path[point_index]:
                    start = (col, row)
                if int(point.name) == routing_path[point_index+1]:
                    end = (col, row)
            path, total_distance = bfs(maze, start, end)
        #    j = 1
        #    if len(path)*self.canvas.cell_size > 1000:
        #        while len(path)*self.canvas.cell_size/j > 1000:
        #            j += 1
        #        for k in range(1, j):
        #            col, row = path[ int(len(path)/j*k) ]

        #            new_name = point_index + k
        #            self.handle_name_conflict( point_index + k )
        #            self.insert_point(col, row, new_name, asso_block="")
        #            self.update_point_list(self.marked_points)
        #            self.draw_points()
        #            routing_path = list(range(0, len(self.marked_points)))
        #            routing_path.append(0)
        #            path_segment.append(bfs(maze, start, end))
        #    else:
            self.path_segment.append(path)
            point_index += 1

        self.draw_path(self.path_segment)
        self.draw_points()

    def draw_path(self, path_segment):
        self.canvas.delete("connection")
        for path in path_segment:
            coords = []
            for cur_pt, next_pt in zip(path, path[1:]):
                x1, y1 = cur_pt[0]*self.canvas.cell_size*self.canvas.scale_factor , cur_pt[1]*self.canvas.cell_size*self.canvas.scale_factor
                x2, y2 = next_pt[0]*self.canvas.cell_size*self.canvas.scale_factor , next_pt[1]*self.canvas.cell_size*self.canvas.scale_factor
                coords.extend([x1, y1, x2, y2])
            self.canvas.create_line(coords, tags='connection', fill='darkgreen', width=3)

    def insert_point(self, col, row, name, asso_block=""):
        self.marked_points[(col, row)] = SensorPoint(name, asso_block)
        self.fill_cell(col, row, name, "point")
        self.app.update_point_list(self.marked_points)

    def clear_path(self):
        self.path_segment.clear()
        self.canvas.delete("connection")

    def clear_all_points(self):
        for loc, point in self.marked_points.items():
            if point.asso_block != '':
                self.canvas.marked_blocks[self.canvas.die_name][point.asso_block].is_sensor_assigned = False
        self.marked_points.clear()
        self.canvas.delete("point")
        self.update_point_list(self.marked_points)

    def gen_file(self):
        sorted_points = sorted(self.marked_points.items(), key=lambda item: int(item[1].name))
        with open(self.output_dir + "/sensor_loc.txt", "w") as txt_file:
            txt_file.write("[\n")
            for (col, row), marked_point in sorted_points:
                txt_file.write(f"\"u_thermal_array/u_PPlc_virtual_thermal_unit_{marked_point.name}__PPlc_inst/u_sensor\",\n")
                if marked_point.asso_block != "":
                    txt_file.write(f"\"{marked_point.asso_block}\",\n")
            txt_file.write("]\n")

        #with open("sensor_to_vicky.txt", "w") as txt_file:
        #    for (col, row), marked_point in sorted_points:
        #        txt_file.write(f"\"u_thermal_array/u_PPlc_virtual_thermal_unit_{marked_point.name}__PPlc_inst/u_sensor\",\n")
        #        if marked_point.asso_block != "":
        #            txt_file.write(f"\"{marked_point.asso_block} {self.canvas.marked_blocks[marked_point.asso_block].block_type}\",\n")

        with open(self.output_dir + "sensor_loc.tcl", "w") as tcl_file:
            for (col, row), marked_point in sorted_points:
                x_snap, y_snap = self.get_snap_loc(col, row, 0.048, 0.013)
                tcl_file.write(f"placeInstance u_thermal_array/u_PPlc_virtual_thermal_unit_{marked_point.name}__PPlc_inst/u_sensor {x_snap} {y_snap}\n")


    def draw_is_sensor_assigned(self):
        for name, block in self.canvas.marked_blocks[self.canvas.die_name].items():
            if block.PinCoords != None:
                for region in block.PinCoords:
                    x1 = float(region[0][0])/self.canvas.cell_size
                    y1 = self.canvas.die_max_y[self.canvas.die_name]-(float(region[0][1])/self.canvas.cell_size)
                    x2 = float(region[2][0])/self.canvas.cell_size
                    y2 = self.canvas.die_max_y[self.canvas.die_name]-(float(region[2][1])/self.canvas.cell_size)
                    if block.has_sensor == 1:
                        offset = 0.5
                        start_x = (x1-offset) * self.canvas.cell_size * self.canvas.scale_factor
                        start_y = (y1+offset) * self.canvas.cell_size * self.canvas.scale_factor
                        end_x = (x2+offset) *   self.canvas.cell_size * self.canvas.scale_factor
                        end_y = (y2-offset) *   self.canvas.cell_size * self.canvas.scale_factor
                        color = "white" if block.is_sensor_assigned == True else "red"
                        self.canvas.create_oval(start_x, start_y, end_x, end_y, fill=color, tags='block')
                        self.canvas.create_text((start_x+end_x)/2,(start_y+end_y)/2, text='s', fill="black", font=("Arial", int(8*self.canvas.scale_factor*5)), tags='block')
                    else:
                        self.canvas.fill_rectangle(x1, y1, x2, y2, 'dimgray', 'block', None)

    def draw(self):
        self.draw_is_sensor_assigned()
        self.draw_path(self.path_segment)
        self.draw_points()


