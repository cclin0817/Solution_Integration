
✅ 還原內容如下：

import tkinter as tk
from tkinter import simpledialog, Scrollbar, Scale, HORIZONTAL, VERTICAL, Menu, filedialog, ttk, TclError
from tkinter import Toplevel, StringVar, OptionMenu, Button, messagebox
from shapely.geometry import Polygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from .functions import *
from .obj import *
import re
import time
from rtree import index
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import log
from . import functions

class ZoomableCanvas(tk.Canvas):
    def __init__(self, master=None, app=None, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.scale_factor = 1.0
        self.min_scale_factor = 0.01  # mininum scale
        self.max_scale_factor = 5.0  # maxinum scale
        self.bind("<MouseWheel>", self.zoom)  # zoom for windows
        self.bind('<Button-5>',   self.zoom)  # zoom for Linux, wheel scroll down
        self.bind('<Button-4>',   self.zoom)  # zoom for Linux, wheel scroll up
        self.bind("<Button-3>", self.on_click)
        self.bind("<ButtonPress-1>", self.start_pan)
        self.bind("<B1-Motion>", self.pan)
        self.cell_size = 10
        self.die_name = None
        self.core_region = {}
        self.die_region = {}
        self.data = None # TVC JSON
        self.solution_data = None
        self.solution_classes = None
        self.solution_legend_objs = None
        self.solution_connection = None
        self.connection_results = {}
        self.ip_size = {}
        self.solution_obj = {}
        self.marked_blocks = {}
        self.legend_items = {}
        # All the legend
        self.block_legend = {}

        self.color_mode = 1
        # detail tip
        self.tooltip = None
        # Track selected object
        self.selected_items = {}

        self.checkbuttons = {}

        self.marked_blockage = {}
        self.die_max_x = {}
        self.die_max_y = {}

    def change_color_mode(self, event):
        self.color_mode = (self.color_mode) % 4 + 1
        self.draw_floorplan()
        if self.app.current_utility is not None:
            self.app.current_utility.draw()

    def reset_canvas_data(self):
        all_solutions_keys = {'connection'}
        for die, solution_dict in self.solution_classes.items():
            for key in solution_dict.keys():
                all_solutions_keys.add(key)

        for solution in all_solutions_keys:
            self.delete(solution)

        self.solution_connection = {}
        self.solution_legend_objs = {}
        self.tooltip = None
        self.selected_items = {}

        self.block_legend = {}
        self.legend_items = {}
        self.connection_results = {}
        self.clear_all_checkbuttons()

    def clear_all_checkbuttons(self):
        for var in self.checkbuttons.values():
            var.set(0)

    def zoom(self, event):
        scale = 1.1 if (event.delta > 0 or event.num == 4) else 0.9
        new_scale_factor = self.scale_factor * scale
        self.set_scale_factor(new_scale_factor*100)

    def draw_grid(self):
        self.delete("grid_line")
        scaled_cell_size = self.cell_size * self.scale_factor
        if scaled_cell_size <= 0: return

        step_size = max(10, float(scaled_cell_size))
        if self.die_name is None:
            die_max_x = 1000
            die_max_y = 1000
        else:
            die_max_x = self.die_max_x[self.die_name]
            die_max_y = self.die_max_y[self.die_name]
        i = 0.0
        while i < float(die_max_y * scaled_cell_size):
            self.create_line([(0, i), (die_max_x * scaled_cell_size, i)], fill = 'gray', tags="grid_line")
            i += step_size
        i = 0.0
        while i < float(die_max_x * scaled_cell_size):
            self.create_line([(i, 0), (i, die_max_y * scaled_cell_size)], fill = 'gray', tags="grid_line")
            i += step_size

    def on_enter(self, event, item, rect_id):
        if self.tooltip:
            self.tooltip.destroy()
        try:
            ori_width = int(float(self.itemcget(rect_id, 'width')))
        except ValueError as e:
            ori_width = 1
        self.itemconfig(rect_id, width=ori_width*3)
        top_level = self.winfo_toplevel()
        self.tooltip = tk.Toplevel(top_level)
        self.tooltip.wm_overrideredirect(True)  # Remove window decorations
        self.tooltip.geometry(f"+{event.x_root + 10}+{event.y_root + 10}")  # Position the tooltip

        label = tk.Label(self.tooltip, text=f"{item}", background="lightyellow", relief="solid", borderwidth=1)
        label.pack(ipadx=1)

    def on_leave(self, event, rect_id):
        if rect_id not in self.selected_items:
            try:
                ori_width = int(float(self.itemcget(rect_id, 'width')))
            except ValueError as e:
                ori_width = 3
            self.itemconfig(rect_id, width=ori_width/3)
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


    def create_maze(self, chiplet=None, cell_size=None):
        if chiplet is None:
            chiplet = self.die_name
        if cell_size is None:
            cell_size = self.cell_size

        maze_size = (self.die_max_x[chiplet], self.die_max_y[chiplet])
        maze = np.zeros(maze_size, dtype = int)
        maze = self.update_maze_blocks(maze, self.marked_blockage[chiplet], chiplet, cell_size)
        log.logger.debug(f"maze size: {maze.shape}")
        return maze

    def update_maze_blocks(self, maze, blocks, chiplet, cell_size):
        for (_, (col_1, row_1), _, (col_2, row_2), _) in blocks:
            min_x = min(col_1, col_2)
            max_x = max(col_1, col_2)
            min_y = min(row_1, row_2)
            max_y = max(row_1, row_2)
            x_min = int(float(min_x)/cell_size)
            y_min = self.die_max_y[chiplet] - int(float(max_y)/cell_size) - 1
            x_max = int(float(max_x)/cell_size)
            y_max = self.die_max_y[chiplet] - int(float(min_y)/cell_size) - 1
            maze[x_min:x_max, y_min:y_max] = 1
            log.logger.debug(f"Updating block from ({col_1}, {row_1}) to ({col_2}, {row_2}, ori coords: ({x_min}, {y_min}), ({x_max}, {y_max})")
            #print(f"Converted to maze indices: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
        #plot_maze_with_path(maze)
        return maze

    def update_canvas_w_connections(self):

        def shortest_manhattan_distance(points1, points2):
            points1 = np.array(points1)
            points2 = np.array(points2)
            distances = np.abs(points1[:, np.newaxis, :] - points2[np.newaxis, :, :]).sum(axis=2)

            min_index = np.unravel_index(np.argmin(distances), distances.shape)
            min_distance = distances[min_index]

            point1 = points1[min_index[0]]
            point2 = points2[min_index[1]]

            return min_distance, point1, point2


        if self.solution_classes is None or self.die_name is None or self.die_name not in self.solution_classes:
            return

        all_solutions = self.solution_classes[self.die_name]
        if not self.solution_connection:
            return

        #print(self.find_withtag('connection')) 
        self.delete("connection")
        if not self.checkbuttons['Show connections'].get():
            return
        self.connection_results.setdefault('distance', {}).setdefault('routing', {})
        visible_items = {item for item, var in self.checkbuttons.items() if var.get()}

        visible_solution = set()
        for solutions, solution_obj_list in all_solutions.items():
            if solutions not in self.legend_items:
                if solutions in visible_items:
                    visible_solution.add(solutions)
            else:
                legend = self.legend_items[solutions]
                for item, (color, shape, visible) in legend.items():
                    if item in visible_items:
                        visible_solution.add(solutions)
                        break

        #print(visible_solution)
        offset=self.die_max_y[self.die_name] * self.cell_size
        maze = self.create_maze()
        for solution in visible_solution:
            if solution not in self.solution_connection:
                continue
            for pair in self.solution_connection[solution]:
                n1 = pair[0]
                n2 = pair[1]

                n1_pts = get_conn_location(n1)
                n2_pts = get_conn_location(n2)
                #print(n1_pts, n2_pts)                
                min_dist, n1_pt, n2_pt = shortest_manhattan_distance(n1_pts, n2_pts)
                maze_start = (n1_pt[0]/self.cell_size, self.die_max_y[self.die_name] - n1_pt[1]/self.cell_size)
                maze_start = tuple(map(int, maze_start))

                maze_goal = (n2_pt[0]/self.cell_size, self.die_max_y[self.die_name] - n2_pt[1]/self.cell_size)
                maze_goal = tuple(map(int, maze_goal))
                if (maze_start, maze_goal) in self.connection_results['distance']['routing']:
                    path, distance = self.connection_results['distance']['routing'][(maze_start, maze_goal)][0], self.connection_results['distance']['routing'][(maze_start, maze_goal)][1]
                else:
                    path, total_distance = bfs(maze, maze_start, maze_goal)
                    self.connection_results['distance']['routing'][(maze_start, maze_goal)] = [path, total_distance]
                #print(path)
                if path:
                    coords = []
                    for cur_pt, next_pt in zip(path, path[1:]):
                        x1, y1 = cur_pt[0]*self.cell_size*self.scale_factor , cur_pt[1]*self.cell_size*self.scale_factor
                        x2, y2 = next_pt[0]*self.cell_size*self.scale_factor , next_pt[1]*self.cell_size*self.scale_factor
                        coords.extend([x1, y1, x2, y2])
                    self.create_line(coords, tags='connection', fill='darkgreen', width=3)
                else:
                    print("No path found")

    def center_on_point(self, col, row, name):
        x = col * self.cell_size * self.scale_factor
        y = row * self.cell_size * self.scale_factor
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        move_x = x / (self.die_max_x[self.die_name] * self.cell_size * self.scale_factor) - 0.5 * canvas_width / (self.die_max_x[self.die_name] * self.cell_size * self.scale_factor)
        move_y = y / (self.die_max_y[self.die_name] * self.cell_size * self.scale_factor) - 0.5 * canvas_height / (self.die_max_y[self.die_name] * self.cell_size * self.scale_factor)
        self.xview_moveto(x / (self.die_max_x[self.die_name] * self.cell_size * self.scale_factor) - 0.5 * canvas_width / (self.die_max_x[self.die_name] * self.cell_size * self.scale_factor))
        self.yview_moveto(y / (self.die_max_y[self.die_name] * self.cell_size * self.scale_factor) - 0.5 * canvas_height / (self.die_max_y[self.die_name] * self.cell_size * self.scale_factor))

    def center_on_block(self, x1, y1, x2, y2, name):
        x = (x1 + x2) / 2 * self.cell_size * self.scale_factor
        y = (y1 + y2) / 2 * self.cell_size * self.scale_factor
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        self.xview_moveto(x / (self.die_max_x[self.die_name] * self.cell_size * self.scale_factor) - 0.5 * canvas_width / (self.die_max_x[self.die_name] * self.cell_size * self.scale_factor))
        self.yview_moveto(y / (self.die_max_y[self.die_name] * self.cell_size * self.scale_factor) - 0.5 * canvas_height / (self.die_max_y[self.die_name] * self.cell_size * self.scale_factor))

    def update_canvas_w_label(self):
        if self.die_name is None or self.solution_classes is None:
            return
        if self.die_name not in self.solution_classes:
            return
        all_solutions = self.solution_classes[self.die_name]

        for solution in all_solutions.keys():
            self.delete(solution)

        visible_items = {item for item, var in self.checkbuttons.items() if var.get()}
        offset = self.die_max_y[self.die_name] * self.cell_size
        #print("die max y:", self.die_max_y[self.die_name], self.cell_size)
        def compute_coords(coords):
            return [(coord[0] * self.scale_factor, (offset - coord[1]) * self.scale_factor) for coord in coords]


        def create_and_label_polygon(coords, color, text, tags):
            rect_id = self.create_polygon(coords, outline='black', fill=color, tags=tags)
            centroid_x = sum(x for x, y in coords) / len(coords)
            centroid_y = sum(y for x, y in coords) / len(coords)
            if self.checkbuttons['Show text'].get():
                if tags == "Preplace":
                    text = text.split('_')[-1]
                    self.create_text(centroid_x, centroid_y , text=text, fill='black', tags=tags)
                else:
                    self.create_text(centroid_x, centroid_y, text=text, fill='black', tags=tags)
            return rect_id

        def create_and_label_rectangle(loc, color, text, tags, obj=''):
            size_w, size_h = 2, 2
            if obj != '':
                CellName = getattr(obj, 'CellName', '')
                if CellName != '':
                    if CellName in self.ip_size:
                        size_w = self.ip_size[CellName]['width']
                        size_h = self.ip_size[CellName]['height']

            loc_x = loc[0] * self.scale_factor
            loc_y = (offset - loc[1]) * self.scale_factor
            size_w *= self.scale_factor
            size_h *= self.scale_factor
            if tags == "Bump":
                # Bump location is on center
                rect_id = self.create_rectangle(loc_x + size_w/2, loc_y + size_h/2 , loc_x - size_w/2, loc_y - size_h/2, fill=color, outline='black', tags=tags)
            else:
                # Others left bottom corner
                rect_id = self.create_rectangle(loc_x, loc_y, loc_x + size_w, loc_y - size_h, fill=color, outline='black', tags=tags)
            if self.checkbuttons['Show text'].get():
                if tags == "GPIO":
                    cell_name = getattr(obj, 'CellName')
                    if '_H' in cell_name:
                        self.create_text(loc_x, loc_y, text=text, fill='black', tags=tags)
                    else:
                        diff = 13
                        for i, char in enumerate(text):
                            self.create_text(loc_x, loc_y + i * diff, text=char, fill='black', tags=tags)
                elif tags == "Pclamp":
                    text = text.split('pclamp_')[-1]
                    self.create_text(loc_x, loc_y , text=text, fill='black', tags=tags)
                else:
                    self.create_text(loc_x, loc_y , text=text, fill='black', tags=tags)
            return rect_id

        def bind_tags(rect_id, obj):
            if 'Show labels' in self.checkbuttons and self.checkbuttons['Show labels'].get():
                self.tag_bind(rect_id, "<Enter>", lambda e, item=obj, rect_id=rect_id: self.on_enter(e, item, rect_id))
                self.tag_bind(rect_id, "<Leave>", lambda e, rect_id=rect_id: self.on_leave(e, rect_id))

        for solutions, solution_obj_list in all_solutions.items():
            if solutions not in self.legend_items:
                if solutions in visible_items:
                    for obj in solution_obj_list:
                        color = getattr(obj, 'Color', 'blue')
                        coords = compute_coords(getattr(obj, 'Coords', []))
                        if coords:
                            rect_id = create_and_label_polygon(coords, color, str(getattr(obj, 'Name', obj)), solutions)
                        else:
                            loc = getattr(obj, 'Location')
                            rect_id = create_and_label_rectangle(loc, color, str(getattr(obj, 'Name', obj)), solutions, obj=obj)
                        bind_tags(rect_id, obj)
                continue

            legend = self.legend_items[solutions]
            for item, (color, shape, visible) in legend.items():
                obj_list = self.solution_legend_objs[solutions][item]
                if item not in visible_items:
                    continue
                for obj in obj_list:
                    coords = compute_coords(getattr(obj, 'Coords', []))
                    if coords:
                        rect_id = create_and_label_polygon(coords, color, str(getattr(obj, 'Name', obj)), solutions)
                    else:
                        if shape == 'square':
                            loc = getattr(obj, 'Location')
                            rect_id = create_and_label_rectangle(loc, color, str(getattr(obj, 'Name', obj)), solutions, obj=obj)
                    bind_tags(rect_id, obj)

    def draw_floorplan(self):
        self.delete("blockage")
        self.delete("block")

        if self.die_name is None:
            return

        self.draw_grid()

        if self.core_region[self.die_name]:
            x1 = float(self.core_region[self.die_name][0][0])/self.cell_size
            y1 = self.die_max_y[self.die_name] -(float(self.core_region[self.die_name][0][1])/self.cell_size)
            x2 = float(self.core_region[self.die_name][2][0])/self.cell_size
            y2 = self.die_max_y[self.die_name] -(float(self.core_region[self.die_name][2][1])/self.cell_size)
            self.fill_rectangle(x1, y1, x2, y2, 'black', 'block', None, fill_flag=False)
        if self.die_region[self.die_name]:
            x1 = float(self.die_region[self.die_name][0][0])/self.cell_size
            y1 = self.die_max_y[self.die_name] -(float(self.die_region[self.die_name][0][1])/self.cell_size)
            x2 = float(self.die_region[self.die_name][2][0])/self.cell_size
            y2 = self.die_max_y[self.die_name] -(float(self.die_region[self.die_name][2][1])/self.cell_size)
            self.fill_rectangle(x1, y1, x2, y2, 'grey', 'block', None, fill_flag=False)

        for ((col_1, row_1), _, (col_2, row_2), _, _) in self.marked_blockage[self.die_name]:
            self.fill_rectangle(float(col_1)/self.cell_size, self.die_max_y[self.die_name] -(float(row_1)/self.cell_size), float(col_2)/self.cell_size, self.die_max_y[self.die_name] -(float(row_2)/self.cell_size), 'indianred', 'blockage', None)

        for name, block in self.marked_blocks[self.die_name].items():
            x1 = float(block.coords[0][0])/self.cell_size
            y1 = self.die_max_y[self.die_name] -(float(block.coords[0][1])/self.cell_size)
            x2 = float(block.coords[2][0])/self.cell_size
            y2 = self.die_max_y[self.die_name] -(float(block.coords[2][1])/self.cell_size)
            self.fill_rectangle(x1, y1, x2, y2, block.Color, 'block', name)

            if block.PinCoords is not None:
                for region in block.PinCoords:
                    x1 = float(region[0][0])/self.cell_size
                    y1 = self.die_max_y[self.die_name] -(float(region[0][1])/self.cell_size)
                    x2 = float(region[2][0])/self.cell_size
                    y2 = self.die_max_y[self.die_name] -(float(region[2][1])/self.cell_size)
                    if block.has_sensor == 1:
                        offset = 0.5
                        start_x = (x1-offset) * self.cell_size * self.scale_factor
                        start_y = (y1+offset) * self.cell_size * self.scale_factor
                        end_x = (x2+offset) * self.cell_size * self.scale_factor
                        end_y = (y2-offset) * self.cell_size * self.scale_factor
                        color = "white"
                        point_id = self.create_oval(start_x, start_y, end_x, end_y, fill=color, tags='block')
                        self.create_text((start_x+end_x)/2,(start_y+end_y)/2, text='s', fill="black", font=("Arial", int(8*self.scale_factor*5)), tags='block')
                    else:
                        self.fill_rectangle(x1, y1, x2, y2, 'dimgray', 'block', None)

    def on_click(self, event):
        if self.app.current_utility is None: return
        x = self.canvasx(event.x)
        y = self.canvasy(event.y)
        col = int(x / (self.cell_size * self.scale_factor))
        row = int(y / (self.cell_size * self.scale_factor))
        self.app.current_utility.on_click_canvas(col, row)

    def in_block_region(self, col, row):
        for name, block in self.marked_blocks[self.die_name].items():
            if (float(block.coords[0][0])/self.cell_size < (col+1)) \
                and (self.die_max_y[self.die_name] -float(block.coords[0][1])/self.cell_size > row) \
                and (float(block.coords[2][0])/self.cell_size > col) \
                and (self.die_max_y[self.die_name] -float(block.coords[2][1])/self.cell_size < (row+1)):
                return True
        return False

    def in_blockage_region(self, col, row):
        for ((x_min, y_min), _, (x_max, y_max), _, _) in self.marked_blockage[self.die_name]:
            if float(x_min)/self.cell_size < (col+1) and (self.die_max_y[self.die_name] -float(y_min)/self.cell_size) > row \
               and float(x_max)/self.cell_size > col and (self.die_max_y[self.die_name] -float(y_max)/self.cell_size) < (row+1):
                return True
        return False

    def in_core_region(self, col, row):
        x_min, y_min = self.core_region[self.die_name][0][0], self.core_region[self.die_name][0][1]
        x_max, y_max = self.core_region[self.die_name][2][0], self.core_region[self.die_name][2][1]
        if float(x_min)/self.cell_size < (col+1) and (self.die_max_y[self.die_name] -float(y_min)/self.cell_size) > row \
            and float(x_max)/self.cell_size > col and (self.die_max_y[self.die_name] -float(y_max)/self.cell_size) < (row+1):
            return True
        return False

    def fill_rectangle(self, x1, y1, x2, y2, ori_color, tag, name, fill_flag=True):
        color = transparent_color(ori_color, 1/self.color_mode)
        start_x = x1 * self.cell_size * self.scale_factor
        start_y = y1 * self.cell_size * self.scale_factor
        end_x = x2 * self.cell_size * self.scale_factor
        end_y = y2 * self.cell_size * self.scale_factor
        if fill_flag: self.create_rectangle(start_x, start_y, end_x, end_y, fill=color, tags=tag, outline=color)
        else:         self.create_rectangle(start_x, start_y, end_x, end_y, fill='',    tags=tag, outline=color, width=3)

        if name:
            rect_width = end_x - start_x
            font_size = int(10 * self.scale_factor * 5)
            char_width = font_size
            max_chars_per_line = max(1, int(rect_width // char_width))
            name_list = [name[i:i+max_chars_per_line] for i in range(0, len(name), max_chars_per_line)]
            y_offset = 8*self.cell_size * self.scale_factor
            idx = 0
            center_x = (x1*self.cell_size * self.scale_factor + x2*self.cell_size * self.scale_factor) // 2
            center_y = (y1*self.cell_size * self.scale_factor + y2*self.cell_size * self.scale_factor) // 2
            for s_name in name_list:
                self.create_text(center_x, center_y+y_offset*idx, text=s_name, fill=transparent_color('black', 1/self.color_mode), font=("Arial", int(11*self.scale_factor*5)), tags=tag)
                idx += 1
    def start_pan(self, event):
        self.scan_mark(event.x, event.y)

    def pan(self, event):
        self.scan_dragto(event.x, event.y, gain=1)

    def set_scale_factor(self, scale):
        new_scale_factor = float(scale) / 100
        if self.min_scale_factor <= new_scale_factor <= self.max_scale_factor:
            scale = new_scale_factor / self.scale_factor
            self.scale_factor = new_scale_factor
            self.scale("all", 0, 0, scale, scale)
            self.configure(scrollregion=self.bbox("all"))
            self.draw_floorplan()
            self.update_canvas_w_label()
            self.update_canvas_w_connections()
            if self.app.current_utility:
                self.app.current_utility.draw()


    def fit(self, event=None):
        self.xview_moveto(0)
        self.yview_moveto(0)
        self.set_scale_factor(7.5)


