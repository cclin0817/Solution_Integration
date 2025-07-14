
✅ 還原內容如下：

import tkinter as tk
from tkinter import messagebox, Menu, Button, filedialog, ttk, simpledialog
from collections import defaultdict
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon, MultiPolygon
from .base_utility import *
import matplotlib.pyplot as plt
from .functions import *
import numpy as np
import math
from .block import *
from . import log
import random
import json
import re

def plot_candidate_points(blocks, candidate_points, centers, radius, iteration):
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.update_idletasks()
    root.destroy()
    fig_width = screen_width / 200
    fig_height = screen_height / 100
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot blocks
    for b in blocks:
        x1, y1, x2, y2 = b.coords[0][0], b.coords[0][1], b.coords[2][0], b.coords[2][1]
        rectangle = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rectangle)

    # Plot candidate points
    for point in candidate_points.keys():
        ax.plot(point[0], point[1], 'go')  # Plot candidate points in green

    # Plot circles
    for center in centers.keys():
        circle = plt.Circle(center, radius, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(circle)
        ax.plot(center[0], center[1], 'bo')  # Plot the center of the circle

    ax.set_aspect('equal', 'box')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title(f'Iteration {iteration} - Candidate Points and Circles')
    plt.show()

def find_candidate_for_dedicated_domain(blocks):
    candidate_points = defaultdict(str)
    centers = defaultdict(str)
    for block in blocks:
        # Sort points based on x and y coordinates
        points_sorted = sorted(block.coords, key=lambda point: (point[0], point[1]))
        # Identify the corners
        ll = tuple(min(points_sorted, key=lambda point: (point[0], point[1])))
        lr = tuple(max(points_sorted, key=lambda point: (point[0], -point[1])))
        ul = tuple(min(points_sorted, key=lambda point: (point[0], -point[1])))
        ur = tuple(max(points_sorted, key=lambda point: (point[0], point[1])))
        candidate_points.update({ll:'LL', lr:'LR', ul:'UL', ur:'UR'})

    default_1 = min(candidate_points.keys(), key=lambda point: (point[0], point[1]))
    default_2 = max(candidate_points.keys(), key=lambda point: (point[0], point[1]))
    centers[default_1] = 'LL'
    centers[default_2] = 'UR'

    return candidate_points, centers

def cover_with_circles(blocks, net, radius):
    log.logger.info(f"Created pclamp for power domain {net}")
    candidate_points, centers = find_candidate_for_dedicated_domain(blocks)
    if (len(candidate_points) == 0): exit(1)

    uncovered_polygons = [block.polygon for block in blocks]
    for center in centers:
        circle = Point(center).buffer(radius)
        uncovered_polygons = [poly.difference(circle) for poly in uncovered_polygons]
        uncovered_polygons = [part for poly in uncovered_polygons for part in (poly.geoms if isinstance(poly, MultiPolygon) else [poly])]
        uncovered_polygons = [poly for poly in uncovered_polygons if not poly.is_empty]

    min_uncovered_count = float('inf')
    iteration = 0
    while uncovered_polygons:
        # Plot candidate points for debugging
        # plot_candidate_points(blocks, candidate_points, centers, radius, iteration)
        log.logger.info(f"Iteration {iteration}: Number of circles = {len(centers)}, Remaining polygons = {len(uncovered_polygons)}")
        iteration = iteration + 1
        best_point = None
        point_direction = None
        max_covered_area = 0

        for point, direction in candidate_points.items():
            circle = Point(point[0], point[1]).buffer(radius)
            covered_area = sum([circle.intersection(poly).area for poly in uncovered_polygons])

            new_uncovered_polygons = [poly.difference(circle) for poly in uncovered_polygons]
            new_uncovered_polygons = [part for poly in new_uncovered_polygons for part in (poly.geoms if isinstance(poly, MultiPolygon) else [poly])]
            new_uncovered_polygons = [poly for poly in new_uncovered_polygons if not poly.is_empty]
            uncovered_count = len(new_uncovered_polygons)

            if uncovered_count < min_uncovered_count:
                best_point = point
                point_direction = direction
                max_covered_area = covered_area
                min_uncovered_count = uncovered_count
            elif uncovered_count == min_uncovered_count:
                if covered_area > max_covered_area:
                    best_point = point
                    point_direction = direction
                    max_covered_area = covered_area
                elif covered_area == max_covered_area:
                    if best_point is None:
                        best_point = point
                        point_direction = direction
                    else:
                        point_distance_sum = sum(np.linalg.norm(np.array([point[0], point[1]]) - np.array(center))**2 for center in centers.keys())
                        best_point_distance_sum = sum(np.linalg.norm(np.array([best_point[0], best_point[1]]) - np.array(center))**2 for center in centers.keys())
                        if point_distance_sum < best_point_distance_sum:
                            best_point = point
                            point_direction = direction

        if best_point:
            centers[(best_point[0], best_point[1])] = point_direction
            circle = Point(best_point[0], best_point[1]).buffer(radius)
            uncovered_polygons     = [poly.difference(circle) for poly in uncovered_polygons]
            uncovered_polygons     = [part for poly in uncovered_polygons for part in (poly.geoms if isinstance(poly, MultiPolygon) else [poly])]
            uncovered_polygons     = [poly for poly in uncovered_polygons if not poly.is_empty]
            uncovered_count = len(uncovered_polygons)

    # plot_candidate_points(blocks, candidate_points, centers, radius, iteration)
    return centers

class PclampUtility(BaseUtility):
    def __init__(self, app, canvas, args):
        super().__init__(app, canvas, args)
        self.marked_points = {}
        self.marked_blocks = self.canvas.marked_blocks
        for name, block in self.marked_blocks[self.canvas.die_name].items():
            if len(block.TopPowerNets) == 0:
                log.logger.info(f"Block {name} has no power net")
            if block.Name == "ODP_BD_A16ETV5_2025": continue
            if 'VDD_SOC' not in block.TopPowerNets:
                block.TopPowerNets.append('VDD_SOC')

        self.dist_max = simpledialog.askinteger("Input", "Set pclamp protection coverage:", initialvalue=1250)
        log.logger.info(f"pclamp protection coverage is {self.dist_max}")

        self.covered_by_net = defaultdict(lambda: defaultdict(bool))
        self.output_dir = self.app.output_dir + '/pclamp/'
        os.makedirs(self.output_dir, exist_ok=True)

    def create_widgets(self, parent_frame):
        label = tk.Label(parent_frame, text="Pclamp planning")
        label.pack()

        self.save_button = tk.Button(parent_frame, text="Save to JSON", command=self.save_to_json)
        self.save_button.pack(fill="x")

        self.load_button = tk.Button(parent_frame, text="Load from JSON", command=self.load_from_json)
        self.load_button.pack(fill="x")

        self.gen_file_button = tk.Button(parent_frame, text="Generate files", command=self.gen_file)
        self.gen_file_button.pack(fill="x")

        self.gen_file_button = tk.Button(parent_frame, text="Auto place", command=self.auto_place)
        self.gen_file_button.pack(fill="x")

        self.pointView = ttk.Treeview(parent_frame, columns=("name", "coordinates", "layout_coor", "asso_power", "direction"), show="headings")
        self.pointView.column("name", width = 50)
        self.pointView.column("coordinates", width = 150)
        self.pointView.column("layout_coor", width = 150)
        self.pointView.column("asso_power", width = 150)
        self.pointView.column("direction", width = 50)
        self.pointView.heading("name", text="Name")
        self.pointView.heading("coordinates", text="Coordinates in GUI")
        self.pointView.heading("layout_coor", text="Real loaction")
        self.pointView.heading("asso_power", text="Associated power")
        self.pointView.heading("direction", text="Direction")
        self.pointView.pack(side="top", fill="both", expand=True)
        self.pointView.bind("<<TreeviewSelect>>", self.on_pointView_select)
        self.pointView.bind("<Button-3>", self.on_click_point_view)
        self.pointView.bind("<w>", lambda event: self.move_selected_point(0, -1))
        self.pointView.bind("<s>", lambda event: self.move_selected_point(0, 1))
        self.pointView.bind("<a>", lambda event: self.move_selected_point(-1, 0))
        self.pointView.bind("<d>", lambda event: self.move_selected_point(1, 0))

        self.powerView = ttk.Treeview(parent_frame, columns=("name"), show="headings")
        self.powerView.column("name", width = 150)
        self.powerView.heading("name", text="Power nets")
        self.powerView.pack(side="top", fill="both", expand=True)
        self.powerView.bind("<<TreeviewSelect>>", self.on_powerView_select)

        self.return_button = tk.Button(parent_frame, text="Return to utility selection", command=self.app.show_selection_screen)
        self.return_button.pack(fill="x")

        self.context_menu = Menu(parent_frame, tearoff=0)
        self.context_menu.add_command(label="Delete", command=self.delete_point)
        self.context_menu.add_command(label="Change direction", command=self.change_direction)

        self.update_power_list(self.marked_blocks[self.canvas.die_name])

    def update_power_list(self, blocks):
        for i in self.powerView.get_children():
            self.powerView.delete(i)
        net_names = []
        for name, block in blocks.items():
            for power_name in block.TopPowerNets:
                if power_name not in net_names:
                    net_names.append(power_name)

        net_names.sort()
        self.powerView.insert("", "end", values = "None")
        for name in net_names:
            self.powerView.insert("", "end", values = (name))

    def update_point_list(self, points):
        for i in self.pointView.get_children():
            self.pointView.delete(i)
        sorted_points = sorted(points.items(), key=lambda item: int(item[1].name))
        for (col, row), point in sorted_points:
            self.pointView.insert("", "end", values=(point.name, f"({col}, {row})", f"({col*self.canvas.cell_size}, {(self.canvas.die_max_y[self.canvas.die_name]-row)*self.canvas.cell_size})", f"{point.asso_power}", f"{point.direction}"))

    def on_powerView_select(self, event):
        selection = self.powerView.selection()
        if selection:
            item = self.powerView.item(selection[0])
            power_name = item["values"][0]
            self.update_coverage_graph(power_name)
            log.logger.info(f"Select net {power_name}")

    def update_coverage_graph(self, power_name):
        self.check_coverage(power_name)
        self.highlight_power_nets(power_name)
        self.draw_points(power_name)

    def draw(self):
        selection = self.powerView.selection()
        if selection:
            item = self.powerView.item(selection[0])
            power_name = item["values"][0]
        else:
            power_name = 'None'
        self.update_coverage_graph(power_name)
        self.draw_points(power_name)

    def check_coverage(self, power_name):
        circles = []
        for (colx, rowy), cell in self.marked_points.items():
            if cell.asso_power == power_name:
                x, y = (colx*self.canvas.cell_size, (self.canvas.die_max_y[self.canvas.die_name]-rowy)*self.canvas.cell_size)
                new_circle = Point(x, y).buffer(self.dist_max)
                circles.append(new_circle)

        for name, block in self.marked_blocks[self.canvas.die_name].items():
            pname = [name] + block.TopPowerNets
            if power_name in block.TopPowerNets:
                rect = block.polygon
                new_union_circles = unary_union(circles)
                if rect.within(new_union_circles):
                    self.covered_by_net[name][power_name] = True
                else:
                    self.covered_by_net[name][power_name] = False

        self.canvas.delete("core_region")
        if power_name == 'VDD_SOC':
            core_region = self.canvas.core_region[self.canvas.die_name]
            rect = box(core_region[0][0], core_region[2][1], core_region[2][0], core_region[0][1])
            new_union_circles = unary_union(circles)
            if rect.within(new_union_circles):
                x1 = float(core_region[0][0])/self.canvas.cell_size
                y1 = self.canvas.die_max_y[self.canvas.die_name]-(float(core_region[0][1])/self.canvas.cell_size)
                x2 = float(core_region[2][0])/self.canvas.cell_size
                y2 = self.canvas.die_max_y[self.canvas.die_name]-(float(core_region[2][1])/self.canvas.cell_size)
                self.canvas.fill_rectangle(x1, y1, x2, y2, 'lime', 'core_region', None, fill_flag=True)

    def highlight_power_nets(self, power_net_name):
        for name, block in self.marked_blocks[self.canvas.die_name].items():
            x1 = float(block.coords[0][0])/self.canvas.cell_size
            y1 = self.canvas.die_max_y[self.canvas.die_name]-(float(block.coords[0][1])/self.canvas.cell_size)
            x2 = float(block.coords[2][0])/self.canvas.cell_size
            y2 = self.canvas.die_max_y[self.canvas.die_name]-(float(block.coords[2][1])/self.canvas.cell_size)
            pname = name

            if power_net_name == "None":
                self.canvas.fill_rectangle(x1, y1, x2, y2, block.Color, 'block', pname)
            else:
                if power_net_name in block.TopPowerNets:
                    if power_net_name in self.covered_by_net[name].keys() and self.covered_by_net[name][power_net_name] == True:
                        self.canvas.fill_rectangle(x1, y1, x2, y2, block.Color, 'block', pname)
                    else:
                        self.canvas.fill_rectangle(x1, y1, x2, y2, 'white', 'block', pname)
                else:
                    self.canvas.fill_rectangle(x1, y1, x2, y2, darken_color(block.Color), 'block', pname)

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
                        end_x = (x2+offset)   * self.canvas.cell_size * self.canvas.scale_factor
                        end_y = (y2-offset)   * self.canvas.cell_size * self.canvas.scale_factor
                        color = "white"
                        self.canvas.create_oval(start_x, start_y, end_x, end_y, fill=color, tags='block')
                        self.canvas.create_text((start_x+end_x)/2,(start_y+end_y)/2, text='s', fill="black", font=("Arial", int(8*self.canvas.scale_factor*5)), tags='block')
                    else:
                        self.canvas.fill_rectangle(x1, y1, x2, y2, 'dimgray', 'block', None)
        self.canvas.update_idletasks()

    def move_selected_point(self, dx, dy):
        if self.is_point_moved(dx, dy, True):
            self.update_coverage_graph(self.marked_points[self.selected_point].asso_power)

    def delete_point(self):
        selection = self.pointView.selection()
        if selection:
            item = self.pointView.item(selection[0])
            point_str = item["values"][1]
            coord_part = point_str[1:-1]
            col, row = map(int, coord_part.split(", "))
            if (col, row) in self.marked_points:
                power_name = self.marked_points[(col, row)].asso_power
                del self.marked_points[(col, row)]
                self.rename_points()
                self.update_coverage_graph(power_name)

    def change_direction(self):
        selection = self.pointView.selection()
        if selection:
            item = self.pointView.item(selection[0])
            point_str = item["values"][1]
            coord_part = point_str[1:-1]
            col, row = map(int, coord_part.split(", "))
            if (col, row) in self.marked_points:
                point = self.marked_points[(col, row)]
                point.direction = 'H' if point.direction == 'V' else 'V'
                self.update_point_list(self.marked_points)

    def on_click_canvas(self, col, row):
        if (col, row) in self.marked_points: return
        if self.canvas.in_block_region(col, row): return
        if not self.canvas.in_core_region(col, row): return
        selection = self.powerView.selection()
        if selection:
            item = self.powerView.item(selection[0])
            power_name = item["values"][0]
            if power_name == "None":
                messagebox.showwarning("Warning", f"Select power net first.")
                return
            dialog = CustomDirectionDialog(self.canvas)
            direction = dialog.result
            if direction == None: return
            self.create_point(col, row, power_name, direction)
            self.update_coverage_graph(power_name)
        else:
            messagebox.showwarning("Warning", f"Select power net first.")
            return

    def create_point(self, col, row, asso_power, direction, name = None):
        if name == None:
            name = str(len(self.marked_points))
        self.marked_points[(col, row)] = PclampPoint(name, asso_power, direction)
        self.fill_cell(col, row, name, "point")
        self.update_point_list(self.marked_points)

    def save_to_json(self):
        file_path = filedialog.asksaveasfilename(initialfile='pclamp_loc.json', initialdir=self.output_dir, defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            data = {f"{col},{row}": {"name": marked_point.name, "asso_power": marked_point.asso_power, "direction": marked_point.direction} for (col, row), marked_point in self.marked_points.items()}
            with open(file_path, "w") as json_file:
                json.dump(data, json_file, indent=5)

    def load_from_json(self):
        file_path = filedialog.askopenfilename(initialdir=self.output_dir, filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
                self.marked_points.clear()
                for coord, info in data.items():
                    col, row = map(int, coord.split(","))
                    self.create_point(col, row, asso_power=info.get("asso_power", ""), direction=info.get("direction", ""), name=info['name'])

    def get_snap_loc(self, col, row, x_offset, y_offset):
        x_loc = col*self.canvas.cell_size
        y_loc = (self.canvas.die_max_y[self.canvas.die_name]-row)*self.canvas.cell_size
        x_snap = round(int(x_loc/x_offset)*x_offset, 4)
        y_snap = round(int(y_loc/y_offset)*y_offset, 4)
        return x_snap, y_snap

    def gen_file(self):
        sorted_points = sorted(self.marked_points.items(), key=lambda item: int(item[1].name))
        file_path = filedialog.asksaveasfilename(initialfile='pclamp_loc.tcl', initialdir=self.output_dir, defaultextension=".tcl", filetypes=[("TCL files", "*.tcl")])
        if file_path:
            with open(file_path, "w") as tcl_file:
                points_by_net = {}
                for (col, row), marked_point in sorted_points:
                    #marked_point.physical_x, marked_point.physical_y = self.get_snap_loc(col, row, 0.048, 0.013)
                    marked_point.physical_x, marked_point.physical_y = self.get_snap_loc(col, row, 0.048, 0.052)
                    if marked_point.asso_power not in points_by_net:
                        points_by_net[marked_point.asso_power] = []
                    points_by_net[marked_point.asso_power].append(marked_point)

                for power, points in points_by_net.items():
                    #print(f"Association Power: {power}")
                    for idx, point in enumerate(points):
                        inst_name = 'pclamp__'  + point.asso_power + '__VSS__' + str(idx)
                        cell_name = 'PCLAMPC_H' if point.direction == 'H' else 'PCLAMPC_V'
                        #cell_name = 'PCLAMPCCOD_H' if point.direction == 'H' else 'PCLAMPCCOD_V' #A16, size around 26*50
                        print(f"addInst -physical -cell {cell_name} -inst {inst_name} -loc {point.physical_x} {point.physical_y}")
                        tcl_file.write(f"addInst -physical -cell {cell_name} -inst {inst_name} -loc {point.physical_x} {point.physical_y}\n")

        file_path = filedialog.asksaveasfilename(initialfile='pclamp_loc_for_merge.json', initialdir=self.output_dir, defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            result = {}
            result[self.canvas.die_name]={}
            result[self.canvas.die_name]['Pclamp'] = []
            for power, points in points_by_net.items():
                for idx, point in enumerate(points):
                    pclamp_dict = {}
                    inst_name = 'pclamp__'  + point.asso_power + '__VSS__' + str(idx)
                    cell_name = 'PCLAMPC_H' if point.direction == 'H' else 'PCLAMPC_V'
                    #cell_name = 'PCLAMPCCOD_H' if point.direction == 'H' else 'PCLAMPCCOD_V' #A16, size around 26*50
                    pclamp_dict['Name'] = inst_name
                    pclamp_dict['Location'] = [point.physical_x, point.physical_y]
                    pclamp_dict['CellName'] = cell_name
                    pclamp_dict['TopPowerNets'] = [point.asso_power]
                    pclamp_dict['TopGroundNets'] = ['VSS']
                    pclamp_dict['Legend_key'] = cell_name
                    result[self.canvas.die_name]['Pclamp'].append(pclamp_dict)
            with open(file_path, "w") as json_file:
                json.dump(result, json_file, indent=5)


    def auto_place(self):
        def get_random_new_point(x, y, i, j, marked_points, direction):
            # In canvas, (0, 0) in upper-left corner
            if direction in ['LL']:
                offsets = [(-i, j), (-i, -j), (0, -j), (i, 0), (i, j)]
            elif direction in ['LR']:
                offsets = [(-i, 0), (0, -j), (i, -j), (i, j), (-i, j)]
            elif direction in ['UL']:
                offsets = [(-i, j), (-i, -j), (i, -j), (i, 0), (0, j)]
            elif direction in ['UR']:
                offsets = [(-i, 0), (-i, -j), (i, -j), (i, j), (0, j)]

            new_point = (x, y)
            while new_point in marked_points or self.canvas.in_block_region(new_point[0], new_point[1]):
                offset = random.choice(offsets)
                new_point = (x + offset[0], y + offset[1])
            return new_point[0], new_point[1]

        blocks_by_net = defaultdict(list)
        for name, block in self.marked_blocks[self.canvas.die_name].items():
            for net in block.TopPowerNets:
                blocks_by_net[net].append(block)

        skip_nets_patterns = [re.compile(pattern.replace('*', '.*')) for pattern in['VDD_SOC', '*ODP*', '*PLL*', '*eFuse*']]
        for net, blocks in blocks_by_net.items():
            #if any(pattern.match(net) for pattern in skip_nets_patterns): continue
            if 'FCCC' not in net: continue
            centers = cover_with_circles(blocks, net, self.dist_max)
            for center, direction in centers.items():
                x = float(center[0])/self.canvas.cell_size
                y = self.canvas.die_max_y[self.canvas.die_name]-(float(center[1])/self.canvas.cell_size)
                if direction in ['LL']:
                    x = math.floor(x)
                    y = math.floor(y)
                elif direction in ['LR']:
                    x = math.ceil(x)
                    y = math.floor(y)
                elif direction in ['UL']:
                    x = math.floor(x)
                    y = math.ceil(y)
                elif direction in ['UR']:
                    x = math.ceil(x)
                    y = math.ceil(y)
                if (x, y) in self.marked_points:
                    x, y = get_random_new_point(x, y, 2, 2, self.marked_points, direction)
                self.create_point(x, y, asso_power=net, direction='V')
        log.logger.info(f"Auto place finished")

    def fill_cell(self, col, row, name, tag, color="tomato", size=15, power="None"):
        if power != "None":
            if self.marked_points[(col, row)].asso_power != power:
                color = "mistyrose"

        x1 = col * self.canvas.cell_size * self.canvas.scale_factor
        y1 = row * self.canvas.cell_size * self.canvas.scale_factor
        x2 =  x1 + self.canvas.cell_size * self.canvas.scale_factor
        y2 =  y1 + self.canvas.cell_size * self.canvas.scale_factor

        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, tags=tag)

        # Draw the point with a fixed size
        fixed_size = size  # The size of the point that will remain fixed
        self.canvas.create_oval(
            x1 + self.canvas.cell_size * self.canvas.scale_factor / 2 - fixed_size / 2,
            y1 + self.canvas.cell_size * self.canvas.scale_factor / 2 - fixed_size / 2,
            x1 + self.canvas.cell_size * self.canvas.scale_factor / 2 + fixed_size / 2,
            y1 + self.canvas.cell_size * self.canvas.scale_factor / 2 + fixed_size / 2,
            fill=color,
            tags=tag
        )
        if hasattr(self.marked_points[(col, row)], 'asso_power') and  self.marked_points[(col, row)].asso_power == power:
            self.canvas.create_oval(
                x1 - self.canvas.scale_factor * self.dist_max,
                y1 - self.canvas.scale_factor * self.dist_max,
                x1 + self.canvas.scale_factor * self.dist_max,
                y1 + self.canvas.scale_factor * self.dist_max,
                fill="",
                tags=tag
            )
        self.canvas.create_text((col+0.5)*self.canvas.cell_size * self.canvas.scale_factor, (row+0.5)*self.canvas.cell_size * self.canvas.scale_factor, text=name, fill="black", font=("Arial", max(12,int(8*self.canvas.scale_factor*5))), tags=tag)


