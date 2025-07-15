
✅ 還原內容如下：

import matplotlib
matplotlib.use('TkAgg')
from .functions import *
from .obj import *
import time
import re
from multiprocessing import Pool
import dill as pickle
import shapely
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union, voronoi_diagram
from .base_utility import *
import json
from tkinter import Toplevel, StringVar, OptionMenu, Button, messagebox, Text, TclError
import tkinter as tk
from tkinter import simpledialog, Scrollbar, Scale, HORIZONTAL, VERTICAL, Menu, filedialog, ttk, font
from tkinter.ttk import Treeview
from tkinter import Toplevel, StringVar, OptionMenu, Button, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon as PlotPolygon
import matplotlib.patches as patches
import matplotlib.cm as cm
from rtree import index
from matplotlib.colors import Normalize
from .json_encoder import *
from . import log
from shapely.geometry.base import BaseGeometry
import random

class QoRChecker(BaseUtility):

    def __init__(self, app, canvas, args=None):
        super().__init__(app, canvas, args=args)

        self.qor_items = {}
        self.qor_results = {}
        self.output_dir = {}
        self.checkbuttons = {}


        # violation window for all kinds of qor
        self.violation_tree = None
        self.violation_window = None
        self.detail_text = None

        # checking items for each checker
        # distance, coverage, ...
        self.qor_items = {}

        # all tags for each checker
        self.qor_tags = {}

        # qor results need to keep after processing
        self.qor_results = {}

        # for preventing mouse wheel conflict
        self.prev_selected = None
        self.double_click_processing = False

        # For sampling
        self.bkg_points = {}

        self.make_output_dir()

        raw_qor_check_files = read_files_from_directory(self.args.input_dir) or []
        checker_enabled = self.parse_checkers()
        #print(checker_enabled)
        self.parse_qor_list(raw_qor_check_files)

        # Execute all checkers
        if 'distance' in checker_enabled:
            self.distance_qor()
        if 'coverage' in checker_enabled:
            self.coverage_qor_update()
        if 'overlap' in checker_enabled:
            self.overlap_qor()
        if 'bump_current' in checker_enabled:
            self.bump_current_qor()

    def on_click_canvas(self, x, y):
        pass

    def update_qor_legend(self, parent_frame, chiplet=None):
        max_width = 450  # Define the maximum width for wrapping
        current_width = 0

        current_row_frame = tk.Frame(parent_frame)
        current_row_frame.pack(fill="x", pady=5)
        all_qor_results = self.qor_results
        if chiplet is None:
            return
        #print('All',all_qor_results)
        for qor_items, qor_results in all_qor_results.items():
            cur_legend_frame = tk.LabelFrame(current_row_frame, text='QoR '+qor_items)
            cur_legend_frame.pack(side="left", padx=5, pady=5)
            if chiplet not in qor_results:
                continue
            qor_for_chiplet = qor_results[chiplet]
            if 'Legend' not in qor_for_chiplet:
                print('No legend skip')
                continue
            #else:
            #    print(f"Legend in update_qor_legend: {qor_for_chiplet['Legend']}")
            frame_item = ttk.Frame(cur_legend_frame)
            frame_item.pack(anchor='w', pady=2)

            current_check_frame = ttk.Frame(frame_item)
            current_check_frame.pack(anchor='w')

            for legend in qor_for_chiplet['Legend']:
                var = tk.BooleanVar(value=False)
                txt = legend
                if isinstance(legend, tuple) and len(legend) == 2:
                    txt = legend[1]
                default_font = font.nametofont("TkDefaultFont")
                # Measure the text width
                text_width = default_font.measure(txt)
                # Check if adding this checkbutton would exceed the max_width
                if current_width + text_width > max_width:
                    # Start a new row
                    current_check_frame = ttk.Frame(frame_item)
                    current_check_frame.pack(anchor='w')
                    current_width = 0

                checkbutton = ttk.Checkbutton(current_check_frame, text=txt, variable=var, command=self.draw)

                # Use the font measure function to determine the width of the text
                text_width = checkbutton.winfo_reqwidth()


                checkbutton.pack(side="left")
                current_width += text_width  # Update the current width
                self.checkbuttons[legend] = var

            # Update the current width and check if a new row is needed
            #current_width += cur_legend_frame.winfo_reqwidth() + 225
            #print(solution_name, current_width)
            #if current_width > max_width:
            current_row_frame = tk.Frame(parent_frame)
            current_row_frame.pack(fill="x", pady=5)
            current_width = 0
            #continue 

    def clear_canvas(self):
        for key, tag_list in self.qor_tags.items():
            for tag in tag_list:
                self.canvas.delete(tag)
        self.qor_tags = {}
        self.prev_selected = None
        self.double_click_processing = False
        self.selected_items = {}
        # violation window for all kinds of qor
        self.canvas.delete_violation_window()
        self.violation_tree = None
        self.detail_text = None
        self.clear_all_checkbuttons()

    def clear_all_checkbuttons(self):
        for var in self.checkbuttons.values():
            var.set(0)

    def delete_violation_window(self):
        if self.violation_window is not None and self.violation_window.winfo_exists():
            self.violation_window.destroy()
            self.violation_window = None
            print("Violation window destroyed.")


    def create_widgets(self, parent_frame):
        # Distance qor
        #self.load_button = tk.Button(parent_frame, text="Load distance check list", command=self.import_check_list, width=1, height=1, wraplength=450)
        #self.load_button.pack(fill="x")
        var = tk.BooleanVar(value=False)
        checkbutton = ttk.Checkbutton(parent_frame, text="Show distance check", variable=var, command=self.update_canvas_distance_qor)
        checkbutton.pack(side="top", fill="x")
        self.checkbuttons['Show distance check'] = var

        # Coverage qor
        #self.load_button = tk.Button(parent_frame, text="Load coverage check list", command=self.import_coverage_list, width=1, height=1, wraplength=450)
        #self.load_button.pack(fill="x")
        var = tk.BooleanVar(value=False)
        checkbutton = ttk.Checkbutton(parent_frame, text="Show coverage check", variable=var, command=self.update_canvas_coverage_qor)
        checkbutton.pack(side="top", fill="x")
        self.checkbuttons['Show coverage check'] = var

        var = tk.BooleanVar(value=False)
        checkbutton = ttk.Checkbutton(parent_frame, text="Show overlap check", variable=var, command=self.update_canvas_overlap_qor)
        checkbutton.pack(side="top", fill="x")
        self.checkbuttons['Show overlap check'] = var


        self.update_qor_legend(parent_frame, chiplet=self.canvas.die_name)
        self.draw()
        self.return_button = tk.Button(parent_frame, text="Return to utility selection", command=self.app.show_selection_screen)
        self.return_button.pack(fill="x")

    def update_widgets(self, parent_frame, chiplet=None):
        for widget in parent_frame.winfo_children():
            widget.destroy()
        self.create_widgets(parent_frame)

    def draw(self):
        self.update_canvas_distance_qor()
        self.update_canvas_coverage_qor()
        self.update_canvas_overlap_qor()
    def parse_qor_list(self, raw_qor_check_files):
        #print("raw_qor:", raw_qor_check_files)
        #if not raw_qor_check_files:
        #    messagebox.showinfo(f"Info", f"No files to display. {raw_qor_check_files}")
        #    return
        for file_path in raw_qor_check_files:
            #print(file_path)
            if 'routing' in file_path or 'manhattan' in file_path:
                self.import_check_list(file_path)
            if 'coverage' in file_path:
                self.import_coverage_list(file_path)
            if 'overlap' in file_path:
                self.import_overlap_list(file_path)
            if 'bump_current' in file_path:
                self.import_bump_current_list(file_path)
        file_name = './QoR_criteria.json'

        with open(file_name, 'w') as json_file:
            json.dump(self.qor_items, json_file, indent=4, cls=CustomEncoder)

    def parse_checkers(self):
        if self.args.checker:
            checkers = self.args.checker.split('+')
            available_checker = ['distance', 'coverage', 'overlap' ,'all', 'skip', 'bump_current']
            for checker in checkers:
                if checker not in available_checker:
                    raise ValueError(f"No this checker: {checker}, current enabled checker: {available_checker}, please specify like: -c distance+coverage")
                if checker == 'all':
                    checkers = ['distance', 'coverage', 'overlap', 'bump_current']
                    break
                if checker == 'skip':
                    checkers = []
                    break
            print(f"Running specified checker: {checkers}")
        else:
            checkers = []
            print("Skip qor checkers")
        #print(checkers, self.args.checker)
        return checkers


    def import_coverage_list(self, file_path=None):
        #file_path = filedialog.askopenfilename(filetypes=[("txt files", "*.txt")])
        if not file_path: return
        if 'coverage' not in self.qor_items:
            self.qor_items['coverage'] = {}
        chiplet_name = file_path.split('/')[-1].split('.')[0]
        if chiplet_name not in self.qor_items['coverage']:
            self.qor_items['coverage'][chiplet_name] = {'QoR criteria':[]}
        data = []
        with open(file_path, 'r') as input_file:
            for line in input_file:
                line = line.strip()
                #print(line)
                parts = line.split(',')
                #print(parts)
                if len(parts) == 6:
                    parts = [part.strip() for part in parts]
                    self.qor_items['coverage'][chiplet_name]['QoR criteria'].append(parts)
                elif len(parts) == 0:
                    continue
                else:
                    raise ValueError(f"Line format error: {line.strip()}")

        if 'coverage' not in self.qor_results:
            self.qor_results['coverage'] = {}


        if chiplet_name not in self.qor_results['coverage']:
            self.qor_results['coverage'][chiplet_name] = {'QoR results': {}}
            self.qor_results['coverage'][chiplet_name]['QoR results'] = {'Chip domain': set()}
            for pd_name in self.canvas.data['S1_output'][chiplet_name]['TOPShape']['PowerNets']:
                self.qor_results['coverage'][chiplet_name]['QoR results']['Chip domain'].add(pd_name)
            for pd_name in self.canvas.data['S1_output'][chiplet_name]['TOPShape']['GroundNets']:
                self.qor_results['coverage'][chiplet_name]['QoR results']['Chip domain'].add(pd_name)

        for name, block_obj in self.canvas.marked_blocks[chiplet_name].items():
            #print(block_obj.TopPowerNets)
            if len(block_obj.TopPowerNets) > 0:
                for power_nets in block_obj.TopPowerNets:
                    self.qor_results['coverage'][chiplet_name]['QoR results']['Chip domain'].add(power_nets)
                for gnd_nets in block_obj.TopGroundNets:
                    self.qor_results['coverage'][chiplet_name]['QoR results']['Chip domain'].add(gnd_nets)
        #print("All power nets detected: ", self.qor_results['coverage'][self.canvas.die_name])


    def import_check_list(self, file_path=None):
        #file_path = filedialog.askopenfilename(filetypes=[("txt files", "*.txt")])
        if not file_path: return
        if 'routing' in file_path:
            mode = 'routing'
        else:
            mode = 'manhattan'
        if 'distance' not in self.qor_items:
            self.qor_items['distance'] = {}
        chiplet_name = file_path.split('/')[-1].split('.')[0]
        if chiplet_name not in self.qor_items['distance']:
            self.qor_items['distance'][chiplet_name] = {'QoR criteria':[]}

        with open(file_path, 'r') as input_file:
            lines = input_file.readlines()
            for line in lines:
                vec = line.split()
                if len(vec) >= 4:
                    raise ValueError(f"Check list format error!!! In {file_path}")
                if len(vec) == 0:
                    continue
                name_1 = vec[0]
                name_2 = vec[1]
                rule = vec[2]
                if name_1 in self.canvas.solution_obj[chiplet_name]:
                    conn_obj_1 = self.canvas.solution_obj[chiplet_name][name_1]
                elif name_1 in self.canvas.marked_blocks[chiplet_name]:
                    conn_obj_1 = self.canvas.marked_blocks[chiplet_name][name_1]
                else:
                    conn_obj_1 = name_1

                if name_2 in self.canvas.solution_obj[chiplet_name]:
                    conn_obj_2 = self.canvas.solution_obj[chiplet_name][name_2]
                elif name_2 in self.canvas.marked_blocks[chiplet_name]:
                    conn_obj_2 = self.canvas.marked_blocks[chiplet_name][name_2]
                else:
                    conn_obj_2 = name_2
                #print(getattr(conn_obj_1, 'Name', conn_obj_1), getattr(conn_obj_2, 'Name', conn_obj_2), name_1, name_2) 

                self.qor_items['distance'][chiplet_name]['QoR criteria'].append([conn_obj_1, conn_obj_2, rule, mode])

    def import_bump_current_list(self, file_path=None):
        pass

    def import_overlap_list(self, file_path=None):
        #file_path = filedialog.askopenfilename(filetypes=[("txt files", "*.txt")])
        if not file_path: return
        self.qor_items.setdefault('overlap', {})
        chiplet_name = file_path.split('/')[-1].split('.')[0]
        self.qor_items['overlap'].setdefault(chiplet_name, {'QoR criteria':{}})

        with open(file_path, 'r') as input_file:
            lines = input_file.readlines()
            for line in lines:
                line = line.strip()
                overlap = ''
                if 'N_OVERLAP' in line:
                    overlap = 'N_OVERLAP'
                    vec = line.split('N_OVERLAP')
                    new_vec = []
                    for v in vec:
                        if v.strip() != "":
                            new_vec.append(v)
                    vec = new_vec
                elif ' OVERLAP ' in line:
                    overlap = 'OVERLAP'
                    ele = line.split('OVERLAP')
                    ele[0] = ele[0].strip()
                    ele_1 = ele[0].split(',')
                    ele[1] = ele[1].strip()
                    ele_2 = ele[1]
                    vec = [ele_1, ele_2]
                else:
                    raise ValueError(f"[ERROR] No this OVERLAP or N_OVERLAP found in the overlap list: {line}")
                #print(vec)
                self.qor_items['overlap'][chiplet_name]['QoR criteria'].setdefault(overlap, []).append(vec)


    def distance_qor(self, mode='manhattan'):
        #print(self.qor_items['distance'].keys())
        for chiplet_name, qor_dict in self.qor_items['distance'].items():
            all_solutions = self.canvas.solution_classes[chiplet_name]
            violation_messages = ''

            offset = self.canvas.die_max_y[chiplet_name] * self.canvas.cell_size
            violation_markers = []
            for idx, vec in enumerate(self.qor_items['distance'][chiplet_name]['QoR criteria']):
                n1 = vec[0]
                n2 = vec[1]
                rule = vec[2]
                mode = vec[3]
                skip_flag = False

                if isinstance(n1, str):
                    if is_valid_location(n1):
                        cleaned_string = n1.strip('[]')
                        string_parts = cleaned_string.split(',')
                        n1 = (float(string_parts[0]), float(string_parts[1]))
                    else:
                        print(f"[Distance QoR Error]Not found {n1}")
                        violation_messages += f"Not found {n1}\n"
                        skip_flag = True

                if isinstance(n2, str):
                    if is_valid_location(n2):
                        cleaned_string = n2.strip('[]')
                        string_parts = cleaned_string.split(',')
                        n2 = (float(string_parts[0]), float(string_parts[1]))
                    else:
                        print(f"[Distance QoR Error]Not found {n2}")
                        violation_messages += f"Not found {n2}\n"
                        skip_flag = True

                if skip_flag:
                    continue

                n1_pts = get_conn_location(n1)
                n2_pts = get_conn_location(n2)
                min_dist, n1_pt, n2_pt = shortest_manhattan_distance(n1_pts, n2_pts)

                maze_start = (n1_pt[0] / self.canvas.cell_size, self.canvas.die_max_y[chiplet_name] - n1_pt[1] / self.canvas.cell_size)
                maze_start = tuple(map(int, maze_start))
                maze_goal = (n2_pt[0] / self.canvas.cell_size, self.canvas.die_max_y[chiplet_name] - n2_pt[1] / self.canvas.cell_size)
                maze_goal = tuple(map(int, maze_goal))
                if 'distance' not in self.qor_results:
                    self.qor_results['distance'] = {}
                if chiplet_name not in self.qor_results['distance']:
                    self.qor_results['distance'][chiplet_name] = {}
                    self.qor_results['distance'][chiplet_name]['routing'] = {}
                    self.qor_results['distance'][chiplet_name]['manhattan'] = {}

                if 'routing' not in self.qor_results['distance'][chiplet_name]:
                    self.qor_results['distance'][chiplet_name]['routing'] = {}
                if 'manhattan' not in self.qor_results['distance'][chiplet_name]:
                    self.qor_results['distance'][chiplet_name]['manhattan'] = {}

                if (maze_start, maze_goal) in self.qor_results['distance'][chiplet_name]['routing']:
                    path, total_distance = self.qor_results['distance'][chiplet_name]['routing'][(maze_start, maze_goal)][0], self.qor_results['distance'][chiplet_name]['routing'][(maze_start, maze_goal)][1]
                elif mode == 'routing':
                    path, total_distance = bfs(maze, maze_start, maze_goal)
                    total_distance *= self.canvas.cell_size
                    self.qor_results['distance'][chiplet_name]['routing'][(maze_start, maze_goal)] = [path, total_distance]
                else:
                    path, total_distance = [], 0

                color = 'blue'
                if mode == 'routing':
                    distance_used = total_distance
                    path_to_draw = path
                else:
                    distance_used = min_dist
                    path_to_draw = [(n1_pt[0] / self.canvas.cell_size, self.canvas.die_max_y[chiplet_name] - n1_pt[1] / self.canvas.cell_size), 
                                    (n2_pt[0] / self.canvas.cell_size, self.canvas.die_max_y[chiplet_name] - n2_pt[1] / self.canvas.cell_size)]
                #print(min_dist, total_distance, distance_used)
                if '<=' in rule:
                    std_dist = float(rule.split('<=')[-1])
                    if distance_used > std_dist:
                        color = 'red'
                elif '>=' in rule:
                    std_dist = float(rule.split('>=')[-1])
                    if distance_used < std_dist:
                        color = 'red'
                else:
                    std_dist = float(rule)
                    if distance_used != std_dist:
                        color = 'red'

                if color == 'red':
                    n1_vio = getattr(n1, 'Name', str(n1))
                    n2_vio = getattr(n2, 'Name', str(n2))
                    tag = f'qor-distance-{idx}'
                    violation_messages += f"{n1_vio} {n2_vio}, distance = {distance_used}, constraint is {rule}\n"
                    violation_markers.append([f"{n1_vio} {n2_vio}, distance = {distance_used}, constraint is {rule}", n1, n2, 'Distance - '+mode, rule, f'qor-distance-{idx}'])

            if violation_messages:
                output_file = f"{self.output_dir[chiplet_name]}/vio_distance_qor.rpt"
                with open(output_file, 'w') as f:
                    print(f"Distances violation dump in file {output_file}")
                    f.write(violation_messages)


    def get_core_region_points(self, chiplet_name, grid_size=10):
        points = []
        if chiplet_name not in self.canvas.core_region:
            log.logger.critical(f"Not found this name {chiplet_name}")
            return
        x_min, y_min = self.canvas.core_region[chiplet_name][0][0], self.canvas.core_region[chiplet_name][0][1]
        x_max, y_max = self.canvas.core_region[chiplet_name][2][0], self.canvas.core_region[chiplet_name][2][1]
        bkg_polys = []
        checkpoint1 = time.time()
        checkpoint2 = time.time()
        if chiplet_name in self.bkg_points:
            points = self.bkg_points[chiplet_name]['points']
        else:
            self.bkg_points.setdefault(chiplet_name, {})
            self.bkg_points[chiplet_name].setdefault('bkg', set())
            self.bkg_points[chiplet_name].setdefault('block', [])
            for bkg_poly in self.canvas.marked_blockage[chiplet_name]:
                bkg_poly = Polygon(bkg_poly)
                bkg_polys.append(bkg_poly)

            bkg_polys = unary_union(bkg_polys)
            checkpoint2 = time.time()
            print(f"[get_core_region] Time after blkg poly creation: {checkpoint2 - checkpoint1:.4f} seconds")

            block_regions = []
            for block_name, block in self.canvas.marked_blocks[chiplet_name].items():
                if block.CellName in self.canvas.ip_size:
                    block_poly = Polygon(block.coords)
                    bkg_polys = bkg_polys.union(block_poly)
                    continue
                block_poly = Polygon(block.coords)
                block_regions.append(block_poly)
                pts = mark_points_in_polygon(block_poly, grid_size)
                for pt in pts:
                    points.append({'region': block, 'coords': pt})

            block_regions = unary_union(block_regions)
            bkg_polys = bkg_polys.difference(block_regions)
            core_region = Polygon(self.canvas.core_region[chiplet_name])
            core_region = core_region.difference(block_regions).difference(bkg_polys)

            if core_region.geom_type == 'Polygon':
                core_region = [core_region]
            elif core_region.geom_type == 'MultiPolygon':
                core_region = list(core_region)

            #for poly in core_region:
            #    pts = mark_points_in_polygon(poly, grid_size)
            #    for pt in pts:
            #        points.append({'region': 'core', 'coords': pt})
            with ThreadPoolExecutor(max_workers=40) as executor:
                future_to_polygon = {executor.submit(mark_points_in_polygon, poly, grid_size): poly for poly in core_region}
                for future in as_completed(future_to_polygon):
                    poly_points = future.result()
                    for pt in poly_points:
                        points.append({'region': 'core', 'coords': pt})
            self.bkg_points[chiplet_name]['points'] = points

        checkpoint3 = time.time()
        print(f"[get_core_region] Time after points creation: {checkpoint3 - checkpoint2:.4f} seconds")

        return points


    def coverage_qor(self):
        if self.canvas.solution_classes is None:
            return

        qor_items = self.qor_items
        if 'coverage' not in qor_items or not qor_items['coverage']:
            return

        qor_results = self.qor_results.setdefault('coverage', {})

        for chiplet_name, qor_dict in qor_items['coverage'].items():
            all_solutions = self.canvas.solution_classes[chiplet_name]
            chip_qor_results = qor_results.setdefault(chiplet_name, {}).setdefault('QoR results', {})
            results = chip_qor_results.setdefault('results', {})

            # Power domain check
            pclamp_domain = {getattr(pclamp, 'TopPowerNets', []) for pclamp in all_solutions.get('Pclamp', [])}
            chip_domain = self.qor_results['coverage'][chiplet_name]['QoR results']['Chip domain']

            violation_messages = self._check_power_domain(chip_domain, pclamp_domain)

            for vec in qor_dict['QoR criteria']:
                item1, item2, item2_cell, power_ground, radius, purpose = vec
                radius = float(radius)
                key = (item1, item2, item2_cell, radius, purpose)

                if key in results:
                    continue

                results[key] = []
                group1_objs = all_solutions.get(item1, self.get_core_region_points(item1))
                group2_objs = all_solutions.get(item2, self.get_core_region_points(item2))

                if item2_cell:
                    group2_objs = [obj for obj in group2_objs if getattr(obj, 'CellName', '') == item2_cell]

                group1_coords, group1_obj_loc = self._get_coordinates_and_locations(group1_objs)
                group2_coords, group2_obj_loc = self._get_coordinates_and_locations(group2_objs)

                overlap_dict = self._check_individual_overlap(group1_coords, group2_coords, radius)

                attr = 'TopPowerNets' if power_ground == 'Power' else 'TopGroundNets'
                results[key] = self._process_results(group1_obj_loc, group2_coords, overlap_dict, attr, chip_domain, item1, purpose)

            self._dump_violations(chiplet_name, chip_qor_results, violation_messages)

    def _check_power_domain(self, chip_domain, pclamp_domain):
        violation_messages = ''
        missing_in_pclamp = chip_domain - pclamp_domain
        missing_in_block = pclamp_domain - chip_domain

        if missing_in_pclamp:
            violation_messages += f"Missing power domain in the pclamp: {missing_in_pclamp}\n"
        if missing_in_block:
            violation_messages += f"Missing power domain in the block: {missing_in_block}\n"

        return violation_messages

    def _get_coordinates_and_locations(self, objs):
        coords, obj_loc = {}, {}
        for obj in objs:
            _, locations, objs = get_location(obj, self.canvas.ip_size)
            for loc, obj in zip(locations, objs):
                coords[loc] = obj
                obj_loc.setdefault(obj, []).append(loc)
        return coords, obj_loc

    def _check_individual_overlap(self, group1_coords, group2_coords, radius):
        # Implement the logic for checking individual overlap
        pass

    def _process_results(self, group1_obj_loc, group2_coords, overlap_dict, attr, chip_domain, item1, purpose):
        results = []
        for obj, coords in group1_obj_loc.items():
            golden = getattr(obj, attr, ['VDD_SOC'] if attr == 'TopPowerNets' else ['VSS'])
            centroid = calculate_centroid(list(coords))
            for g in golden:
                if g not in chip_domain:
                    continue
                for coord in coords:
                    if not isinstance(coord, (list, tuple)) or len(coord) != 2 or not all(isinstance(c, (float, int)) for c in coord):
                        raise ValueError(f"coord format error {coord}")

                    passed = any(getattr(group2_coords[overlap_coord], attr, '') == g for overlap_coord in overlap_dict.get(coord, []))
                    if not passed:
                        results.append([coord, obj, g, purpose])
        return results

    def _dump_violations(self, chiplet_name, chip_qor_results, violation_messages):
        offset = self.canvas.die_max_y[chiplet_name] * self.canvas.cell_size
        violation_markers = []
        vio_cnt = 0

        for qor_criteria, qor_result in chip_qor_results['results'].items():
            item_1, item_2, item2_cell, radius = qor_criteria[:4]
            for r in qor_result:
                coord, obj, target_key, purpose = r
                violation_messages += f"{getattr(obj,'Name', obj)} {getattr(obj, 'CellName', '')} {coord} cannot find corresponding {item_2} {item2_cell} for {target_key} {purpose}\n"
                vio_cnt += 1
                violation_markers.append([violation_messages, obj, f"{target_key}, {item2_cell}", purpose, radius, f'qor-coverage-{vio_cnt}'])

        if violation_messages:
            output_file = f"{self.output_dir[chiplet_name]}/vio_coverage_qor.rpt"
            with open(output_file, 'w') as f:
                f.write(violation_messages)

    def bump_current_qor(self):
        def calculate_total_current(block):

            pd_dict = {
                'TSV_ARRAY_FP_BP': {'default': (1, 0.825)},
                'DDR*': {'default': (3, 0.825)},
                'SDR*': {'default': (3, 0.825)},
                'FOM_ARRAY_*': {'default': (13.9, 1.15)},
                'FCCC_ARRAY*': {'default': (4, 0.825)},
                'SRAM_ARRAY*': {'VDD_RAM': (0.35, 0.825), 'VDDM_RAM': (1, 0.825), 'default': (1.35, 0.825)},
                'RT_RING_FP_BP': {'default': (1.5, 0.825)},
            }

            current_dict = {}
            for key, nets in pd_dict.items():
                regex_pattern = '^' + re.escape(key[:-1]) + '.*' if key.endswith('*') else '^' + re.escape(key) + '$'
                if re.match(regex_pattern, block.CellName):
                    for net, (density, voltage) in nets.items():
                        current_dict[net] = (density * block.polygon.area) / (voltage * 1000000)
                    return current_dict
            log.logger.info(f"Block {block.Name} {block.CellName} has no current info.")
            return {}

        def classify_by_attr(all_bumps, all_blocks):
            result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            rtree_idx = index.Index()

            for idx, block in enumerate(all_blocks):
                rtree_idx.insert(idx, block.polygon.bounds)

            for bump in all_bumps:
                cell = bump['CellName']
                pname = bump['PinName']
                loc = bump['Location']
                candidate_indices = list(rtree_idx.intersection((loc[0], loc[1], loc[0], loc[1])))
                if len(candidate_indices) > 1:
                    log.logger.info(f"Error in rtree search")
                    exit(1)
                elif len(candidate_indices) == 1:
                    bump['block'] = all_blocks[candidate_indices[0]].Name
                else:
                    bump['block'] = 'NA'
                bump['DrawVoronoiPoly'] = None
                bump['CurrentVoronoiPoly'] = None
                bump['Area'] = 0
                bump['Current'] = None
                result[cell][pname][bump['block']].append(bump)
            return result

        def get_bump_by_cell(cell, all_bumps_by_attr):
            bumps = []
            result  = []
            for net_key in all_bumps_by_attr[cell].keys():
                for block_key in all_bumps_by_attr[cell][net_key].keys():
                    bumps.extend(all_bumps_by_attr[cell][net_key][block_key])
            for bump in bumps:
                # skip bumps in SOC domain, floating bumps, etc
                if bump['Area'] == 0: continue
                result.append(bump)
            return result

        def calculate_bump_currents(total_current, all_bumps_by_attr, bump_cell, net, block, net_num):
            selected_bumps = all_bumps_by_attr[bump_cell][net][block.Name]
            if (len(selected_bumps) == 0): return []

            log.logger.info(f"Analyzing {name} {net} {bump_cell}({len(selected_bumps)})")
            points = [Point(loc["Location"][0], loc["Location"][1]) for loc in selected_bumps]
            multipoint = MultiPoint(points)
            voronoi = voronoi_diagram(multipoint, envelope=block.polygon)
            voronoi_polys = list(shapely.intersection(voronoi.geoms, block.polygon))

            for bump in selected_bumps:
                p = Point(bump["Location"][0], bump["Location"][1])
                for poly in voronoi_polys:
                    if poly.contains(p):
                        bump['CurrentVoronoiPoly'] = poly.exterior.coords
                        bump['DrawVoronoiPoly'] = poly.exterior.coords
                        bump['Area'] = poly.area
                        if net_num == 1:
                            bump['Current'] = total_current['default'] * (poly.area/block.polygon.area)
                        else:
                            bump['Current'] = total_current[net] * (poly.area/block.polygon.area)
            return selected_bumps

        all_bumps = [vars(obj) for obj in self.canvas.solution_classes[self.canvas.die_name]['Bump']]
        all_blocks = list(self.canvas.marked_blocks[self.canvas.die_name].values())
        all_bump_cells = list({bump['CellName'] for bump in all_bumps})
        #all_bump_cells = ['C4_POWER', 'C4_GROUND', 'GRTSVNDRP_1x1', 'GRTSVNDRG_1x1', 'SAC_PAD_POWER', 'SAC_PAD_GROUND']
        #all_bump_cells = ['C4_POWER', 'C4_GROUND']
        #all_bump_cells = ['GRTSVNDRP_1x1']
        all_bumps_by_attr = classify_by_attr(all_bumps, all_blocks)

        for name, block in self.canvas.marked_blocks[self.canvas.die_name].items():
            total_current = calculate_total_current(block)
            if not total_current: continue
            #block.TopPowerNets = ['-']
            #block.TopGroundNets = ['-']
            if len(block.TopGroundNets) != 1:
                log.logger.info(f"<Skip TopGroundNets != 1> {block.Name} {block.CellName} {total_current} {block.TopPowerNets} {block.TopGroundNets} ================ ")
                continue
            else:
                log.logger.info(f"{block.Name} {block.CellName} {total_current} {block.TopPowerNets} {block.TopGroundNets} ================ ")

            for bump_cell in all_bump_cells:
                all_power_bumps = []
                for net in block.TopPowerNets:
                    returned_bumps = calculate_bump_currents(total_current, all_bumps_by_attr, bump_cell, net, block, len(block.TopPowerNets))
                    all_power_bumps.extend(returned_bumps)

                if len(block.TopPowerNets) != 1:
                    # update DrawVoronoiPoly
                    points = [Point(loc["Location"][0], loc["Location"][1]) for loc in all_power_bumps]
                    multipoint = MultiPoint(points)
                    voronoi = voronoi_diagram(multipoint, envelope=block.polygon)
                    voronoi_polys = list(shapely.intersection(voronoi.geoms, block.polygon))

                    for bump in all_power_bumps:
                        p = Point(bump["Location"][0], bump["Location"][1])
                        for poly in voronoi_polys:
                            if poly.contains(p):
                                bump['DrawVoronoiPoly'] = poly.exterior.coords

                for net in block.TopGroundNets:
                    calculate_bump_currents(total_current, all_bumps_by_attr, bump_cell, net, block, len(block.TopGroundNets))

        #min_max_dict = defaultdict(lambda: defaultdict(float))
        #bump_groups = [['GRTSVNDRP', 'GRTSVNDRG'], ['UBMB_UP18P', 'UBMB_UP18G'], ['C4_POWER', 'C4_GROUND']]
        #for bump_group in bump_groups:
        #    combined_bumps = []
        #    for cell in bump_group:
        #        combined_bumps.extend(get_bump_by_cell(cell, all_bumps_by_attr))
        #    if len(combined_bumps) == 0: continue
        #    #areas = [bump['Area'] for bump in combined_bumps]
        #    #min_area = min(areas)
        #    #max_area = max(areas)
        #    currents = [bump['Current'] for bump in combined_bumps]
        #    min_current = min(currents)
        #    max_current = max(currents)
        #    for cell in bump_group:
        #        min_max_dict[cell]['min'] = min_area
        #        min_max_dict[cell]['max'] = max_area
        #log.logger.info(f'{min_max_dict}')

        bump_max_current = {
            'C4_GROUND': 0.12, \
            'C4_SIGNAL': 0.12, \
            'C4_POWER': 0.12, \
            'UBMB_UP18G': 0.12, \
            'UBMB_UP18P': 0.12, \
            'UBMB_UP18D': 0.12, \
            'UBMB_UP18S': 0.12, \
            'GRTSVNDRP': 0.12, \
            'GRTSVNDRG': 0.12, \
            'GRTSVNDRS': 0.12, \
            'GRTSVNDRD': 0.12 \
        }
        bump_max_current = {
            'GRTSVNDRP_1x1' : 0.08,
            'GRTSVNDRG_1x1' : 0.08,
            'C4_POWER' : 0.35,
            'C4_GROUND' : 0.35,
            'HBP6P' : 0.05,
            'HBP6G' : 0.05,
            'SAC_PAD_POWER' : 0.3,
            'SAC_PAD_GROUND' : 0.3
        }

        for cell in all_bump_cells:
            bumps = get_bump_by_cell(cell, all_bumps_by_attr)

            if len(bumps) == 0: continue

            currents = [bump['Current'] for bump in bumps]
            min_current = min(currents)
            max_current = max(currents)
            max_current = bump_max_current[cell]
            #min_area = min_max_dict[cell]['min']
            #max_area = min_max_dict[cell]['max']

            log.logger.info(f'Drawing result for: {cell} with {len(bumps)} polys')
            #areas = [bump['Area'] for bump in bumps]
            norm_currents = [(current - min_current) / (max_current - min_current) for current in currents]
            loc_x, loc_y = zip(*[(bump['Location'][0], bump['Location'][1]) for bump in bumps])
            polys = [Polygon(bump['DrawVoronoiPoly']) for bump in bumps]

            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.withdraw()
            fig_width = screen_width / 200
            fig_height = screen_height / 100
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            colormap = cm.get_cmap('coolwarm')
            for poly, norm_current, x_loc, y_loc in zip(polys, norm_currents, loc_x, loc_y):
                x, y = poly.exterior.xy
                color = colormap(norm_current)
                ax.fill(x, y, color=color, alpha=0.5)
                #ax.scatter(x_loc, y_loc, color='blue', s=2, label="Points")

            norm = Normalize(vmin=min_current, vmax=max_current)
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=min_current, vmax=max_current))
            sm.set_array([])  # Only needed for compatibility with matplotlib < 3.1
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Current (A)')

            data = self.canvas.data['S1_output'][self.canvas.die_name]
            chip_poly = PlotPolygon(data['TOPShape']['DieCoords'], closed=True, edgecolor='black', facecolor='none')
            ax.add_patch(chip_poly)

            for block_info in data['BlockShapes']:
                block_poly = PlotPolygon(block_info['Coords'], closed=True, edgecolor='gray', facecolor='none')
                ax.add_patch(block_poly)
                shapely_poly = Polygon(block_info['Coords'])
                centroid = shapely_poly.centroid
                name = block_info['DesignName'].split('_')[0]
                if len(name) < 5:
                    ax.text(centroid.x, centroid.y, block_info['DesignName'].split('_')[0], ha='center', va='center', fontsize=8)

            fig.suptitle(f"Bump current estimation result for {cell}")
            ax.set_aspect('equal')
            output_file = f"{self.app.output_dir}/bump_current_est_{cell}.png"
            plt.savefig(output_file)
            #plt.show()
        log.logger.info(f"End of bump current estimation")

    def overlap_qor(self):
        if self.canvas.solution_classes is None:
            return

        qor_items = self.qor_items
        if 'overlap' not in qor_items or not qor_items['overlap']:
            return

        qor_results = self.qor_results.setdefault('overlap', {})


        for chiplet_name, qor_dict in qor_items['overlap'].items():
            output_file = f"{self.output_dir[chiplet_name]}/vio_overlap_qor.rpt"
            with open(output_file, 'w') as f:
                pass
            idx = 0
            offset = self.canvas.die_max_y[chiplet_name] * self.canvas.cell_size
            for overlap_mode, vec in qor_dict['QoR criteria'].items():
                for v in vec:
                    result, overlap_item_list = self.check_overlap(overlap_mode, v, chiplet_name, output_file)
                    if result:
                        qor_results.setdefault(chiplet_name, {}).setdefault('QoR results', {}).setdefault('results', {}).setdefault(overlap_mode, {}).setdefault(str(idx), result)
                        qor_results[chiplet_name].setdefault('Legend', set()).add(overlap_mode)
                        idx += 1
                        with open(output_file, 'a') as f:
                            f.write(f"{overlap_mode}============\n")
                            for item in overlap_item_list:
                                if isinstance(item, str):
                                    name = item
                                else:
                                    name = getattr(item, Name, f'{item}')
                                f.write(f"{name}\n")
                            f.write('==============\n')
                            print(f"[Overlap QoR] Please find overlap violations in {output_file}")
        #print(qor_results)



    def check_overlap(self, overlap_mode, vec, chiplet_name, output_file):
        if overlap_mode == 'OVERLAP':
            vec_2 = vec[1]
            vec_1 = vec[0]
        else:
            vec_1 = vec
        polygon_coords = []
        shape_obj = {}
        for item in vec_1:
            skip_invalid_poly = False
            coord = []
            item = item.strip()
            #print(f"Item: {item}")
            if isinstance(item, str) and item.startswith('[') and item.endswith(']'):
                coordinates_str = item[1:-1]
                coordinates = [float(x) for x in coordinates_str.split(',')]
                coord = coordinates
            elif isinstance(item, str):
                if item == "CORE" or item == "CORE_W_PBKG":
                    coord = self.canvas.core_region[chiplet_name]
                else:
                    #print(chiplet_name, item)
                    #print(self.canvas.solution_obj[chiplet_name].keys())
                    if item in self.canvas.solution_obj[chiplet_name]:
                        obj = self.canvas.solution_obj[chiplet_name][item]
                    else:
                        messagebox.showerror("Error", f"No this {item} found in the chiplet {chiplet_name} item")
                        raise ValueError(f"No this {item} found in the chiplet {chiplet_name} item")
                    coord = get_coord(obj, self.canvas.ip_size)
                    poly = Polygon(coord)
                    if poly.is_empty or not isinstance(poly, BaseGeometry):
                        with open(output_file, 'a') as f:
                            f.write(f"[Error] Cannot find shape for this object {obj}, found the coord {coord}\n")
                            print(f"[Error] Cannot find shape for this object {obj}, found the coord {coord}")
                            skip_invalid_poly = True
                    else:
                        shape_obj[poly] = item
            if skip_invalid_poly == False:
                polygon_coords.append(coord)
        polygons = [Polygon(coords) for coords in polygon_coords]
        if overlap_mode == 'N_OVERLAP':
            overlap_areas = find_overlaps_with_rtree(polygons)

            if overlap_areas:
                overlap_poly = overlap_areas[0]
                for overlap in overlap_areas:
                    overlap_poly = overlap_poly.union(overlap)
                    overlap_items = set()
                    if isinstance(overlap_poly, BaseGeometry):
                        for poly, item in shape_obj.items():
                            if poly.intersection(overlap_poly):
                                overlap_items.add(item)
                    else:
                        raise TypeError(f"Invalid type in N_OVERLAP check {overlap_poly}")
                return overlap_areas, overlap_items
        else:
            item = vec_2.strip()
            if isinstance(item, str) and item.startswith('[') and item.endswith(']'):
                coordinates_str = item[1:-1]
                coordinates = [float(x) for x in coordinates_str.split(',')]
                container_coords = coordinates
            elif isinstance(item, str):
                if item == "CORE":
                    container_coords = self.canvas.core_region[chiplet_name]
                    blockage = self.canvas.marked_blockage[chiplet_name]
                    container_polygon = Polygon(container_coords)
                    for blockage_coords in blockage:
                        blockage_polygon = Polygon(blockage_coords)
                        container_polygon = container_polygon.difference(blockage_polygon)
                elif item == "CORE_W_PBKG":
                    container_coords = self.canvas.core_region[chiplet_name]
                    container_polygon = Polygon(container_coords)
                    #poly_test = []
                    for name, block in self.canvas.marked_blocks[chiplet_name].items():
                        block_polygon = Polygon(block.coords)
                        container_polygon = container_polygon.difference(block_polygon)
                        #poly_test.append(block_polygon)
                else:
                    obj = self.canvas.solution_obj[chiplet_name][item]
                    container_coords = get_coord(obj, self.canvas.ip_size)
                    container_polygon = Polygon(container_coords)
            #print(item, 'container', container_coords)
            polygons = [Polygon(coords) for coords in polygon_coords]
            if not isinstance(container_polygon, BaseGeometry) or container_polygon.is_empty:
                with open(output_file, 'a') as f:
                    f.write(f"[Error] Cannot find shape for this object {item}, {container_polygon}\n")
                    print(f"[Error] Cannot find shape for this object {item}, {container_polygon}")
                return [], []
            outside_areas = find_outside_areas(polygons, container_polygon)
            #return poly_test
            if outside_areas:
                print("not in specified region")
                #for outside in outside_areas:
                #    print(outside)
                return outside_areas, []
        return [], []


    def on_line_double_click(self, tag):
        if self.double_click_processing:
            return
        self.double_click_processing = True
        if self.violation_tree is not None:
            if self.violation_window:
                self.violation_window.deiconify()
                self.violation_window.lift()
            for item in self.violation_tree.get_children():
                item_tags = self.violation_tree.item(item, "tags")
                if tag in item_tags:
                    self.violation_tree.selection_set(item)
                    self.violation_tree.see(item)
                    break
        self.double_click_processing = False

    def update_canvas_overlap_qor(self):
        if self.canvas.solution_classes is None or self.canvas.die_name is None:
            return

        if 'overlap' not in self.qor_items:
            return
        if len(self.qor_items['overlap']) == 0:
            return

        if 'overlap' in self.qor_tags:
            for tag in self.qor_tags['overlap']:
                self.canvas.delete(tag)
        self.qor_tags['overlap'] = []

        if not self.checkbuttons['Show overlap check'].get():
            return

        if 'overlap' not in self.qor_results:
            return
        legend = self.qor_results['overlap'][self.canvas.die_name]['Legend']
        selected_legends = set()
        #print(legend)
        for l in legend:
            if self.checkbuttons[l].get():
                selected_legends.add(l)
        if not selected_legends:
            self.update_violation_tree([])
            return

        offset = self.canvas.die_max_y[self.canvas.die_name] * self.canvas.cell_size
        violation_markers = []
        tag = 0
        for qor_criteria, qor_result in self.qor_results['overlap'][self.canvas.die_name]['QoR results']['results'].items():
            if (qor_criteria) not in selected_legends:
                    continue

            for idx, result_list in qor_result.items():
                tag = f'qor-overlap-{idx}'
                for shapely_polygon in result_list:
                    coords = list(shapely_polygon.exterior.coords)
                    transformed_coords = []
                    for x, y in coords:
                        transformed_x = x * self.canvas.scale_factor
                        transformed_y = (offset - y) * self.canvas.scale_factor
                        transformed_coords.extend([transformed_x, transformed_y])
                    poly_tag = self.canvas.create_polygon(transformed_coords, fill='red', outline='black', tags=tag, width=2)

                    self.canvas.tag_bind(poly_tag, '<Enter>', lambda event, t=tag: self.highlight_selected_violation(t))
                    self.canvas.tag_bind(poly_tag, '<Leave>', lambda event, t=tag: self.clear_highlight(t))
                    self.canvas.tag_bind(poly_tag, '<Double-Button-1>', lambda event, t=tag: self.on_line_double_click(t))
                self.qor_tags['overlap'].append(tag)
                violation_markers.append([f"Violated {qor_criteria} region\n", '', f'Violated {idx}', qor_criteria, qor_criteria, tag])

        self.open_violation_browser(violation_markers)


    def highlight_selected_violation(self, tag):
        self.canvas.itemconfig(tag, width=4)

    def clear_highlight(self, tag):
        self.canvas.itemconfig(tag, width=2)

    def update_violation_tree(self, violations):
        if self.violation_tree is None:
            return
        current_items = {self.violation_tree.item(item, 'tags')[0]: item for item in self.violation_tree.get_children()}
        all_existed_tags = set()
        #print(self.qor_tags)
        for key, tag_list in self.qor_tags.items():
            for tag in tag_list:
                all_existed_tags.add(tag)
        #print('Current:', current_items)
        #print('Existed:', all_existed_tags)
        # Clear existing items in the tree
        #print('Violation tree:',self.violation_tree.keys())
        for item in set(current_items.keys()) - all_existed_tags:
            self.violation_tree.delete(current_items[item])

        # Insert new items
        for violation in violations:
            violation_message, n1, n2, marker_type, rule, tag = violation
            if tag not in current_items:
                self.violation_tree.insert('', tk.END, values=(marker_type, getattr(n1, 'Name', str(n1)), getattr(n2, 'Name', str(n2)), rule, violation_message), tags=(tag,))

    def open_violation_browser(self, violations):
        if self.violation_window is not None and tk.Toplevel.winfo_exists(self.violation_window):
            self.update_violation_tree(violations)
            return
        self.violation_window = Toplevel(self.canvas.master)
        self.violation_window.title("Violation Browser")
        self.violation_window.geometry("800x400")

        frame = tk.Frame(self.violation_window)
        frame.pack(fill=tk.BOTH, expand=True)

        self.violation_tree = Treeview(frame, columns=("Vio category", "Item1", "Item2", "Description"), show='headings')
        self.violation_tree.heading("Vio category", text="Vio category")
        self.violation_tree.heading("Item1", text="Item1")
        self.violation_tree.heading("Item2", text="Item2")
        self.violation_tree.heading("Description", text="Description")

        v_scrollbar = Scrollbar(frame, orient=tk.VERTICAL, command=self.violation_tree.yview)
        h_scrollbar = Scrollbar(frame, orient=tk.HORIZONTAL, command=self.violation_tree.xview)

        self.violation_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        self.violation_tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        for violation in violations:
            violation_message, n1, n2, marker_type, rule, tag = violation
            self.violation_tree.insert('', tk.END, values=(marker_type, getattr(n1, 'Name', str(n1)), getattr(n2, 'Name', str(n2)), rule, violation_message), tags=(tag,))

        def on_select(event):
            if self.double_click_processing:
                return
            selected = self.violation_tree.selection()
            if self.prev_selected == selected:
                return
            self.double_click_processing = True

            if self.prev_selected:
                try:
                    prev_tag = self.violation_tree.item(self.prev_selected[0])['tags'][0]
                    self.clear_highlight(prev_tag)
                except TclError:
                    pass

            if selected:
                self.prev_selected = selected
                item = self.violation_tree.item(selected[0])
                tag = item['tags'][0]

                self.highlight_selected_violation(tag)
                self.update_detail_text(selected[0])
            self.double_click_processing = False

        self.violation_tree.bind('<<TreeviewSelect>>', on_select)
        self.violation_tree.bind_all('<MouseWheel>', self._on_mouse_wheel_treeview)
        # Add detail text box
        self.detail_text = Text(self.violation_window, height=10, state=tk.DISABLED)
        self.detail_text.pack(fill=tk.BOTH, expand=True)


    def update_detail_text(self, item):
        details = self.violation_tree.item(item, "values")
        detail_message = details[-1]
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)
        self.detail_text.insert(tk.END, detail_message)
        self.detail_text.config(state=tk.DISABLED)

    def on_double_click(self, event, item, title, rect_id):
        self.canvas.itemconfig(rect_id, width=3)
        detail_window = tk.Toplevel(self.winfo_toplevel())
        detail_window.title(f"{title}")
        # marked the object as selected
        self.selected_items[rect_id] = detail_window
        label = tk.Label(detail_window, text=f"{item}", pady=10)
        label.pack(padx=20, pady=20)
        def on_close():
            self.canvas.itemconfig(rect_id, width=1)
            del self.selected_items[rect_id]
            detail_window.destroy()

        detail_window.protocol("WM_DELETE_WINDOW", on_close)

    def _on_mouse_wheel_treeview(self, event):
        if event.widget == self.violation_tree:
            self.violation_tree.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

    def update_canvas_coverage_qor(self):
        if self.canvas.solution_classes is None or self.canvas.die_name is None:
            return

        if 'coverage' not in self.qor_items:
            return
        if len(self.qor_items['coverage']) == 0:
            return

        if 'coverage' in self.qor_tags:
            for tag in self.qor_tags['coverage']:
                self.canvas.delete(tag)
        self.qor_tags['coverage'] = []

        if not self.checkbuttons['Show coverage check'].get():
            return
        show_coverage = False
        #print(self.checkbuttons.keys())
        if 'Show covered region' not in self.checkbuttons:
            return
        if self.checkbuttons['Show covered region'].get():
            show_coverage = True

        legend = self.qor_results['coverage'][self.canvas.die_name]['Legend']
        selected_legends = set()

        for l in legend:
            if self.checkbuttons[l].get():
                selected_legends.add(l)
        if not selected_legends:
            self.update_violation_tree([])
            return
        offset = self.canvas.die_max_y[self.canvas.die_name] * self.canvas.cell_size
        violation_markers = []
        tag = 0
        # results dict contains: 
        # results[key] = [item_1, item_2, radius, purpose, violation_dict]
        vio_cnt = 0
        for qor_criteria, qor_result in self.qor_results['coverage'][self.canvas.die_name]['QoR results']['results'].items():
            item_1, item_2, item2_cell, radius, purpose = qor_criteria[0], qor_criteria[1], qor_criteria[2], qor_criteria[3], qor_criteria[4]
            #print(item_1, item_2, item2_cell, radius)
            real_r = radius * self.canvas.scale_factor
            for pg_net, vio_list in qor_result.items():
                tag = f'qor-coverage-{vio_cnt}'
                if ('coverage', purpose) not in selected_legends:
                    continue
                #print("draw coord:", coord, getattr(obj, 'Location'))
                #print(getattr(obj,'Name', obj), getattr(obj, 'CellName', ''), coord, f'cannot find corresponding {item2} for', golden, purpose)
                #print(coord, r)
                color = 'red'

                if isinstance(vio_list, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
                    vio_list = [vio_list]
                for vio_item in vio_list:
                    if isinstance(vio_item, shapely.geometry.Point):
                        x = vio_item.x * self.canvas.scale_factor
                        y = (offset - vio_item.y) * self.canvas.scale_factor

                        oval_tag = self.canvas.create_oval(x-2, y-2, x+2, y+2, outline=color, tags=tag, width=2)

                        if show_coverage:
                            cover_tag = self.canvas.create_oval(x - real_r, y - real_r, x + real_r, y + real_r, outline=color, tags=tag, width=2)
                            self.canvas.tag_bind(cover_tag, '<Enter>', lambda event, t=tag: self.highlight_selected_violation(t))
                            self.canvas.tag_bind(cover_tag, '<Leave>', lambda event, t=tag: self.clear_highlight(t))
                            self.canvas.tag_bind(cover_tag, '<Double-Button-1>', lambda event, t=tag: self.on_line_double_click(t))
                        else:
                            self.canvas.tag_bind(oval_tag, '<Enter>', lambda event, t=tag: self.highlight_selected_violation(t))
                            self.canvas.tag_bind(oval_tag, '<Leave>', lambda event, t=tag: self.clear_highlight(t))
                            self.canvas.tag_bind(oval_tag, '<Double-Button-1>', lambda event, t=tag: self.on_line_double_click(t))

                    elif isinstance(vio_item, shapely.geometry.Polygon):
                        polygons = [vio_item]
                    elif isinstance(vio_item, shapely.geometry.MultiPolygon):
                        polygons = vio_item.geoms

                    for polygon in polygons:
                        coords = [(coord[0] * self.canvas.scale_factor, (offset - coord[1]) * self.canvas.scale_factor) for coord in polygon.exterior.coords]
                        flattened_coords = [val for coord_pair in coords for val in coord_pair]
                        poly_tag = self.canvas.create_polygon(flattened_coords, outline=color, fill='red', tags=tag, width=2)

                        if show_coverage:
                            buffered_polygon = polygon.buffer(real_r)
                            buffered_coords = [(coord[0] * self.canvas.scale_factor, (offset - coord[1]) * self.canvas.scale_factor) for coord in buffered_polygon.exterior.coords]
                            flattened_buffered_coords = [val for coord_pair in buffered_coords for val in coord_pair]
                            cover_tag = self.canvas.create_polygon(flattened_buffered_coords, outline=color, fill='black', tags=tag, width=2)
                            self.canvas.tag_bind(cover_tag, '<Enter>', lambda event, t=tag: self.highlight_selected_violation(t))
                            self.canvas.tag_bind(cover_tag, '<Leave>', lambda event, t=tag: self.clear_highlight(t))
                            self.canvas.tag_bind(cover_tag, '<Double-Button-1>', lambda event, t=tag: self.on_line_double_click(t))
                        else:
                            self.canvas.tag_bind(poly_tag, '<Enter>', lambda event, t=tag: self.highlight_selected_violation(t))
                            self.canvas.tag_bind(poly_tag, '<Leave>', lambda event, t=tag: self.clear_highlight(t))
                            self.canvas.tag_bind(poly_tag, '<Double-Button-1>', lambda event, t=tag: self.on_line_double_click(t))

                    for polygon in polygons:
                        if not polygon.interiors:
                            continue
                        coords = []
                        for ring in polygon.interiors:
                            coords = [(coord[0] * self.canvas.scale_factor, (offset - coord[1]) * self.canvas.scale_factor) for coord in ring.coords]
                            flattened_coords = [val for coord_pair in coords for val in coord_pair]
                            poly_tag = self.canvas.create_polygon(flattened_coords, outline=color, fill='white', tags=tag, width=2)

                    vio_cnt += 1
                    self.qor_tags['coverage'].append(tag)
                    violation_markers.append([f"{item_1} {item_2} {radius} cannot find corresponding {item_2} for {purpose}\n", item_1, pg_net, purpose, radius, f'qor-coverage-{vio_cnt}'])


        self.open_violation_browser(violation_markers)




    def update_canvas_distance_qor(self, mode='manhattan'):

        if self.canvas.solution_classes is None or self.canvas.die_name is None:
            return

        if 'distance' in self.qor_tags:
            for tag in self.qor_tags['distance']:
                self.canvas.delete(tag)

        if 'distance' not in self.qor_items:
            return

        if len(self.qor_items['distance']) == 0:
            return

        self.qor_tags['distance'] = []

        legend = self.qor_results.get('distance', {}).get(self.canvas.die_name, {}).get('Legend', None)

        if legend is None:
            return
        #print(legend, self.checkbuttons.keys())
        selected_legends = set()


        all_solutions = self.canvas.solution_classes[self.canvas.die_name]
        violation_messages = ''

        # Debugging output
        #print("Existing items with tag 'qor-distance-':", self.find_withtag(f'qor-distance-'))

        if not self.checkbuttons['Show distance check'].get():
            return

        for l in legend:
            if l not in self.checkbuttons:
                continue
            if self.checkbuttons[l].get():
                selected_legends.add(l)

        if not selected_legends:
            return
        offset = self.canvas.die_max_y[self.canvas.die_name] * self.canvas.cell_size
        violation_markers = []
        for qor_mode, qor_result in self.qor_results['distance'][self.canvas.die_name]['QoR results']['results'].items():
            if qor_mode == 'routing' and qor_mode in selected_legends:
                for r in qor_result:
                    n1, n2, rule, path_to_draw, distance_used, color = r[:6]
                    if color == 'red':
                        vio_cnt = r[6]
                        tag = f'qor-distance-{vio_cnt}'
                    else:
                        tag = f'qor-distance'

                    if path_to_draw:
                        coords = []
                        for cur_pt, next_pt in zip(path_to_draw, path_to_draw[1:]):
                            x1, y1 = cur_pt[0] * self.canvas.cell_size * self.canvas.scale_factor, cur_pt[1] * self.canvas.cell_size * self.canvas.scale_factor
                            x2, y2 = next_pt[0] * self.canvas.cell_size * self.canvas.scale_factor, next_pt[1] * self.canvas.cell_size * self.canvas.scale_factor
                            coords.extend([x1, y1, x2, y2])
                        self.qor_tags['distance'].append(tag)

                        line_tag = self.canvas.create_line(coords, tags=tag, fill=color, width=2)
                        if color == 'red':
                            self.canvas.tag_bind(line_tag, '<Enter>', lambda event, t=tag: self.highlight_selected_violation(t))
                            self.canvas.tag_bind(line_tag, '<Leave>', lambda event, t=tag: self.clear_highlight(t))
                            self.canvas.tag_bind(line_tag, '<Double-Button-1>', lambda event, t=tag: self.on_line_double_click(t))
                            n1_vio = getattr(n1, 'Name', str(n1))
                            n2_vio = getattr(n2, 'Name', str(n2))
                            violation_markers.append([f"{n1_vio} {n2_vio}, distance = {distance_used}, constraint is {rule}", n1, n2, 'Distance - '+qor_mode, rule, tag])

            elif qor_mode == 'manhattan' and qor_mode in selected_legends:
                for r in qor_result:
                    n1, n2, rule, path_to_draw, distance_used, color, n1_pt, n2_pt = r[:8]
                    if color == 'red':
                        vio_cnt = r[8]
                        tag = f'qor-distance-{vio_cnt}'
                    else:
                        tag = f'qor-distance'
                    self.qor_tags['distance'].append(tag)
                    txt = ''
                    if color == 'red':
                        txt = int(distance_used)
                    line_tag = self.draw_line(n1_pt, n2_pt, offset=offset, text=txt, color=color, tags=tag, dash=(5, 2))
                    if color == 'red':
                        self.canvas.tag_bind(line_tag, '<Enter>', lambda event, t=tag: self.highlight_selected_violation(t))
                        self.canvas.tag_bind(line_tag, '<Leave>', lambda event, t=tag: self.clear_highlight(t))
                        self.canvas.tag_bind(line_tag, '<Double-Button-1>', lambda event, t=tag: self.on_line_double_click(t))
                        n1_vio = getattr(n1, 'Name', str(n1))
                        n2_vio = getattr(n2, 'Name', str(n2))
                        violation_markers.append([f"{n1_vio} {n2_vio}, distance = {distance_used}, constraint is {rule}", n1, n2, 'Distance - '+qor_mode, rule, tag])



        self.open_violation_browser(violation_markers)


    def delete_violation_window(self):
        if self.violation_window is not None and self.violation_window.winfo_exists():
            self.violation_window.destroy()
            self.violation_window = None
            print("Violation window destroyed.")


    def draw_line(self, point1, point2, offset=-1, text='', color='', tags='', dash=''):
        if offset == -1:
            offset = self.canvas.die_max_y[self.canvas.die_name]
        line_id = self.canvas.create_line(point1[0]*self.canvas.scale_factor, (offset-point1[1])*self.canvas.scale_factor, point2[0]*self.canvas.scale_factor, (offset-point2[1])*self.canvas.scale_factor, fill=color, width=2, tags=tags, dash=dash)
        if text:
            mid_x = (point1[0] + point2[0]) / 2 * self.canvas.scale_factor
            mid_y = (offset - (point1[1] + point2[1]) / 2) * self.canvas.scale_factor
            self.canvas.create_text(mid_x, mid_y, text=text, fill=color, tags=tags)
        return line_id
        #print(f"Line ID: {line_id}, Tags: {self.gettags(line_id)}")    

    def coverage_qor(self):
        start_time = time.time()  # Start timing the function
        if self.canvas.solution_classes is None:
            return

        qor_items = self.qor_items
        if 'coverage' not in qor_items or not qor_items['coverage']:
            return

        qor_results = self.qor_results.setdefault('coverage', {})
        checkpoint1 = time.time()
        print(f"Time after setup: {checkpoint1 - start_time:.4f} seconds")

        for chiplet_name, qor_dict in qor_items['coverage'].items():
            #print('Chiplet name', chiplet_name, qor_dict)
            all_solutions = self.canvas.solution_classes[chiplet_name]
            chip_qor_results = qor_results.setdefault(chiplet_name, {}).setdefault('QoR results', {})
            legend = qor_results[chiplet_name].setdefault('Legend', {'Show covered region'})
            results = chip_qor_results.setdefault('results', {})

            # Power domain check
            pclamp_domain = set()
            for pclamp in all_solutions.get('Pclamp', []):
                VDD = getattr(pclamp, 'TopPowerNets', [])
                for vdd in VDD:
                    pclamp_domain.add(vdd)
                VSS = getattr(pclamp, 'TopGroundNets', [])
                for vss in VSS:
                    pclamp_domain.add(vss)
            chip_domain = self.qor_results['coverage'][chiplet_name]['QoR results']['Chip domain']

            violation_messages = self._check_power_domain(chiplet_name, chip_domain, pclamp_domain)
            checkpoint2 = time.time()
            #print(f"Time after power domain check: {checkpoint2 - checkpoint1:.4f} seconds")
            checkpoint3 = checkpoint2
            for vec in qor_dict['QoR criteria']:
                item1, item2, item2_cell, power_ground, radius, purpose = vec
                radius = float(radius)
                key = (item1, item2, item2_cell, radius, purpose, power_ground)
                if key in results:
                    continue
                #print(f"[Coverage QoR checking] {key}")
                if power_ground == "Power":
                    color = 'red'
                else:
                    color = 'blue'
                results[key] = []
                if item1 in all_solutions:
                    group1_objs = all_solutions[item1]
                elif item1 in self.canvas.data['S1_output'].keys():
                    group1_objs = self.get_core_region_points(item1)
                else:
                    group1_objs = []
                    violation_messages += f"[Error] Cannot find {item1}, please check the coverage limitation again!!!\n"
                if item2 in all_solutions:
                    group2_objs = all_solutions[item2]
                elif item2 in self.canvas.data['S1_output'].keys():
                    group2_objs = self.get_core_region_points(item2)
                else:
                    group2_objs = []
                    violation_messages += f"[Error] Cannot find {item2}, please check the coverage limitation again!!!\n"

                if item2_cell:
                    group2_objs = [obj for obj in group2_objs if getattr(obj, 'CellName', '') == item2_cell]
                checkpoint2 = checkpoint3
                checkpoint3 = time.time()
                #print(f"Time after get core region points {chiplet_name}: {checkpoint3 - checkpoint2:.4f} seconds")
                group1_coords, group1_obj_loc = self._get_coordinates_and_locations(group1_objs)
                group2_coords, group2_obj_loc = self._get_coordinates_and_locations(group2_objs)
                checkpoint4 = time.time()
                #print(f"Time after get coordinate points {chiplet_name}: {checkpoint4 - checkpoint3:.4f} seconds")
                overlap_dict = check_individual_overlap(group1_coords, group2_coords, radius)
                checkpoint5 = time.time()
                #print(f"Time after check overlap points {chiplet_name}: {checkpoint5 - checkpoint4:.4f} seconds")
                attr = 'TopPowerNets' if power_ground == 'Power' else 'TopGroundNets'
                missing_pclamp_domain = set()
                if item1 not in all_solutions and item2 == 'Pclamp':
                    missing_pclamp_domain = chip_domain - pclamp_domain
                if item1 in ['Pclamp', 'Bump'] or item2 in ['Pclamp', 'Bump']:
                    results[key], violation_messages = self._process_results(
                    chiplet_name, group1_obj_loc, group2_coords, overlap_dict, attr, chip_domain, item1, purpose,
                    missing_pclamp_domain, violation_messages, color)
                else:
                    results[key], violation_messages = self._process_normal_results(
                    chiplet_name, group1_obj_loc, group2_coords, overlap_dict, attr, chip_domain, item1, purpose,
                    missing_pclamp_domain, violation_messages, color)
                    #print(results[key])
                checkpoint6 = time.time()
                #print(f"Time after processing all results {chiplet_name}: {checkpoint6 - checkpoint5:.4f} seconds")

            self._dump_violations(chiplet_name, chip_qor_results, violation_messages, legend)

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.4f} seconds")


    def coverage_qor_update(self):
        if self.canvas.solution_classes is None:
            return

        qor_items = self.qor_items
        if 'coverage' not in qor_items or not qor_items['coverage']:
            return

        qor_results = self.qor_results.setdefault('coverage', {})
        checkpoint1 = time.time()

        for chiplet_name, qor_dict in qor_items['coverage'].items():
            all_solutions = self.canvas.solution_classes[chiplet_name]
            chip_qor_results = qor_results.setdefault(chiplet_name, {}).setdefault('QoR results', {})
            legend = qor_results[chiplet_name].setdefault('Legend', {'Show covered region'})
            results = chip_qor_results.setdefault('results', {})

            # Power domain check
            pclamp_domain = set()
            for pclamp in all_solutions.get('Pclamp', []):
                VDD = getattr(pclamp, 'TopPowerNets', [])
                for vdd in VDD:
                    pclamp_domain.add(vdd)
                VSS = getattr(pclamp, 'TopGroundNets', [])
                for vss in VSS:
                    pclamp_domain.add(vss)
            chip_domain = self.qor_results['coverage'][chiplet_name]['QoR results']['Chip domain']

            violation_messages = self._check_power_domain(chiplet_name, chip_domain, pclamp_domain)
            for vec in qor_dict['QoR criteria']:
                item1, item2, item2_cell, power_ground, radius, purpose = vec
                radius = float(radius)
                key = (item1, item2, item2_cell, radius, purpose, power_ground)
                if key in results:
                    continue
                #print(f"[Coverage QoR checking] {key}")
                if power_ground == "Power":
                    color = 'red'
                else:
                    color = 'blue'
                results[key] = []
                if item1 in all_solutions:
                    group1_objs = all_solutions[item1]
                elif item1 in self.canvas.data['S1_output'].keys():
                    group1_objs = self.canvas.data['S1_output'][item1]
                else:
                    group1_objs = []
                    violation_messages += f"[Error] Cannot find {item1}, please check the coverage limitation again!!!\n"
                if item2 in all_solutions:
                    group2_objs = all_solutions[item2]
                elif item2 in self.canvas.data['S1_output'].keys():
                    group2_objs = self.cavas.data['S1_output'][item2]
                else:
                    group2_objs = []
                    violation_messages += f"[Error] Cannot find {item2}, please check the coverage limitation again!!!\n"

                if item2_cell:
                    group2_objs = [obj for obj in group2_objs if getattr(obj, 'CellName', '') == item2_cell]

                violation_dict, messages = self._check_individual_overlap_revised(group1_objs, group2_objs, radius, item1, item2, chiplet_name, power_ground)
                if violation_dict:
                    results[key] = violation_dict

                for qor_criteria, qor_result in chip_qor_results['results'].items():
                    item_1, item_2, item2_cell, radius = qor_criteria[:4]
                    for pg_net, violated_list in violation_dict.items():
                        legend.add(('coverage', purpose))
                        violation_messages += f"[Coverage violations]{item_1} {item_2} {pg_net} cannot find corresponding {item2_cell}\n"
                        #print(violated_list)

                violation_messages += messages
                if violation_messages:
                    output_file = f"{self.output_dir[chiplet_name]}/vio_coverage_qor.rpt"
                    print(f"[Coverage QoR] violations dump in {output_file}")
                    with open(output_file, 'w') as f:
                        f.write(violation_messages)

    def plot_covered_region(self, all_pd_region_unioned, chiplet_name, text='', block_shape=None):
        color_map = {}
        # Create a figure and axis for plotting
        # Plot each unioned polygon
        for pg_net, unioned_polygon in all_pd_region_unioned.items():
            fig, ax = plt.subplots()
            if pg_net not in color_map:
                color_map[pg_net] = (random.random(), random.random(), random.random())
            if isinstance(unioned_polygon, MultiPolygon):
                for poly in unioned_polygon.geoms:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, color=color_map[pg_net], alpha=1)
                    #ax.plot(x, y)
                    for interior in poly.interiors:
                        x_interior, y_interior = interior.xy
                        ax.fill(x_interior, y_interior, color='white', alpha=1)
                for poly in unioned_polygon.geoms:
                    x, y = poly.exterior.xy
                    if not poly.interiors:
                        ax.fill(x, y, color=color_map[pg_net], alpha=1)
            else:
                x, y = unioned_polygon.exterior.xy
                ax.fill(x, y, color=color_map[pg_net], alpha=1)
                for interior in unioned_polygon.interiors:
                    x_interior, y_interior = interior.xy
                    ax.fill(x_interior, y_interior, color='white', alpha=1)

            if block_shape is not None:
                for poly in block_shape:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color='black')
            unique_handles = [patches.Patch(color=color_map[pg_net], label=f'{pg_net}')]

            # Add legend, labels, and title
            ax.legend(handles=unique_handles, loc='upper left', bbox_to_anchor=(1, 1))
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_title(f'{text}_{pg_net}')

            plt.axis('scaled')

            # Save the figure
            plt.savefig(self.output_dir[chiplet_name]+ f'/{text}_{pg_net}_power_domain.png')

        fig, ax = plt.subplots()
        for pg_net, unioned_polygon in all_pd_region_unioned.items():
            if pg_net not in color_map:
                color_map[pg_net] = (random.random(), random.random(), random.random())
            if isinstance(unioned_polygon, MultiPolygon):
                for poly in unioned_polygon.geoms:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, color=color_map[pg_net], alpha=1)
                    #ax.plot(x, y)
                    for interior in poly.interiors:
                        x_interior, y_interior = interior.xy
                        ax.fill(x_interior, y_interior, color='white', alpha=1)
                for poly in unioned_polygon.geoms:
                    x, y = poly.exterior.xy
                    if not poly.interiors:
                        ax.fill(x, y, color=color_map[pg_net], alpha=1)
            else:
                x, y = unioned_polygon.exterior.xy
                ax.fill(x, y, color=color_map[pg_net], alpha=1)
                for interior in unioned_polygon.interiors:
                    x_interior, y_interior = interior.xy
                    ax.fill(x_interior, y_interior, color='white', alpha=1)

        unique_handles = [patches.Patch(color=color_map[pg_net], label=f'{pg_net}') for pg_net in color_map]
        if block_shape is not None:
            for poly in block_shape:
                x, y = poly.exterior.xy
                ax.plot(x, y, color='black')

        # Add legend, labels, and title
        ax.legend(handles=unique_handles, loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'{text}_summary_all_region')

        plt.axis('scaled')

        # Save the figure
        plt.savefig(self.output_dir[chiplet_name]+ f'/{text}_summary_power_domain.png')

        # Show the plot
        plt.show()
    def _check_individual_overlap_revised(self, group1_objs, group2_objs, radius, item1, item2, chiplet_name, power_ground='Power'):
        """
        Check if points in group1 have any points in group2 within a given radius.
        Circle of group 1 points => group 2 points
        """
        # Real attribute record in the object from json
        attr = 'TopPowerNets' if power_ground == 'Power' else 'TopGroundNets'

        chip_dict = self.canvas.data['S1_output']

        execution_message = ""

        if not group1_objs:
            raise ValueError(f"Cannot find any {item1}, please check again")
            exit()
        if not group2_objs:
            raise ValueError(f"Cannot find any {item2}, please check again")
            exit()

        # All power domain needs to be checked
        all_chip_domain = set()

        if power_ground == 'Power':
            key = 'PowerNets'
        else:
            key = 'GroundNets'

        execution_message += f"Checking {item1}, {item2}, pg attribute is: {key}\n"

        for pg_net in chip_dict[chiplet_name]['TOPShape'][key]:
            all_chip_domain.add(pg_net)

        execution_message += f"{key} collected on {chiplet_name}: {all_chip_domain}\n"

        block_dict = chip_dict[chiplet_name]['BlockShapes']
        block_domain = {}
        ip_domain = {}
        for block in block_dict:
            if block['DesignName'] in self.canvas.ip_size:
                # skip the check of power domain of IPs
                ip_domain.setdefault(block['DesignName'], set())
                for pg_net in block[attr]:
                    ip_domain[block['DesignName']].add(pg_net)
            else:
                block_domain.setdefault(block['DesignName'], set())
                for pg_net in block[attr]:
                    block_domain[block['DesignName']].add(pg_net)
                    all_chip_domain.add(pg_net)

        execution_message += f"{key} excluded IP domain on {chiplet_name}: {ip_domain}\n"

        all_pd_region_to_be_covered = {}
        block_shape = None

        item1_cell = ""
        if item1 in chip_dict:
            # All regions need to be checked on top design and sub-blocks
            # Saperated by it's power domain
            shapes = []

            core_region = Polygon(self.canvas.core_region[chiplet_name])
            shapes.append(core_region)

            if power_ground == 'Power':
                key = 'PowerNets'
            else:
                key = 'GroundNets'

            for pg_net in chip_dict[chiplet_name]['TOPShape'][key]:
                all_pd_region_to_be_covered.setdefault(pg_net, [])
                bkg_polys = []
                for bkg_poly in self.canvas.marked_blockage[chiplet_name]:
                    poly = Polygon(bkg_poly)
                    bkg_polys.append(poly)
                if bkg_polys:
                    core_region = core_region.difference(unary_union(bkg_polys))
                all_pd_region_to_be_covered[pg_net].append(core_region)

            block_dict = chip_dict[chiplet_name]['BlockShapes']
            for block in block_dict:
                if block['DesignName'] in self.canvas.ip_size:
                    # skip the check of power domain of IPs
                    pass
                else:
                    for pg_net in block[attr]:
                        all_pd_region_to_be_covered.setdefault(pg_net, [])
                        core_region = Polygon(block['Coords'])
                        all_pd_region_to_be_covered[pg_net].append(core_region)
                        shapes.append(core_region)
            block_shape = shapes

        else:
            # Coords of all objects need to be checked
            # Saparated by power domain
            pd_coords_to_be_checked = {}
            objs_failed = {}
            for obj in group1_objs:
                item1_cell = getattr(obj, 'CellName', "")
                location = [calculate_centroid(get_coord(obj, self.canvas.ip_size))]
                #print(obj)
                for pg_net in getattr(obj, attr, []):
                    pd_coords_to_be_checked.setdefault(pg_net, [])
                    pd_coords_to_be_checked[pg_net].append(location[0])
                all_pd_region_to_be_covered.setdefault(pg_net, [])
                for pg_net, pt_list in pd_coords_to_be_checked.items():
                    for (x,y) in pt_list:
                        point = Point(x, y)
                        circle = point.buffer(radius)
                        all_pd_region_to_be_covered[pg_net].append(circle)


        for pg_net, polygons in all_pd_region_to_be_covered.items():
            unioned_polygon = unary_union(polygons)
            if not unioned_polygon.is_valid:
                unioned_polygon = unioned_polygon.buffer(0)
            all_pd_region_to_be_covered[pg_net] = unioned_polygon

        self.plot_covered_region(all_pd_region_to_be_covered, chiplet_name, f'{item1} {item1_cell} needed to be covered', block_shape=block_shape)
        #exit()

        all_pd_covered = {}
        all_pd_covered_coords = {}
        all_pd_region_covered = {}
        all_pd_coords = {}
        all_pd_region_to_be_covered_set = set(all_pd_region_to_be_covered.keys())

        item2_cell = ""
        for obj in group2_objs:
            item2_cell = getattr(obj, 'CellName', "")
            location = [calculate_centroid(get_coord(obj, self.canvas.ip_size))]
            relevant_pg_nets = set(getattr(obj, attr, []) or []) & all_pd_region_to_be_covered_set

            for pg_net in relevant_pg_nets:
                all_pd_coords.setdefault(pg_net, []).append(location[0])

        compared_poly = {}
        for pg_net, coords in all_pd_coords.items():
            clusters = self.cluster_coordinates(coords, radius)

            polygons_to_union = []

            if clusters:
                multi_polygon = MultiPolygon(clusters)
                if not multi_polygon.is_valid:
                    multi_polygon = multi_polygon.buffer(0)
                    if not multi_polygon.is_valid:
                        log.logger.critical(f"Invalid polygon: {multi_polygon}")
                        exit()
                all_pd_region_covered[pg_net] = multi_polygon
            else:
                log.logger.critical(f"{pg_net}: No polygons, setting an empty list.")


        self.plot_covered_region(all_pd_region_covered, chiplet_name, f'{item2} {item2_cell} covered', block_shape=block_shape)

        violated_region = {}
        all_intersection_region = {}
        if item1 not in chip_dict:
            for pg_net, poly in all_pd_region_to_be_covered.items():
                if pg_net not in all_pd_region_covered:
                    violated_region[pg_net] = pd_coords_to_be_checked[pg_net]
                    continue
                if poly.is_valid and all_pd_region_covered[pg_net].is_valid:
                    intersection = poly.intersection(all_pd_region_covered[pg_net])
                    if intersection.is_empty:
                        violated_region[pg_net] = pd_coords_to_be_checked[pg_net]
                        continue
                    else:
                        all_intersection_region[pg_net] = intersection
                else:
                    violated_region[pg_net] = pd_coords_to_be_checked[pg_net]
                    continue
                for coord in pd_coords_to_be_checked[pg_net]:
                    pt = Point(coord)
                    if not intersection.contains(pt):
                        violated_region.setdefault(pg_net, []).append(pt)
        else:
            for pg_net, poly in all_pd_region_to_be_covered.items():
                if pg_net not in all_pd_region_covered:
                    violated_region[pg_net] = poly
                    continue
                if poly.is_valid and all_pd_region_covered[pg_net].is_valid:
                    #self.plot_polygon(ax, poly, color='blue', alpha=0.5, label='MultiPolygon A')
                    #plt.show()
                    #self.plot_polygon(ax, all_pd_region_covered[pg_net], color='green', alpha=0.5, label='MultiPolygon B')
                    #plt.show()
                    difference = poly.difference(all_pd_region_covered[pg_net])
                    #plt.show()
                    #difference = poly
                    #self.plot_polygon(ax, difference, color='red', alpha=0.5, label='MultiPolygon A - B')
                    all_intersection_region[pg_net] = difference
                    if not difference.is_empty:
                        violated_region[pg_net] = difference
                else:
                    violated_region[pg_net] = poly

        if item1 not in chip_dict:
            text = 'intersection region'
        else:
            text = 'difference region'
        #self.plot_covered_region(all_intersection_region, chiplet_name, f'{item1}_{item2} {text}', block_shape=block_shape)
        self.plot_covered_region(violated_region, chiplet_name, f'Violated_result_{item1}_{item1_cell}_{item2}_{item2_cell}', block_shape=block_shape)

        return violated_region, execution_message

    def plot_polygon(self, ax, poly, color='blue', alpha=0.5, label=None):
        if poly.is_empty:
            return
        if isinstance(poly, Polygon):
            patch = patches.Polygon(list(poly.exterior.coords), closed=True, fill=True, color=color, alpha=alpha, edgecolor='black', label=label)
            ax.add_patch(patch)
        elif isinstance(poly, MultiPolygon):
            for p in poly.geoms:
                self.plot_polygon(ax, p, color=color, alpha=alpha, label=label)

    def cluster_coordinates(self, coords, radius):
        if not coords:
            return []
        #print(coords)
        tree = KDTree(coords)
        clusters = []
        single_polys = []
        visited = set()

        for idx, coord in enumerate(coords):
            if idx in visited:
                continue

            # Query neighbors within the given radius
            neighbor_indices = tree.query_ball_point(coord, radius//2)
            cluster_coords = [coords[i] for i in neighbor_indices]

            cluster_poly = []
            for pt in cluster_coords:
                point = Point(pt)
                cluster_poly.append(point.buffer(radius))
            cluster_poly.append(Point(coord).buffer(radius))

            if cluster_poly:
                cluster_poly = unary_union(cluster_poly)
                clusters.append(cluster_poly)

            visited.update(neighbor_indices)

        return clusters

    def _check_power_domain(self, chiplet_name, chip_domain, pclamp_domain):
        violation_messages = ''
        missing_in_pclamp = chip_domain - pclamp_domain
        missing_in_block = pclamp_domain - chip_domain

        if missing_in_pclamp:
            violation_messages += f"{chiplet_name} missing power domain in the pclamp: {missing_in_pclamp}\n"
            print(f"[Coverage QoR Error] {chiplet_name} missing power domain in the pclamp: {missing_in_pclamp}")
        if missing_in_block:
            violation_messages += f"{chiplet_name} missing power domain in the block: {missing_in_block}\n"
            print(f"[Coverage QoR Error] {chiplet_name} missing power domain in the block: {missing_in_block}")

        return violation_messages

    def _get_coordinates_and_locations(self, objs):
        coords, obj_loc = {}, {}
        for obj in objs:
            locations, search_objs = get_location(obj, self.canvas.ip_size)
            for loc, obj_s in zip(locations, search_objs):
                coords[loc] = obj_s
                obj_loc.setdefault(obj_s, []).append(loc)
            #print(obj_loc)
        return coords, obj_loc

    def _process_normal_results(self, chiplet_name, group1_obj_loc, group2_coords, overlap_dict, attr, chip_domain, item1, purpose, missing_pclamp_domain, violation_messages, color):
        results = []
        for obj, coords in group1_obj_loc.items():
            centroid = calculate_centroid(list(coords))
            offset = self.canvas.die_max_y[chiplet_name] * self.canvas.cell_size
            #print(obj, coords)


            for coord in coords:
                passed = False
                if not isinstance(coord, (list, tuple)) or len(coord) != 2 or not all(isinstance(c, (float, int)) for c in coord):
                    raise ValueError(f"coord format error {coord}")
                pd_list = set()
                overlap_coord = overlap_dict.get(coord, [])
                if len(overlap_coord) >= 1:
                    #print(f"found {overlap_coord}")
                    passed = True
                if not passed:
                    results.append([coord, obj, '', purpose, color])
        return results, violation_messages

    def _process_results(self, chiplet_name, group1_obj_loc, group2_coords, overlap_dict, attr, chip_domain, item1, purpose, missing_pclamp_domain, violation_messages, color):
        results = []
        for obj, coords in group1_obj_loc.items():
            golden = getattr(obj, attr, ['VDD_SOC'] if attr == 'TopPowerNets' else ['VSS'])
            centroid = calculate_centroid(list(coords))
            offset = self.canvas.die_max_y[chiplet_name] * self.canvas.cell_size
            #print(obj, coords)
            for g in golden:
                if g not in chip_domain:
                    print(f"[Coverage QoR Error] No this {g} in chip power domain {item1}!!")
                    violation_messages += f"No this {g} in chip power domain!!"
                    continue
                if g in missing_pclamp_domain:
                    print(f"[Coverage QoR Error] This power domain {g} do not have pclamp cells detected in {item1}!!")
                    violation_messages += f"This power domain {g} do not have pclamp cells detected in {item1}!!"
                    continue

                for coord in coords:
                    passed = True
                    if not isinstance(coord, (list, tuple)) or len(coord) != 2 or not all(isinstance(c, (float, int)) for c in coord):
                        raise ValueError(f"coord format error {coord}")
                    pd_list = set()
                    for overlap_coord in overlap_dict.get(coord, []):
                        #print(group2_coords)
                        #print(group2_coords.keys(), '\n', overlap_coord, '\n', group2_coords[overlap_coord].Name, getattr(group2_coords[overlap_coord], attr, set()))
                        pd = getattr(group2_coords[overlap_coord], attr, [])
                        if pd is None:
                            pd = []
                        pd = set(pd)

                        pd_list = pd | pd_list
                        #loc = getattr(group2_coords[overlap_coord], 'Location')
                        #results.append([loc, obj, g, purpose])
                    pd_list = set(pd_list)
                    #print(g, pd_list)
                    if g not in pd_list:
                        passed = False
                        #print({getattr(obj,'Name', obj)} ,{getattr(obj, 'CellName', '')}, item1, g, pd_list, passed)
                    if not passed:
                        results.append([coord, obj, g, purpose, color])
                        #for overlap_coord in overlap_dict.get(coord, []):
                        #    loc = getattr(group2_coords[overlap_coord], 'Location')
                        #    results.append([loc, obj, g, purpose])
        return results, violation_messages


    #def _process_results(self, group1_obj_loc, group2_coords, overlap_dict, attr, chip_domain, item1, purpose, missing_pclamp_domain, violation_messages, color):
    #    all_results = []
    #    all_violation_messages = ''

    #    args_list = [
    #        (obj, coords, group2_coords, overlap_dict, attr, chip_domain, item1, purpose, missing_pclamp_domain, color)
    #        for obj, coords in group1_obj_loc.items()
    #    ]

    #    with Pool(os.cpu_count()) as pool:
    #        results = pool.map(_process_single, args_list)

    #    for result, vm in results:
    #        all_results.extend(result)
    #        all_violation_messages += vm

    #    return all_results, all_violation_messages

    def _dump_violations(self, chiplet_name, chip_qor_results, violation_messages, legend):
        offset = self.canvas.die_max_y[chiplet_name] * self.canvas.cell_size
        violation_markers = []
        vio_cnt = 0

        for qor_criteria, qor_result in chip_qor_results['results'].items():
            item_1, item_2, item2_cell, radius = qor_criteria[:4]
            for r in qor_result:
                coord, obj, target_key, purpose, color = r[:]
                legend.add(('coverage', purpose))
                violation_messages += f"{getattr(obj,'Name', obj)} {getattr(obj, 'CellName', '')} {coord} cannot find corresponding {item_2} {item2_cell} for {target_key} {purpose}\n"
                violation_markers.append([violation_messages, obj, f"{target_key}, {item2_cell}", purpose, radius, f'qor-coverage-{vio_cnt}'])
                r.append(vio_cnt)
                vio_cnt += 1

        if violation_messages:
            output_file = f"{self.output_dir[chiplet_name]}/vio_coverage_qor.rpt"
            print(f"[Coverage QoR] violations dump in {output_file}")
            with open(output_file, 'w') as f:
                f.write(violation_messages)

    def make_output_dir(self):
        for chiplet in self.canvas.data['S1_output'].keys():
            self.output_dir[chiplet] = self.app.output_dir+'/qor_rpt/'+str(chiplet)
            if not os.path.exists(self.output_dir[chiplet]):
                os.makedirs(self.output_dir[chiplet])

    def distance_qor(self, mode='manhattan'):

        def shortest_manhattan_distance(points1, points2):
            points1 = np.array(points1)
            points2 = np.array(points2)
            distances = np.abs(points1[:, np.newaxis, :] - points2[np.newaxis, :, :]).sum(axis=2)

            min_index = np.unravel_index(np.argmin(distances), distances.shape)
            min_distance = distances[min_index]

            point1 = points1[min_index[0]]
            point2 = points2[min_index[1]]

            return min_distance, point1, point2

        for chiplet_name, qor_dict in self.qor_items['distance'].items():
            all_solutions = self.canvas.solution_classes[chiplet_name]
            violation_messages = ''
            offset = self.canvas.die_max_y[chiplet_name] * self.canvas.cell_size
            maze = self.canvas.create_maze(chiplet=chiplet_name, cell_size=10)
            violation_markers = []
            qor_results = self.qor_results.setdefault('distance', {}).setdefault(chiplet_name, {}).setdefault('QoR results', {})
            legend = self.qor_results['distance'][chiplet_name].setdefault('Legend', set())
            final_results = qor_results.setdefault('results', {})
            final_results.setdefault('routing', [])
            final_results.setdefault('manhattan', [])
            process_results = qor_results.setdefault('process', {})

            vio_cnt = 0
            for idx, vec in enumerate(qor_dict['QoR criteria']):
                n1, n2, rule, mode = vec
                skip_flag = False

                n1, skip_flag = self._process_node(n1, skip_flag, violation_messages)
                n2, skip_flag = self._process_node(n2, skip_flag, violation_messages)

                if skip_flag:
                    continue

                n1_pts = get_conn_location(n1)
                n2_pts = get_conn_location(n2)
                min_dist, n1_pt, n2_pt = shortest_manhattan_distance(n1_pts, n2_pts)

                maze_start = self._get_maze_coordinates(n1_pt, chiplet_name)
                maze_goal = self._get_maze_coordinates(n2_pt, chiplet_name)

                if (maze_start, maze_goal) in process_results:
                    path, total_distance = process_results[(maze_start, maze_goal)]
                elif mode == 'routing':
                    path, total_distance = bfs(maze, maze_start, maze_goal)
                    total_distance *= self.canvas.cell_size
                    process_results[(maze_start, maze_goal)] = [path, total_distance]
                else:
                    path, total_distance = [], 0

                color, distance_used, path_to_draw = self._evaluate_distance(rule, mode, min_dist, total_distance, n1_pt, n2_pt, chiplet_name, path)

                if color == 'red':
                    legend.add(mode)
                    n1_vio = getattr(n1, 'Name', str(n1))
                    n2_vio = getattr(n2, 'Name', str(n2))
                    tag = f'qor-distance-{idx}'
                    violation_messages += f"{n1_vio} {n2_vio}, distance = {distance_used}, constraint is {rule}\n"
                    violation_markers.append([f"{n1_vio} {n2_vio}, distance = {distance_used}, constraint is {rule}", n1, n2, 'Distance - ' + mode, rule, f'qor-distance-{idx}'])
                if mode == 'manhattan':
                    r = [n1, n2, rule, path_to_draw, distance_used, color, n1_pt, n2_pt]
                elif mode == 'routing':
                    r = [n1, n2, rule, path_to_draw, distance_used, color]
                if color == 'red':
                    r.append(vio_cnt)
                    vio_cnt += 1
                final_results[mode].append(r)


            if violation_messages:
                output_file = f"{self.output_dir[chiplet_name]}/vio_distance_qor.rpt"
                with open(output_file, 'w') as f:
                    print(f"Distances violation dump in file {output_file}")
                    f.write(violation_messages)

    def _process_node(self, node, skip_flag, violation_messages):
        if isinstance(node, str):
            if is_valid_location(node):
                cleaned_string = node.strip('[]')
                string_parts = cleaned_string.split(',')
                node = (float(string_parts[0]), float(string_parts[1]))
            else:
                print(f"[Distance QoR Error] Not found {node}")
                violation_messages += f"Not found {node}\n"
                skip_flag = True
        return node, skip_flag

    def _get_maze_coordinates(self, point, chiplet_name):
        maze_coord = (point[0] / self.canvas.cell_size, self.canvas.die_max_y[chiplet_name] - point[1] / self.canvas.cell_size)
        return tuple(map(int, maze_coord))

    def _evaluate_distance(self, rule, mode, min_dist, total_distance, n1_pt, n2_pt, chiplet_name, path):
        if mode == 'routing':
            distance_used = total_distance
            path_to_draw = path
        else:
            distance_used = min_dist
            path_to_draw = [(n1_pt[0] / self.canvas.cell_size, self.canvas.die_max_y[chiplet_name] - n1_pt[1] / self.canvas.cell_size),
                            (n2_pt[0] / self.canvas.cell_size, self.canvas.die_max_y[chiplet_name] - n2_pt[1] / self.canvas.cell_size)]

        color = 'blue'
        std_dist = float(rule.split('<=' if '<=' in rule else '>=' if '>=' in rule else '')[-1])
        if ('<=' in rule and distance_used > std_dist) or ('>=' in rule and distance_used < std_dist) or (distance_used != std_dist and '<=' not in rule and '>=' not in rule):
            color = 'red'

        return color, distance_used, path_to_draw
def _process_single(args):
    #print(f"ID: {os.getpid()}")
    obj, coords, group2_coords, overlap_dict, attr, chip_domain, item1, purpose, missing_pclamp_domain, color = args
    results = []
    violation_messages = ''
    golden = getattr(obj, attr, ['VDD_SOC'] if attr == 'TopPowerNets' else ['VSS'])

    for g in golden:
        if g not in chip_domain:
            print(f"[Coverage QoR Error] No this {g} in chip power domain {item1}!!")
            violation_messages += f"No this {g} in chip power domain!!"
            continue
        if g in missing_pclamp_domain:
            print(f"[Coverage QoR Error] This power domain {g} do not have pclamp cells detected in {item1}!!")
            violation_messages += f"This power domain {g} do not have pclamp cells detected in {item1}!!"
            continue

        for coord in coords:
            if not isinstance(coord, (list, tuple)) or len(coord) != 2 or not all(isinstance(c, (float, int)) for c in coord):
                raise ValueError(f"coord format error {coord}")

            pd_list = set()
            for overlap_coord in overlap_dict.get(coord, []):
                pd = getattr(group2_coords[overlap_coord], attr, [])
                if pd is None:
                    pd = []
                pd = set(pd)
                pd_list |= pd

            if g not in pd_list:
                results.append([coord, obj, g, purpose, color])

    return results, violation_messages

def read_files_from_directory(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        filtered_files = [f for f in files if not (f.endswith('.swp') or f.endswith('.swo'))]
        for file in filtered_files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


