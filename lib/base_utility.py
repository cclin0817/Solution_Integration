
✅ 還原內容如下：

import tkinter as tk
from tkinter import simpledialog, StringVar
from collections import defaultdict
from abc import ABC, abstractmethod
from shapely.geometry import Point, Polygon, MultiPolygon
import matplotlib.pyplot as plt
from .functions import *
import numpy as np
from .block import *
from . import log

class CustomDirectionDialog(simpledialog.Dialog):
    def body(self, master):
        tk.Label(master, text="Select the direction for created object:").grid(row=0)
        self.var = StringVar(master)
        self.var.set("H")
        self.options = ["V", "H"]
        self.option_menu = tk.OptionMenu(master, self.var, *self.options)
        self.option_menu.grid(row=0, column=1)
        return self.option_menu

    def apply(self):
        self.result = self.var.get()

class BaseUtility(ABC):
    def __init__(self, app, canvas, args=None):
        self.app = app
        self.canvas = canvas
        self.args = args
        self.selected_point = None

    @abstractmethod
    def create_widgets(self, parent_frame):
        pass

    def update_widgets(self, parent_frame, chiplet):
        exit()

    @abstractmethod
    def on_click_canvas(self, x, y):
        pass

    @abstractmethod
    def draw(self):
        pass

    def is_point_moved(self, dx, dy, place_in_blockage):
        if self.selected_point:
            col, row = self.selected_point
            new_col, new_row = col + dx, row + dy

            if (new_col, new_row) in self.marked_points: return False
            if not self.canvas.in_core_region(col, row): return False
            if self.canvas.in_block_region(new_col, new_row): return False
            if not place_in_blockage and self.canvas.in_blockage_region(new_col, new_row): return False

            marked_point = self.marked_points.pop((col, row))
            self.marked_points[(new_col, new_row)] = marked_point
            self.selected_point = (new_col, new_row)
            self.update_point_list(self.marked_points)
            return True
        else:
            return False

    def on_click_point_view(self, event):
        selection = self.pointView.selection()
        if selection:
            self.context_menu.tk_popup(event.x_root, event.y_root)

    def on_pointView_select(self, event):
        selection = self.pointView.selection()
        if selection:
            item = self.pointView.item(selection[0])
            point_name = item["values"][0]
            point_str = item["values"][1]
            coord_part = point_str[1:-1]
            col, row = map(int, coord_part.split(", "))
            self.canvas.center_on_point(col, row, item["values"][0])
            self.selected_point = tuple(map(int, coord_part.split(", ")))
            log.logger.info(f"[INFO] Select point {point_name} ({col}, {row})")

    def rename_points(self):
        sorted_points = sorted(self.marked_points.items(), key=lambda item: int(item[1].name))
        new_marked_points = {}
        for index, ((col, row), marked_point) in enumerate(sorted_points):
            new_name = str(index)
            marked_point.name = new_name
        self.update_point_list(self.marked_points)

    def draw_points(self, power="None"):
        self.canvas.delete("point")
        for (col, row), marked_point in self.marked_points.items():
            self.fill_cell(col, row, marked_point.name, "point", power=power)

    def fill_cell(self, col, row, name, tag, color="tomato", size=15, power="None"):
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


