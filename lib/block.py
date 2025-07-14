
✅ 還原內容如下：

from shapely.geometry import box
from abc import ABC, abstractmethod

class BasePoint(ABC):
    def __init__(self, name):
        self.name = name

class PclampPoint(BasePoint):
    def __init__(self, name, asso_power, direction):
        super().__init__(name)
        self.asso_power = asso_power
        self.direction = direction

class SensorPoint(BasePoint):
    def __init__(self, name, asso_block=""):
        super().__init__(name)
        self.asso_block = asso_block

class BusPoint(BasePoint):
    def __init__(self, name):
        super().__init__(name)

class MarkedBlock:
    def __init__(self, coords, CellName, Name, color, PinCoords, has_sensor, is_sensor_assigned = False, top_power_nets = '', top_ground_nets=''):
        self.coords = coords
        self.Color = color
        self.Name = Name
        self.CellName = CellName
        self.PinCoords = PinCoords
        self.has_sensor = has_sensor
        self.is_sensor_assigned = is_sensor_assigned
        self.TopPowerNets = top_power_nets if top_power_nets is not "" else []
        self.TopGroundNets = top_ground_nets if top_ground_nets is not "" else []
        self.polygon = box(coords[0][0], coords[0][1], coords[2][0], coords[2][1])

    def __repr__(self):
        return (f"MarkedBlock(coords={self.coords}, CellName={self.CellName}, Name={self.Name}, "
                f"color={self.Color}, PinCoords={self.PinCoords}, has_sensor={self.has_sensor}, "
                f"is_sensor_assigned={self.is_sensor_assigned}, top_power_nets={self.TopPowerNets})")

    def to_dict(self):
        dict_representation = vars(self).copy()
        dict_representation.pop('polygon')
        return dict_representation

