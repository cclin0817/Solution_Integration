
✅ 還原內容如下：

from shapely.geometry import Polygon

def create_classes_from_dict(data):
    major_classes = {}

    # Define methods
    def __repr__(self):
        attributes = vars(self)
        description = ''.join(f"{key}: {value}\n" for key, value in attributes.items())
        return f"{self.__class__.__name__}\n{description}"

    def to_dict(self):
        return vars(self)

    for major_class, subclasses in data.items():
        major_classes[major_class] = {}
        for class_name, instances in subclasses.items():
            # Collect all unique attribute names across all dictionaries in the list
            attributes = {key for instance in instances for key in instance.keys()}

            # Create a dictionary with default None values for attributes and add methods
            class_dict = {attr: None for attr in attributes}
            class_dict['__repr__'] = __repr__
            class_dict['to_dict'] = to_dict

            # Create a new class dynamically
            new_class = type(class_name, (object,), class_dict)

            # Assign instances to the class
            instance_objects = []
            for instance in instances:
                obj = new_class()
                for attr, value in instance.items():
                    setattr(obj, attr, value)
                instance_objects.append(obj)

            major_classes[major_class][class_name] = instance_objects

    return major_classes

def search_objects(subclass, attribute, value):
    results = []
    for instance in subclass:
        if hasattr(instance, attribute) and getattr(instance, attribute) == value:
            results.append(instance)
            break
    if len(results) == 0:
        return False
    return results

def get_conn_location(obj):
    if isinstance(obj, (tuple, list)) and len(obj) == 2 and all(isinstance(i, (int, float)) for i in obj):
        return [obj]
    box = []
    total_x, total_y = 0, 0
    if hasattr(obj, 'PinCoords'):
        box = obj.PinCoords
    elif hasattr(obj, 'Coords'):
        box = [obj.Coords]
    
    if hasattr(obj, 'Location') and not hasattr(obj, 'PinCoords'):
        return [(obj.Location[0], obj.Location[1])]
    
    location = []
    for b in box:
        polygon = Polygon(b)
        x, y = polygon.centroid.x, polygon.centroid.y
        location.append((x,y))
    
    return location

