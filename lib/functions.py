
✅ 還原內容如下：

import colorsys
from collections import deque
import numpy as np
import shapely
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import unary_union, voronoi_diagram
from scipy.spatial import KDTree
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from shapely.prepared import prep
from multiprocessing import Pool
from shapely.strtree import STRtree
from rtree import index
import matplotlib.pyplot as plt
import webcolors
from shapely.geometry.base import BaseGeometry
from . import log
import triangle

def bfs(maze, start, goal, merge_path=True):
    rows, cols = maze.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    def clamp_point_to_bounds(point):
        x, y = point
        x = max(0, min(x, rows - 1))
        y = max(0, min(y, cols - 1))
        return (x, y)
    def find_nearest_accessible_point(maze, point):
        point = clamp_point_to_bounds(point)
        if maze[point[0], point[1]] == 0:
            return point
        queue = deque([point])
        visited = set()
        while queue:
            current = queue.popleft()
            if maze[current[0], current[1]] == 0:
                return current
            visited.add(current)
            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and neighbor not in visited):
                    queue.append(neighbor)
        raise ValueError(f"Cannot find an accessible point for {point}")

    s1 = find_nearest_accessible_point(maze, start)
    s2 = find_nearest_accessible_point(maze, goal)

    queue = deque([s1])
    came_from = {s1: None}

    while queue:
        current = queue.popleft()

        if current == s2:
            path = []
            if s1 != start:
                path = [start]
            path.extend(reconstruct_path(came_from, current))
            if s2 != goal:
                path.append(goal)
            if merge_path:
                return merge_collinear_points(path)
            else:
                new_path = []
                if not path:
                    return path, 0
                prev = path[0]
                if isinstance(prev, list):
                    prev = (prev[0], prev[1])
                new_path.append(prev)
                for node_pt in path[1:]:
                    if isinstance(node_pt, list):
                        node_pt = (node_pt[0], node_pt[1])
                    if node_pt == prev:
                        pass
                    else:
                        new_path.append(node_pt)
                    prev = node_pt
                return new_path, len(new_path)-1

        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                    maze[neighbor[0], neighbor[1]] == 0 and neighbor not in came_from):
                queue.append(neighbor)
                came_from[neighbor] = current

    print(f"No path found from {start} to {goal}")
    return [], 0.0  # Return an empty list if no path is found

def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()

    return path

def merge_collinear_points(points):
    if not points:
        return [], 0.0

    merged_points = [points[0]]
    total_distance = 0.0

    for i in range(1, len(points) - 1):
        prev_point = points[i - 1]
        curr_point = points[i]
        next_point = points[i + 1]

        if (prev_point[0] == curr_point[0] == next_point[0]) or (prev_point[1] == curr_point[1] == next_point[1]):
            continue
        else:
            #print(merged_points[-1], curr_point, abs(curr_point[0] - merged_points[-1][0]) + abs(curr_point[1] - merged_points[-1][1]))
            total_distance += abs(curr_point[0] - merged_points[-1][0]) + abs(curr_point[1] - merged_points[-1][1])
            merged_points.append(curr_point)
    final_point = points[-1]
    merged_points.append(final_point)
    #print(final_point, points[-2], abs(final_point[0] - points[-2][0]) + abs(final_point[1] - points[-2][1]))
    total_distance += abs(final_point[0] - points[-2][0]) + abs(final_point[1] - points[-2][1])
    merged_points.append(points[-1])
    #print(merged_points)
    return merged_points, total_distance

def plot_maze_with_path(maze, path=[], start=[0,0], goal=[0,0]):
    fig, ax = plt.subplots()

    # Create a colormap for the maze
    cmap = plt.cm.colors.ListedColormap(['white', 'black'])
    bounds = [0, 0.5, 1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Display the maze
    ax.imshow(maze.T, cmap=cmap, norm=norm)

    # Plot the path
    if path:
        x_coords, y_coords = zip(*path)
        if isinstance(y_coords, (list, tuple)):
            y_coords = np.array(y_coords)
        ax.plot(x_coords, y_coords, color='red', linewidth=2, label='Path')

    # Highlight the start and goal points
    ax.scatter(start[0], start[1], color='blue', s=100, label='Start')
    ax.scatter(goal[0], goal[1], color='green', s=100, label='Goal')

    # Add a legend
    ax.legend()

    # Add grid lines for better visualization
    ax.set_xticks(np.arange(-.5, maze.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-.5, maze.shape[1], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    plt.show()

def shortest_manhattan_distance(points1, points2):
    points1 = np.array(points1)
    points2 = np.array(points2)
    distances = np.abs(points1[:, np.newaxis, :] - points2[np.newaxis, :, :]).sum(axis=2)
    min_index = np.unravel_index(np.argmin(distances), distances.shape)
    min_distance = distances[min_index]
    point1 = points1[min_index[0]]
    point2 = points2[min_index[1]]
    return min_distance, point1, point2

#def find_neighbors(points, maze, max_distance=250, max_deviation=10):
#    neighbors = {point: [] for point in points}
#    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#
#    for (col, row) in points:
#        for direction in directions:
#            found_neighbor = False
#            for distance in range(1, max_distance + 1):
#                for deviation in range(-max_deviation, max_deviation + 1):
#                    neighbor_col = col + direction[0] * distance + (direction[1] != 0) * deviation
#                    neighbor_row = row + direction[1] * distance + (direction[0] != 0) * deviation
#                    neighbor = (neighbor_col, neighbor_row)
#                    if neighbor in points and len(bfs(maze, (col, row), (neighbor_col, neighbor_row))) < 260:
#                        neighbors[(col, row)].append(neighbor)
#                        found_neighbor = True
#                        break
#                if found_neighbor:
#                    break
#    return neighbors

def draw_tsp_graph(points, tsp_path):
    G = nx.Graph()

    # Add nodes
    for point in points:
        G.add_node(point)

    # Add edges from the TSP path
    for i in range(len(tsp_path) - 1):
        G.add_edge(points[tsp_path[i]], points[tsp_path[i + 1]])

    # Draw the graph
    pos = {point: (point[0], point[1]) for point in points}  # Position nodes at their coordinates
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='red', node_size=500, font_size=10)
    plt.show()

def draw_graph(G, points):
    pos = {i: (points[i][0], points[i][1]) for i in range(len(points))}
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.show()

def darken_color(color_name, darken_factor=0.3):
    try:
        rgb = webcolors.name_to_rgb(color_name)
        r, g, b = [int(c * darken_factor) for c in rgb]
        return webcolors.rgb_to_hex((r, g, b))
    except ValueError as e:
        print(f"Error: {e}")
        return None

def transparent_color(base_color, transparency_ratio = 0.1):
    # transparency_ratio: 0~1
    def color_to_rgb(color):
        try:
            if isinstance(color, str) and color.startswith("#"):
                color = color.lstrip("#")
                return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
            return tuple(webcolors.name_to_rgb(color))
        except ValueError:
            raise ValueError(f"Unknown: {color}")

    def rgb_to_hex(rgb_color):
        return f"#{rgb_color[0]:02X}{rgb_color[1]:02X}{rgb_color[2]:02X}"

    base_rgb = color_to_rgb(base_color)
    background_rgb = color_to_rgb("white")

    r = round(base_rgb[0] * transparency_ratio + background_rgb[0] * (1 - transparency_ratio))
    g = round(base_rgb[1] * transparency_ratio + background_rgb[1] * (1 - transparency_ratio))
    b = round(base_rgb[2] * transparency_ratio + background_rgb[2] * (1 - transparency_ratio))

    return rgb_to_hex((r, g, b))

def is_close_to_white(rgb, threshold=0.9):
    r, g, b = rgb
    lightness = (max(r, g, b) + min(r, g, b)) / 2
    return lightness > threshold

used_colors = set()

def reset_used_color():
    global used_colors
    used_colors = set()

def generate_colors(n):
    colors = []
    global used_colors
    for i in range(n):
        hue = i / n
        lightness = 0.5
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

        while hex_color in used_colors:
            hue = (hue + 0.03) % 1.0
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            hex_color = "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

        used_colors.add(hex_color)
        colors.append(hex_color)

    return colors

def create_circles(points, radius):
    circles = []
    for point in points:
        circle = Point(point).buffer(radius)
        circles.append((point, circle))
    return circles




#def check_individual_overlap(group1_coords, group2_coords, radius):
#    """
#    Check if points in group1 have any points in group2 within a given radius.
#    Circle of group 1 points => group 2 points
#    """
#    overlap_dict = {coord: [] for coord in group1_coords.keys()} 
#
#    group2_coords_list = list(group2_coords.keys())
#    
#    if len(group2_coords_list) == 0:
#        print("No item2 detected")
#        return overlap_dict
#
#    checkpoint1 = time.time()
#    tree2 = KDTree(group2_coords_list)
#    checkpoint2 = time.time()
#    print(f"Build up KDTree {checkpoint2-checkpoint1:.4f}")
#    def query_point(coord):
#        indices = tree2.query_ball_point(coord, radius)
#        return [group2_coords_list[i] for i in indices]
#    num_workers = os.cpu_count() 
#    for coord in group1_coords.keys():
#        results = query_point(coord)
#        #print(results)
#        overlap_dict[coord].extend(results)
#
#    print(f"Total query time: {time.time() - checkpoint2:.4f}")
#    return overlap_dict

# Executor
#def check_individual_overlap(group1_coords, group2_coords, radius):
#    """
#    Check if points in group1 have any points in group2 within a given radius.
#    Circle of group 1 points => group 2 points
#    """
#    overlap_dict = {coord: [] for coord in group1_coords.keys()}
#
#    if not group2_coords:
#        print("No item2 detected")
#        return overlap_dict
#
#    group2_coords_list = list(group2_coords.keys())
#
#    # Build KDTree for group2
#    start_time = time.time()
#    tree2 = KDTree(group2_coords_list)
#    build_time = time.time() - start_time
#    print(f"Build up KDTree {build_time:.4f}")
#
#    # Function to query a single point
#    def query_point(coord):
#        indices = tree2.query_ball_point(coord, radius)
#        return coord, [group2_coords_list[i] for i in indices]
#
#    # Parallel query using ThreadPoolExecutor
#    start_query_time = time.time()
#    with ThreadPoolExecutor() as executor:
#        futures = {executor.submit(query_point, coord): coord for coord in group1_coords.keys()}
#        for future in as_completed(futures):
#            coord, results = future.result()
#            overlap_dict[coord].extend(results)
#
#    total_query_time = time.time() - start_query_time
#    print(f"Total query time: {total_query_time:.4f}")
#    
#    return overlap_dict

def query_point(args):
    #print(f"qurery ID: {os.getpid()}")
    tree2, group2_coords_list, coord, radius = args
    indices = tree2.query_ball_point(coord, radius)
    return coord, [group2_coords_list[i] for i in indices]

def check_individual_overlap(group1_coords, group2_coords, radius):
    """
    Check if points in group1 have any points in group2 within a given radius.
    Circle of group 1 points => group 2 points
    """


    overlap_dict = {coord: [] for coord in group1_coords.keys()}

    if not group2_coords:
        print("No item2 detected")
        return overlap_dict

    group2_coords_list = list(group2_coords.keys())

    # Build KDTree for group2
    start_time = time.time()
    tree2 = KDTree(group2_coords_list)
    build_time = time.time() - start_time
    print(f"Build up KDTree {build_time:.4f}")

    # Prepare arguments for parallel processing
    args = [(tree2, group2_coords_list, coord, radius) for coord in group1_coords.keys()]

    # Parallel query using multiprocessing.Pool
    start_query_time = time.time()
    with Pool() as pool:
        results = pool.map(query_point, args)

    for coord, result in results:
        overlap_dict[coord].extend(result)

    total_query_time = time.time() - start_query_time
    print(f"Total query time: {total_query_time:.4f}")

    return overlap_dict



def get_location(point_obj, ip_size):
    CellName = getattr(point_obj, 'CellName', '')
    min_coverage = 1
    pt_loc = getattr(point_obj, 'Location', None)
    location, objs = [], []

    if isinstance(point_obj, dict) and 'region' in point_obj:
        coords = point_obj['coords']
        location.append(coords)
        if point_obj['region'] == 'core':
            objs.append('VDD_SOC')
            return location, objs
        else:
            objs.append(point_obj['region'])
            return location, objs
    #print(point_obj)
    location = [(pt_loc[0], pt_loc[1])]
    objs = [point_obj]
    #print(point_obj, pt_loc)
    if CellName and CellName in ip_size:
        size_w = ip_size[CellName]['width']
        size_h = ip_size[CellName]['height']
        min_coverage = min(min_coverage, size_w, size_h)
        #print(CellName, size_w, size_h)
        if pt_loc :
            location = [(float(pt_loc[0] + size_w/2), float(pt_loc[1] + size_h/2))]
            objs = [point_obj]
        else:
            raise ValueError("[Error] No location for the object")
    if location is None:
        raise ValueError("[Error] No location for this object {point_obj}, please check again!!!")
    return location, objs

def get_coord(point_obj, ip_size):

    pt_loc = getattr(point_obj, 'Location', None)
    CellName = getattr(point_obj, 'CellName', '')
    coord = getattr(point_obj, 'Coords', None)
    if coord is not None:
        return coord

    if CellName and CellName in ip_size:
        size_w = ip_size[CellName]['width']
        size_h = ip_size[CellName]['height']
        if pt_loc :
            if type(point_obj).__name__ == "Bump":
                # Bump location is on center
                coord = [(float(pt_loc[0] - size_w/2), float(pt_loc[1] - size_h/2)),
                     (float(pt_loc[0] + size_w/2), float(pt_loc[1] - size_h/2)),
                     (float(pt_loc[0] + size_w/2), float(pt_loc[1] + size_h/2)),
                     (float(pt_loc[0] - size_w/2), float(pt_loc[1] + size_h/2)),
                     (float(pt_loc[0] - size_w/2), float(pt_loc[1] - size_h/2))]
            else:
                coord = [(float(pt_loc[0]), float(pt_loc[1])),
                     (float(pt_loc[0] + size_w), float(pt_loc[1])),
                     (float(pt_loc[0] + size_w), float(pt_loc[1] + size_h)),
                     (float(pt_loc[0]), float(pt_loc[1] + size_h)),
                     (float(pt_loc[0]), float(pt_loc[1]))]

            return coord
        else:
            return None
            #raise ValueError(f"[Error] No real size found for the object {point_obj}")

def calculate_centroid(coords):
    """
    Calculate the centroid (center) of a list of coordinates.
    """
    if not coords:
        return None
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]
    centroid_x = sum(x_coords) / len(coords)
    centroid_y = sum(y_coords) / len(coords)
    return (float(centroid_x), float(centroid_y))

def is_point_in_polygon(x, y, polygon):
    """
    Check if a point (x, y) is inside a polygon with only 90-degree or 270-degree angles.
    polygon is a list of tuples representing the vertices of the polygon.
    """
    num = len(polygon)
    j = num - 1
    inside = False

    for i in range(num):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside

def is_valid_location(input_string):
    pattern = r'^\[\d+(\.\d+)?,\d+(\.\d+)?\]$'
    return re.match(pattern, input_string) is not None

def mark_points_in_polygon(polygon, grid_size):
    min_x, min_y, max_x, max_y = polygon.bounds

    x_coords = np.arange(min_x, max_x + grid_size, grid_size)
    y_coords = np.arange(min_y, max_y + grid_size, grid_size)

    xv, yv = np.meshgrid(x_coords, y_coords)
    points = np.vstack((xv.flatten(), yv.flatten())).T
    args = [(point, polygon) for point in points]
    #print(os.cpu_count())
    with Pool(os.cpu_count()) as pool:
        valid_points = pool.map(point_in_polygon, args)
    result_points = [tuple(points[i]) for i, valid in enumerate(valid_points) if valid]

    return result_points
    #prepared_polygon = prep(polygon)
    #all_points = MultiPoint([Point(point) for point in points]) 
    #valid_points = [point.coords[0] for point in all_points if prepared_polygon.contains(point)]
    #return valid_points

def point_in_polygon(args):
    #print(f"polygon ID: {os.getpid()}")
    point, prepared_polygon = args
    return prepared_polygon.contains(Point(point))

def find_overlaps_with_rtree(polygons):
    for polygon in polygons:
        if not isinstance(polygon, BaseGeometry):
            raise ValueError(f"Not a valid polygon {polygon}")
    tree = STRtree(polygons)
    overlaps = []

    for polygon in polygons:
        potential_overlaps_indices = tree.query(polygon)
        #print(potential_overlaps_indices)
        potential_overlaps = tree.geometries.take(potential_overlaps_indices).tolist()
        #print(potential_overlaps)
        for other in potential_overlaps:
            if polygon != other:
                if isinstance(polygon, BaseGeometry) and isinstance(other, BaseGeometry):
                    if polygon.intersects(other):
                        overlap = polygon.intersection(other)
                        #print(overlap, other, polygon)
                    if not overlap.is_empty:
                        overlaps.append(overlap)
                else:
                    raise TypeError(f"polygon {polygon} or {other} should be Geometry")
    #return overlaps
    #idx = index.Index()
    #new_polygons = []
    #for i, polygon in enumerate(polygons):
    #    if not polygon.is_valid:
    #        print(f"Polygon {i} is invalid: {explain_validity(polygon)}")
    #        polygon = polygon.buffer(0)
    #        if not polygon.is_valid:
    #            print(f"Polygon {i} could not be fixed and will be skipped.")
    #            continue
    #    bounds = polygon.bounds 
    #    new_polygons.append(polygon) 
    #    idx.insert(i, bounds)

    #polygons = new_polygons

    #overlaps = []
    #for i, polygon in enumerate(polygons):
    #    possible_matches = list(idx.intersection(polygon.bounds))
    #    possible_matches.remove(i)  
    #    for j in possible_matches:
    #        if polygon.intersects(polygons[j]):
    #            overlaps.append(i)
    return overlaps

def find_outside_areas(polygons, container_polygon):
    outside_areas = []

    for polygon in polygons:
        if not container_polygon.contains(polygon):
            outside_area = polygon.difference(container_polygon)
            if not outside_area.is_empty:
                outside_areas.append(outside_area)

    return outside_areas

def polygon_with_holes_to_non_holes(pol):
    # Convert the exterior and interior coordinates to the format required by triangle
    exterior_coords = pol.exterior.coords[:-1]
    interior_coords = [list(interior.coords[:-1]) for interior in pol.interiors]
    if not interior_coords:
        return [pol]

    # Create the dict required by triangle
    polygon = {
        'vertices': exterior_coords,
        'holes': [[sum(coord) / len(coord) for coord in zip(*hole)] for hole in interior_coords]
    }
    triangulation = triangle.triangulate(polygon)
    non_hole_polygons = []
    for triangle_verts in triangulation['triangles']:
        coords = [triangulation['vertices'][i] for i in triangle_verts]
        non_hole_polygons.append(shapely.geometry.Polygon(coords))

    return non_hole_polygons

