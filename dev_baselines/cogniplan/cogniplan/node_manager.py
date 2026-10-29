import time
import heapq
import numpy as np
from copy import deepcopy
from . import quads
from .utils import *
from .parameter import *


class NodeManager:
    def __init__(self, start=None):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.key_node_dict = {}
        self.cluster_center_node_dict = {}

        self.new_nodes = set()
        self.removed_nodes = set()
        self.updated_edges = set()

        self.path_to_nearest_frontier = None
        self.dist_to_nearest_frontier = 1e8

        if start is not None:
            self.start = np.around(start, 1)
            self.add_node_to_dict(self.start, [], None)
        else:
            self.start = None

        self.last = self.start

    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        return self.nodes_dict.find(key)

    def add_node_to_dict(self, coords, frontiers, updating_map_info):
        key = (coords[0], coords[1])
        node = Node(coords, frontiers, updating_map_info)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def remove_node_from_dict(self, node):
        for neighbor_coords in node.neighbor_set:
            if neighbor_coords[0] == node.coords[0] and neighbor_coords[1] == node.coords[1]:
                continue
            neighbor_entry = self.nodes_dict.find(neighbor_coords)
            if neighbor_entry is not None:
                neighbor_entry.data.neighbor_set.discard((node.coords[0], node.coords[1]))
        self.nodes_dict.remove(node.coords.tolist())

    def check_valid_node(self, robot_location, map_info):
        checking_boundary = get_quad_tree_box(robot_location, 2 * (SENSOR_RANGE + NODE_RESOLUTION))
        nodes_to_check = self.nodes_dict.within_bb(checking_boundary)
        for entry in nodes_to_check:
            node = entry.data
            if not is_free(node.coords, map_info):
                self.removed_nodes.add(node)
                self.remove_node_from_dict(node)

    def update_graph(self, robot_location, frontiers, updating_map_info, map_info):
        node_coords, _ = get_updating_node_coords(robot_location, updating_map_info)
        global_frontiers = get_frontier_in_map(map_info)

        all_node_list = []
        for coords in node_coords:
            entry = self.check_node_exist_in_dict(coords)
            if entry is None:
                node = self.add_node_to_dict(coords, frontiers, updating_map_info)
                self.new_nodes.add(node)
            else:
                node = entry.data
                dist_to_robot = np.linalg.norm(node.coords - robot_location)
                if dist_to_robot > 2 * SENSOR_RANGE:
                    pass
                elif dist_to_robot > THR_GRAPH_HARD_UPDATE and node.utility == 0:
                    pass
                elif dist_to_robot < THR_GRAPH_HARD_UPDATE and node.utility > 0:
                    node.initialize_observable_frontiers(frontiers, updating_map_info)
                else:
                    node.update_node_observable_frontiers(frontiers, updating_map_info, map_info, global_frontiers)
            all_node_list.append(node)

        for node in all_node_list:
            dist_to_robot = np.linalg.norm(node.coords - robot_location)
            updated_edges = set()
            if node.need_update_neighbor and dist_to_robot < (SENSOR_RANGE + NODE_RESOLUTION * 3):
                updated_edges = node.update_neighbor_nodes(updating_map_info, self.nodes_dict)
            elif dist_to_robot < THR_GRAPH_HARD_UPDATE:
                updated_edges = node.update_neighbor_nodes(updating_map_info, self.nodes_dict, hard_update=True)
            self.updated_edges = self.updated_edges.union(updated_edges)

        if self.start is not None:
            self.remove_unconnected_nodes(self.start)

        redundant = self.new_nodes & self.removed_nodes
        self.new_nodes -= redundant
        self.removed_nodes -= redundant

        current_entry = self.nodes_dict.find(robot_location.tolist())
        if current_entry is not None:
            self.last = robot_location
            return robot_location
        else:
            nearest = self.nodes_dict.nearest_neighbors(robot_location.tolist(), 1)
            if nearest:
                self.last = nearest[0].data.coords
                return nearest[0].data.coords
            return robot_location

    def remove_unconnected_nodes(self, origin):
        all_coords = set()
        for entry in self.nodes_dict.__iter__():
            c = entry.data.coords
            all_coords.add((c[0], c[1]))

        origin_entry = self.nodes_dict.find(origin.tolist())
        if origin_entry is None:
            return

        connected = set()
        connected.add((origin_entry.data.coords[0], origin_entry.data.coords[1]))
        to_expand = deepcopy(origin_entry.data.neighbor_set)

        while to_expand:
            next_expand = set()
            for coords in to_expand:
                entry = self.nodes_dict.find(coords)
                if entry is None:
                    continue
                for neighbor in entry.data.neighbor_set:
                    if neighbor not in all_coords:
                        pass
                    elif neighbor in connected:
                        pass
                    else:
                        next_expand.add(neighbor)
                connected.add(coords)
            to_expand = next_expand

        disconnected = all_coords - connected
        for coords in disconnected:
            entry = self.nodes_dict.find(coords)
            if entry is not None:
                self.removed_nodes.add(entry.data)
                self.remove_node_from_dict(entry.data)

    def Dijkstra(self, start, boundary=None):
        q = set()
        dist_dict = {}
        prev_dict = {}

        for entry in self.nodes_dict.__iter__():
            coords = entry.data.coords
            key = (coords[0], coords[1])
            dist_dict[key] = 1e8
            prev_dict[key] = None
            q.add(key)

        start_key = (start[0], start[1])
        if start_key not in dist_dict:
            return dist_dict, prev_dict
        dist_dict[start_key] = 0

        while q:
            u = min(q, key=lambda k: dist_dict[k])
            q.remove(u)
            entry = self.nodes_dict.find(u)
            if entry is None:
                continue
            for neighbor_coords in entry.data.neighbor_set:
                v = (neighbor_coords[0], neighbor_coords[1])
                if v in q:
                    cost = np.round(np.linalg.norm(np.array(v) - np.array(u)), 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict

    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        key = (end[0], end[1])
        if key not in dist_dict:
            return [], 1e8
        dist = dist_dict[key]
        path = [key]
        prev = prev_dict[key]
        while prev is not None:
            path.append(prev)
            prev = prev_dict[prev]
        path.reverse()
        return path[1:], np.round(dist, 2)

    def get_rarefied_graph(self, robot_location, map_info):
        self.dist_to_nearest_frontier = 1e8

        non_zero_utility_nodes = set()
        for entry in self.nodes_dict.__iter__():
            if entry.data.utility > 0:
                non_zero_utility_nodes.add(entry.data)

        clustered_non_zero_utility_nodes = set()

        self.key_node_dict = {}
        current_entry = self.nodes_dict.find(robot_location.tolist())
        if current_entry is None:
            nearest = self.nodes_dict.nearest_neighbors(robot_location.tolist(), 1)
            if not nearest:
                return
            current_entry = nearest[0]
        current_node = current_entry.data
        key_current_node = KeyNode(current_node.coords, current_node.utility, current_node.visited)
        self.key_node_dict[(key_current_node.coords[0], key_current_node.coords[1])] = key_current_node

        cluster_boundary = get_quad_tree_box(current_node.coords, 1 * SENSOR_RANGE)
        nodes_in_cluster = [e.data for e in self.nodes_dict.within_bb(cluster_boundary)]

        for node in nodes_in_cluster:
            if node.utility > 0:
                if (node.coords[0], node.coords[1]) in self.key_node_dict:
                    non_zero_utility_nodes.discard(node)
                    continue
                path, dist = self.a_star(current_node.coords, node.coords)
                if dist == 1e8:
                    continue
                if dist < self.dist_to_nearest_frontier:
                    self.dist_to_nearest_frontier = dist
                    self.path_to_nearest_frontier = path
                non_zero_utility_nodes.discard(node)
                ref = (robot_location[0], robot_location[1])
                ref_node = self.key_node_dict[ref]
                first_node = True
                for i, coords in enumerate(path):
                    if first_node:
                        ref_entry = self.nodes_dict.find(ref)
                        if (coords not in ref_entry.data.neighbor_set and
                                (check_collision(np.array(ref), np.array(coords), map_info) or
                                 np.linalg.norm(np.array(coords) - np.array(ref)) >
                                 (THR_NEXT_WAYPOINT + NODE_RESOLUTION))):
                            assert i >= 1
                            key_node_coords = path[i - 1]
                            if key_node_coords not in self.key_node_dict:
                                n = self.nodes_dict.find(key_node_coords).data
                                self.key_node_dict[key_node_coords] = KeyNode(n.coords, n.utility, n.visited)
                            self.key_node_dict[key_node_coords].add_neighbor_coords(ref)
                            ref_node.add_neighbor_coords(key_node_coords)
                            first_node = False
                            if coords not in self.key_node_dict:
                                n = self.nodes_dict.find(coords).data
                                self.key_node_dict[coords] = KeyNode(n.coords, n.utility, n.visited)
                        elif self.nodes_dict.find(coords).data.utility > 0:
                            if coords not in self.key_node_dict:
                                n = self.nodes_dict.find(coords).data
                                self.key_node_dict[coords] = KeyNode(n.coords, n.utility, n.visited)
                            self.key_node_dict[coords].add_neighbor_coords(ref)
                            ref_node.add_neighbor_coords(coords)
                            first_node = False
                    else:
                        if coords not in self.key_node_dict:
                            n = self.nodes_dict.find(coords).data
                            self.key_node_dict[coords] = KeyNode(n.coords, n.utility, n.visited)

        center_nodes_to_delete = []
        for center_node, _ in self.cluster_center_node_dict.values():
            if center_node.utility == 0 or center_node in nodes_in_cluster:
                center_nodes_to_delete.append(center_node)
                continue
            if self.nodes_dict.find((center_node.coords[0], center_node.coords[1])) is None:
                center_nodes_to_delete.append(center_node)
                continue
            self.update_clustered_nodes(center_node, clustered_non_zero_utility_nodes)
        for center_node in center_nodes_to_delete:
            del self.cluster_center_node_dict[(center_node.coords[0], center_node.coords[1])]

        for center_node in non_zero_utility_nodes:
            if center_node in clustered_non_zero_utility_nodes:
                continue
            dstar = None
            self.cluster_center_node_dict[(center_node.coords[0], center_node.coords[1])] = [center_node, dstar]
            self.update_clustered_nodes(center_node, clustered_non_zero_utility_nodes)

        find_one_path = False
        i = 0
        for center_node, dstar in self.cluster_center_node_dict.values():
            i += 1
            path, dist = self.a_star(robot_location, center_node.coords)
            if dist == 1e8:
                if find_one_path or i < len(self.cluster_center_node_dict):
                    continue
                else:
                    continue
            find_one_path = True
            if dist < self.dist_to_nearest_frontier:
                self.dist_to_nearest_frontier = dist
                self.path_to_nearest_frontier = path
            ref = (robot_location[0], robot_location[1])
            ref_node = self.key_node_dict[ref]
            first_node = True
            for i, coords in enumerate(path):
                if first_node:
                    ref_entry = self.nodes_dict.find(ref)
                    if (coords not in ref_entry.data.neighbor_set and
                            (check_collision(np.array(ref), np.array(coords), map_info) or
                             np.linalg.norm(np.array(coords) - np.array(ref)) >
                             (THR_NEXT_WAYPOINT + NODE_RESOLUTION))):
                        assert i >= 1
                        key_node_coords = path[i - 1]
                        if key_node_coords not in self.key_node_dict:
                            n = self.nodes_dict.find(key_node_coords).data
                            self.key_node_dict[key_node_coords] = KeyNode(n.coords, n.utility, n.visited)
                        self.key_node_dict[key_node_coords].add_neighbor_coords(ref)
                        ref_node.add_neighbor_coords(key_node_coords)
                        first_node = False
                        if coords not in self.key_node_dict:
                            n = self.nodes_dict.find(coords).data
                            self.key_node_dict[coords] = KeyNode(n.coords, n.utility, n.visited)
                    elif self.nodes_dict.find(coords).data.utility > 0:
                        if coords not in self.key_node_dict:
                            n = self.nodes_dict.find(coords).data
                            self.key_node_dict[coords] = KeyNode(n.coords, n.utility, n.visited)
                        self.key_node_dict[coords].add_neighbor_coords(ref)
                        ref_node.add_neighbor_coords(coords)
                        first_node = False
                else:
                    if coords not in self.key_node_dict:
                        n = self.nodes_dict.find(coords).data
                        self.key_node_dict[coords] = KeyNode(n.coords, n.utility, n.visited)

        for node in non_zero_utility_nodes:
            coords = (node.coords[0], node.coords[1])
            if coords not in self.key_node_dict:
                self.key_node_dict[coords] = KeyNode(node.coords, node.utility, node.visited)

        for node_coords, key_node in self.key_node_dict.items():
            entry = self.nodes_dict.find(node_coords)
            if entry is None:
                continue
            for neighbor_coords in entry.data.neighbor_set:
                if neighbor_coords in self.key_node_dict:
                    key_node.add_neighbor_coords(neighbor_coords)

        self.new_nodes = set()
        self.removed_nodes = set()
        self.updated_edges = set()

    def update_clustered_nodes(self, center_node, clustered_non_zero_utility_nodes):
        cluster_boundary = get_quad_tree_box(center_node.coords, 2 * CLUSTER_RANGE)
        nodes_in_cluster_entries = self.nodes_dict.within_bb(cluster_boundary)
        non_zero_in_cluster = {e.data for e in nodes_in_cluster_entries if e.data.utility > 0}

        nodes_to_check = set()
        for neighbor_coords in center_node.neighbor_set:
            if neighbor_coords == (center_node.coords[0], center_node.coords[1]):
                continue
            nb_entry = self.nodes_dict.find(neighbor_coords)
            if nb_entry and nb_entry.data in non_zero_in_cluster:
                clustered_non_zero_utility_nodes.add(nb_entry.data)
                nodes_to_check.add(nb_entry.data)

        while nodes_to_check:
            next_check = set()
            for node in nodes_to_check:
                for coords in node.neighbor_set:
                    if coords == (node.coords[0], node.coords[1]):
                        continue
                    nb_entry = self.nodes_dict.find(coords)
                    if nb_entry is None:
                        continue
                    nb_node = nb_entry.data
                    if nb_node in clustered_non_zero_utility_nodes or nb_node not in non_zero_in_cluster:
                        continue
                    clustered_non_zero_utility_nodes.add(nb_node)
                    for inner_coords in node.neighbor_set:
                        if inner_coords == (node.coords[0], node.coords[1]):
                            continue
                        inner_entry = self.nodes_dict.find(inner_coords)
                        if inner_entry and inner_entry.data in non_zero_in_cluster and \
                                inner_entry.data not in clustered_non_zero_utility_nodes:
                            next_check.add(inner_entry.data)
            nodes_to_check = next_check

    def h(self, coords_1, coords_2):
        return np.linalg.norm(np.array([coords_1[0] - coords_2[0], coords_1[1] - coords_2[1]]))

    def a_star(self, start, destination, max_dist=None):
        if not self.check_node_exist_in_dict(start):
            return [], 1e8
        if not self.check_node_exist_in_dict(destination):
            return [], 1e8
        if start[0] == destination[0] and start[1] == destination[1]:
            return [], 0

        start_key = (start[0], start[1])
        dest_key = (destination[0], destination[1])
        open_set = {start_key}
        closed_set = set()
        g = {start_key: 0}
        parents = {start_key: start_key}
        open_heap = [(0, start_key)]

        while open_set:
            _, n = heapq.heappop(open_heap)
            if n not in open_set:
                continue
            if max_dist is not None and g[n] > max_dist:
                return [], 1e8
            if n == dest_key:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.reverse()
                return path, np.round(length, 2)

            entry = self.nodes_dict.find(n)
            if entry is None:
                open_set.discard(n)
                closed_set.add(n)
                continue
            for neighbor_coords in entry.data.neighbor_set:
                m = (neighbor_coords[0], neighbor_coords[1])
                cost = np.linalg.norm(np.array(m) - np.array(n))
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                    heapq.heappush(open_heap, (g[m] + self.h(m, dest_key), m))
                elif g.get(m, 1e8) > g[n] + cost:
                    g[m] = g[n] + cost
                    parents[m] = n
                    heapq.heappush(open_heap, (g[m] + self.h(m, dest_key), m))

            open_set.discard(n)
            closed_set.add(n)

        return [], 1e8


class Node:
    def __init__(self, coords, frontiers, updating_map_info):
        self.coords = coords
        self.utility_range = UTILITY_RANGE
        self.utility = 0
        self.observable_frontiers = self.initialize_observable_frontiers(frontiers, updating_map_info)
        self.visited = 0

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_set = set()
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_set.add((self.coords[0], self.coords[1]))
        self.need_update_neighbor = True

    def initialize_observable_frontiers(self, frontiers, updating_map_info):
        if len(frontiers) == 0:
            self.utility = 0
            return set()
        observable_frontiers = set()
        frontiers_arr = np.array(list(frontiers)).reshape(-1, 2)
        dist_list = np.linalg.norm(frontiers_arr - self.coords, axis=-1)
        in_range = frontiers_arr[dist_list < self.utility_range]
        for point in in_range:
            if not check_collision(self.coords, point, updating_map_info, allow_unknown_endpoint=True):
                observable_frontiers.add((point[0], point[1]))
        self.utility = len(observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            observable_frontiers = set()
        return observable_frontiers

    def update_neighbor_nodes(self, updating_map_info, nodes_dict, hard_update=False):
        old_neighbor_set = deepcopy(self.neighbor_set)

        if hard_update:
            self.neighbor_matrix = -np.ones((5, 5))
            self.neighbor_matrix[2, 2] = 1

        center_index = self.neighbor_matrix.shape[0] // 2
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                if i == center_index and j == center_index:
                    self.neighbor_matrix[i, j] = 1
                    continue

                neighbor_coords = np.around(
                    np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                               self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                neighbor_entry = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                if neighbor_entry is None:
                    continue

                neighbor_node = neighbor_entry.data
                in_map = not (
                    neighbor_node.coords[0] <= updating_map_info.map_origin_x or
                    neighbor_node.coords[1] <= updating_map_info.map_origin_y or
                    neighbor_node.coords[0] >= updating_map_info.map_origin_x +
                    updating_map_info.map.shape[1] * updating_map_info.cell_size or
                    neighbor_node.coords[1] >= updating_map_info.map_origin_y +
                    updating_map_info.map.shape[0] * updating_map_info.cell_size
                )
                if not in_map:
                    continue

                collision = check_collision_type(self.coords, neighbor_coords, updating_map_info)
                ni = center_index + (center_index - i)
                nj = center_index + (center_index - j)
                if collision == FREE:
                    self.neighbor_matrix[i, j] = 1
                    self.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))
                    neighbor_node.neighbor_matrix[ni, nj] = 1
                    neighbor_node.neighbor_set.add((self.coords[0], self.coords[1]))
                elif collision == OCCUPIED:
                    self.neighbor_matrix[i, j] = 0
                    self.neighbor_set.discard((neighbor_coords[0], neighbor_coords[1]))
                    neighbor_node.neighbor_matrix[ni, nj] = 0
                    neighbor_node.neighbor_set.discard((self.coords[0], self.coords[1]))
                else:
                    self.neighbor_set.discard((neighbor_coords[0], neighbor_coords[1]))
                    neighbor_node.neighbor_set.discard((self.coords[0], self.coords[1]))

        if self.utility == 0:
            self.need_update_neighbor = False

        updated = self.neighbor_set ^ old_neighbor_set
        coords_key = (self.coords[0], self.coords[1])
        return {(coords_key, n) for n in updated}

    def update_node_observable_frontiers(self, frontiers, updating_map_info, map_info, global_frontiers):
        to_remove = [f for f in self.observable_frontiers if f not in global_frontiers]
        for f in to_remove:
            self.observable_frontiers.discard(f)

        new_frontiers = frontiers - self.observable_frontiers
        if new_frontiers:
            new_arr = np.array(list(new_frontiers)).reshape(-1, 2)
            dist_list = np.linalg.norm(new_arr - self.coords, axis=-1)
            in_range = new_arr[dist_list < self.utility_range]
            for point in in_range:
                if not check_collision(self.coords, point, updating_map_info, allow_unknown_endpoint=True):
                    self.observable_frontiers.add((point[0], point[1]))

        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            self.observable_frontiers = set()

    def set_visited(self):
        self.visited = 0
        self.observable_frontiers = set()
        self.utility = 0
        self.need_update_neighbor = False


class KeyNode:
    def __init__(self, coords, utility, visited):
        self.coords = coords
        self.utility = utility
        self.visited = visited
        self.neighbor_set = set()
        self.neighbor_set.add((self.coords[0], self.coords[1]))

    def add_neighbor_coords(self, neighbor_coords):
        self.neighbor_set.add(neighbor_coords)


class DStarLite:
    def __init__(self, nodes_dict, start, goal, max_t=0.1):
        self.nodes_dict = nodes_dict
        self.start = (start[0], start[1])
        self.goal = (goal[0], goal[1])
        self.km = 0
        self.rhs = {}
        self.g = {}
        self.U = []
        self.last = start
        self.first_run = True
        self.max_t = max_t
        self.start_t = None
        self.initialize()

    def initialize(self):
        import math
        self.km = 0
        self.rhs = {}
        self.g = {}
        self.U = []
        for entry in self.nodes_dict.__iter__():
            node = entry.data
            coords = (node.coords[0], node.coords[1])
            self.rhs[coords] = float('inf')
            self.g[coords] = float('inf')
        self.rhs[self.goal] = 0
        heapq.heappush(self.U, (self.calculate_key(self.goal), self.goal))

    def calculate_key(self, coords):
        import math
        g_rhs_min = min(self.g[coords], self.rhs[coords])
        return np.round(g_rhs_min + self.heuristic(self.start, coords) + self.km, 3), g_rhs_min

    def heuristic(self, coords1, coords2):
        import math
        return np.round(math.sqrt((coords2[0] - coords1[0]) ** 2 + (coords2[1] - coords1[1]) ** 2), 3)

    def update_node(self, node):
        import math
        node_coords = (node.coords[0], node.coords[1])
        if node_coords != self.goal:
            min_rhs = float('inf')
            for neighbor_coords in node.neighbor_set:
                if node_coords != neighbor_coords:
                    cost = np.round(math.sqrt(
                        (node_coords[0] - neighbor_coords[0]) ** 2 +
                        (node_coords[1] - neighbor_coords[1]) ** 2), 3)
                    min_rhs = min(min_rhs, np.round(self.g.get(neighbor_coords, float('inf')) + cost, 3))
            self.rhs[node_coords] = min_rhs
        self.U = [item for item in self.U if item[1] != node_coords]
        heapq.heapify(self.U)
        if self.g[node_coords] != self.rhs[node_coords]:
            heapq.heappush(self.U, (self.calculate_key(node_coords), node_coords))

    def compute_shortest_path(self):
        import time as _time
        while self.U and (self.U[0][0] < self.calculate_key(self.start) or
                          self.rhs[self.start] != self.g[self.start]):
            k_old, u = heapq.heappop(self.U)
            entry = self.nodes_dict.find(u)
            if entry is None:
                continue
            node = entry.data
            if k_old < self.calculate_key(u):
                heapq.heappush(self.U, (self.calculate_key(u), u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for neighbor_coords in node.neighbor_set:
                    if neighbor_coords != (node.coords[0], node.coords[1]):
                        nb = self.nodes_dict.find(neighbor_coords)
                        if nb:
                            self.update_node(nb.data)
            else:
                self.g[u] = float('inf')
                self.update_node(node)
                for neighbor_coords in node.neighbor_set:
                    if neighbor_coords != (node.coords[0], node.coords[1]):
                        nb = self.nodes_dict.find(neighbor_coords)
                        if nb:
                            self.update_node(nb.data)
            if _time.perf_counter() - self.start_t > self.max_t:
                return

    def get_shortest_path(self):
        import time as _time
        import math
        self.start_t = _time.perf_counter()
        self.compute_shortest_path()
        if _time.perf_counter() - self.start_t > self.max_t:
            return [], 1e8

        path = []
        current = self.start
        current_entry = self.nodes_dict.find(current)
        if current_entry is None:
            return [], 1e8
        current_node = current_entry.data
        dist = 0
        i = 0
        while current != self.goal:
            i += 1
            best_neighbor = None
            best_g = float('inf')
            for neighbor_coords in current_node.neighbor_set:
                if neighbor_coords != current:
                    g = self.g.get(neighbor_coords, float('inf'))
                    if g < best_g:
                        best_g = g
                        best_neighbor = neighbor_coords
            if len(path) >= 2 and best_neighbor == path[-2]:
                self.initialize()
                self.compute_shortest_path()
                path = []
                current = self.start
                current_entry = self.nodes_dict.find(current)
                if current_entry is None:
                    return [], 1e8
                current_node = current_entry.data
                dist = 0
                print("Path search failed! Recompute Dstar-lite")
                continue
            if self.g.get(current, float('inf')) == float('inf') or best_neighbor is None:
                print("Path not found! Unreachable node encountered")
                return [], 1e8
            if _time.perf_counter() - self.start_t > self.max_t:
                return [], 1e8
            path.append(best_neighbor)
            dist += np.round(math.sqrt((current[0] - best_neighbor[0]) ** 2 +
                                        (current[1] - best_neighbor[1]) ** 2), 3)
            current = best_neighbor
            entry = self.nodes_dict.find(current)
            if entry is None:
                return [], 1e8
            current_node = entry.data
        return path, np.round(dist, 2)

    def update_graph(self, new_start=None, new_nodes=None, deleted_nodes=None, updated_edges=None):
        if new_start:
            self.start = new_start
            self.km += self.heuristic(self.last, self.goal) - self.heuristic(self.start, self.goal)
            self.last = self.start

        nodes_to_update = set()

        if deleted_nodes:
            deleted_coords = set()
            for node in deleted_nodes:
                node_coords = (node.coords[0], node.coords[1])
                self.g.pop(node_coords, None)
                self.rhs.pop(node_coords, None)
                deleted_coords.add(node_coords)
                for neighbor_coords in node.neighbor_set:
                    if neighbor_coords != node_coords:
                        nb = self.nodes_dict.find(neighbor_coords)
                        if nb:
                            nodes_to_update.add(nb.data)
            self.U = [item for item in self.U if item[1] not in deleted_coords]
            heapq.heapify(self.U)

        if new_nodes:
            for node in new_nodes:
                coords = (node.coords[0], node.coords[1])
                self.rhs[coords] = float('inf')
                self.g[coords] = float('inf')
                nodes_to_update.add(node)
                for neighbor_coords in node.neighbor_set:
                    if neighbor_coords != coords:
                        nb = self.nodes_dict.find(neighbor_coords)
                        if nb:
                            nodes_to_update.add(nb.data)

        if updated_edges:
            for edge in updated_edges:
                u_coords, v_coords = edge[0], edge[1]
                u = self.nodes_dict.find(u_coords)
                if u:
                    nodes_to_update.add(u.data)
                v = self.nodes_dict.find(v_coords)
                if v:
                    nodes_to_update.add(v.data)

        for node in nodes_to_update:
            self.update_node(node)
