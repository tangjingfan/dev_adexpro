import numpy as np
import torch
import matplotlib.pyplot as plt

from . import quads
from .utils import *
from . import parameter


class PredictionNodeManager:
    def __init__(self, node_manager, pred_map_info, map_info, key_coords, robot_loc, device='cpu', plot=False):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.node_manager = node_manager
        self.pred_map_info = pred_map_info
        self.pred_node_coords = None
        self.pred_node_utility = None
        self.explored_sign = None
        self.pred_prob = None
        self.device = device
        self.plot = plot
        self.path_to_nearest_frontier = None
        self.map_info = map_info
        self.initialize_graph(key_coords, robot_loc)

    def get_predicted_observation(self, robot_location, pred_map_info):
        self.update_graph()
        self.pred_map_info = pred_map_info

        all_node_coords = []
        for entry in self.nodes_dict.__iter__():
            all_node_coords.append(entry.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)

        utility = []
        explored_sign = []
        pred_prob = []
        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j

        for i, coords in enumerate(all_node_coords):
            node = self.nodes_dict.find((coords[0], coords[1])).data
            cell = get_cell_position_from_coords(coords, self.pred_map_info)
            prob = self.pred_map_info.map[cell[1], cell[0]]
            pred_prob.append(prob)
            utility.append(node.utility)
            explored_sign.append(node.explored)
            for neighbor in node.neighbor_set:
                indices = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                if indices.size > 0:
                    adjacent_matrix[i, indices[0][0]] = 0

        utility = np.array(utility)
        explored_sign = np.array(explored_sign)
        pred_prob = np.array(pred_prob)

        indices = np.argwhere(utility > 0).reshape(-1)
        utility_node_coords = all_node_coords[indices]
        dist_dict, prev_dict = self.Dijkstra(robot_location)
        guidepost = np.zeros_like(utility)
        nearest_utility_coords = robot_location
        nearest_dist = 1e8
        for end in utility_node_coords:
            if end[0] != robot_location[0] or end[1] != robot_location[1]:
                dist = dist_dict.get((end[0], end[1]), 1e8)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_utility_coords = end
        self.path_to_nearest_frontier, _ = self.get_Dijkstra_path_and_dist(dist_dict, prev_dict, nearest_utility_coords)
        for coords in self.path_to_nearest_frontier:
            idx = np.argwhere(node_coords_to_check == coords[0] + coords[1] * 1j)
            if idx.size > 0:
                guidepost[idx[0]] = 1

        current_matches = np.argwhere(node_coords_to_check == robot_location[0] + robot_location[1] * 1j)
        current_index = current_matches[0][0] if current_matches.size > 0 else 0

        neighbor_indices = []
        current_node_entry = self.nodes_dict.find((robot_location[0], robot_location[1]))
        if current_node_entry is not None:
            for neighbor in current_node_entry.data.neighbor_set:
                idx = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                if idx.size > 0:
                    neighbor_indices.append(idx[0][0])
        if current_index not in neighbor_indices:
            neighbor_indices.append(current_index)
        neighbor_indices = np.sort(np.array(neighbor_indices))

        self.pred_node_coords = all_node_coords
        self.pred_node_utility = utility
        self.explored_sign = explored_sign
        self.pred_prob = pred_prob

        node_coords = all_node_coords
        node_utility = utility.reshape(-1, 1)
        node_predprob = pred_prob.reshape(-1, 1)
        node_guidepost = explored_sign.reshape(-1, 1)
        node_guidepost2 = guidepost.reshape(-1, 1)
        edge_mask = adjacent_matrix
        current_edge = neighbor_indices
        n_node = node_coords.shape[0]

        current_node_coords = node_coords[current_index]
        node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                      node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                     axis=-1) / parameter.UPDATING_MAP_SIZE / 2
        node_utility = node_utility / (parameter.SENSOR_RANGE * 3.14 // parameter.FRONTIER_CELL_SIZE)
        node_predprob = 1 - node_predprob / parameter.OCCUPIED  # probability transformed here
        node_inputs = np.concatenate((node_coords, node_utility, node_predprob, node_guidepost, node_guidepost2), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)

        current_in_edge_matches = np.argwhere(current_edge == current_index)
        current_in_edge = current_in_edge_matches[0][0] if current_in_edge_matches.size > 0 else 0
        current_edge_t = torch.tensor(current_edge).unsqueeze(0).to(self.device)
        k_size = current_edge_t.size()[-1]
        current_edge_t = current_edge_t.unsqueeze(-1)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1

        current_index_t = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        return [node_inputs, None, edge_mask, current_index_t, current_edge_t, edge_padding_mask], \
               [all_node_coords, utility, guidepost, explored_sign, adjacent_matrix, neighbor_indices]

    def add_node_to_dict(self, coords):
        key = (coords[0], coords[1])
        node = Node(coords)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def initialize_graph(self, key_coords, robot_loc):
        node_coords = self.get_predicted_node_coords(robot_loc, key_coords, self.pred_map_info)
        for coords in node_coords:
            self.add_node_to_dict(coords)
        for entry in self.nodes_dict.__iter__():
            entry.data.get_neighbor_nodes(self.pred_map_info, self.nodes_dict)

    def update_graph(self):
        for coords, node in self.node_manager.key_node_dict.items():
            if self.nodes_dict.find(coords) is None:
                pred_node = self.add_node_to_dict(np.array(coords))
                pred_node.get_neighbor_nodes(self.pred_map_info, self.nodes_dict)
            pred_entry = self.nodes_dict.find(coords)
            if pred_entry is not None:
                pred_entry.data.utility = node.utility
                pred_entry.data.explored = 1
                pred_entry.data.visited = node.visited
                for neighbor in node.neighbor_set:
                    pred_entry.data.neighbor_set.add((neighbor[0], neighbor[1]))
                    if self.nodes_dict.find((neighbor[0], neighbor[1])) is None:
                        new_node = self.add_node_to_dict(np.array(neighbor))
                        new_node.get_neighbor_nodes(self.pred_map_info, self.nodes_dict)

    def get_predicted_node_coords(self, location, key_coords, pred_map_info):
        x_min = pred_map_info.map_origin_x
        y_min = pred_map_info.map_origin_y
        x_max = pred_map_info.map_origin_x + (pred_map_info.map.shape[1] - 1) * parameter.CELL_SIZE
        y_max = pred_map_info.map_origin_y + (pred_map_info.map.shape[0] - 1) * parameter.CELL_SIZE

        if x_min % parameter.NODE_RESOLUTION != 0:
            x_min = (x_min // parameter.NODE_RESOLUTION + 1) * parameter.NODE_RESOLUTION
        if x_max % parameter.NODE_RESOLUTION != 0:
            x_max = x_max // parameter.NODE_RESOLUTION * parameter.NODE_RESOLUTION
        if y_min % parameter.NODE_RESOLUTION != 0:
            y_min = (y_min // parameter.NODE_RESOLUTION + 1) * parameter.NODE_RESOLUTION
        if y_max % parameter.NODE_RESOLUTION != 0:
            y_max = y_max // parameter.NODE_RESOLUTION * parameter.NODE_RESOLUTION

        x_coords = np.arange(x_min, x_max + 0.1, parameter.NODE_RESOLUTION)
        y_coords = np.arange(y_min, y_max + 0.1, parameter.NODE_RESOLUTION)
        t1, t2 = np.meshgrid(x_coords, y_coords)
        nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        nodes = np.around(nodes, 1)

        free_connected_map = get_free_and_connected_map(location, pred_map_info)
        free_connected_map = np.array(free_connected_map)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, pred_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            if not (0 <= cell[1] < free_connected_map.shape[0] and 0 <= cell[0] < free_connected_map.shape[1]):
                continue
            if free_connected_map[cell[1], cell[0]] == 1:
                indices.append(i)

        key_indices = []
        key_coords_to_check = key_coords[:, 0] + key_coords[:, 1] * 1j
        for idx in indices:
            coords = nodes[idx]
            cell = get_cell_position_from_coords(coords, self.map_info)
            if not (0 <= cell[1] < self.map_info.map.shape[0] and 0 <= cell[0] < self.map_info.map.shape[1]):
                continue
            if self.map_info.map[cell[1], cell[0]] == parameter.FREE:
                if coords[0] + coords[1] * 1j in key_coords_to_check:
                    key_indices.append(idx)
            else:
                key_indices.append(idx)

        if len(key_indices) == 0:
            return nodes[indices].reshape(-1, 2) if indices else np.empty((0, 2))

        key_indices = np.array(key_indices)
        nodes = nodes[key_indices].reshape(-1, 2)

        if nodes.shape[0] > 0:
            max_predicted_distance = parameter.UPDATING_MAP_SIZE / 2
            distances = np.linalg.norm(nodes - np.array(location).reshape(1, 2), axis=1)
            known_cells = get_cell_position_from_coords(nodes, self.map_info).reshape(-1, 2)
            known_cells[:, 0] = np.clip(known_cells[:, 0], 0, self.map_info.map.shape[1] - 1)
            known_cells[:, 1] = np.clip(known_cells[:, 1], 0, self.map_info.map.shape[0] - 1)
            known_values = self.map_info.map[known_cells[:, 1], known_cells[:, 0]]
            predicted_mask = known_values == parameter.UNKNOWN
            keep_mask = np.ones(nodes.shape[0], dtype=bool)
            keep_mask[predicted_mask] = distances[predicted_mask] <= max_predicted_distance
            nodes = nodes[keep_mask].reshape(-1, 2)

        return nodes

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


class Node:
    def __init__(self, coords):
        self.coords = coords
        self.utility = -(parameter.SENSOR_RANGE * 3.14 // parameter.FRONTIER_CELL_SIZE)
        self.explored = 0
        self.visited = 0
        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_set = set()
        self.neighbor_set.add((self.coords[0], self.coords[1]))

    def get_neighbor_nodes(self, pred_map_info, nodes_dict):
        center_index = self.neighbor_matrix.shape[0] // 2
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                if i == center_index and j == center_index:
                    self.neighbor_matrix[i, j] = 1
                    continue
                neighbor_coords = np.around(
                    np.array([self.coords[0] + (i - center_index) * parameter.NODE_RESOLUTION,
                               self.coords[1] + (j - center_index) * parameter.NODE_RESOLUTION]), 1)
                entry = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                if entry is None:
                    continue
                neighbor_node = entry.data
                collision = check_collision(self.coords, neighbor_coords, pred_map_info)
                ni = center_index + (center_index - i)
                nj = center_index + (center_index - j)
                if not collision:
                    self.neighbor_matrix[i, j] = 1
                    self.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))
                    neighbor_node.neighbor_matrix[ni, nj] = 1
                    neighbor_node.neighbor_set.add((self.coords[0], self.coords[1]))
