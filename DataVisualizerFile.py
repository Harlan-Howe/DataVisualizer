import math
import random
import threading
import time
from typing import Optional, List, Tuple, Callable

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.collections as collections

from threading import Thread, Lock

import numpy as np

WINDOW_RADIUS = 20
DATA_POINT_RADIUS = 0.5
STAR_RADIUS = 6
WINDOW_COLOR = (1.0, 1.0, 0.0, 0.33)

class DataVisualizer:

    def __init__(self, background_filename: str, color_map: Optional[List[Tuple[float,float,float,float]]] = None):

        if color_map is None:
            color_map = [(0,0,0,1)] # black only.
        self.color_map = color_map

        self.background = mpimg.imread(background_filename)
        self.fig, self.ax = plt.subplots()
        height, width = self.background.shape[:2]
        self.ax.imshow(self.background, extent= (0.0, float(width), float(height), 0.0))

        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)

        self.data_positions: List[Tuple[int, int]] = []
        self.data_size_list: List[float] = []
        self.data_color_indices: List[int] = []
        self.data_color_list: List[Tuple[float, float, float, float]] = []
        self.data_circles_collection: Optional[collections.Collection] = None

        self.window_positions: List[Tuple[int, int]] = []
        self.window_size_list: List[float] = []
        self.window_color_list: List[Tuple[float, float, float, float]] = []
        self.window_collection: Optional[collections.Collection] = None

        self.attractor_star_positions = self.window_positions
        self.attractor_star_size_list: List[float] = []
        self.attractor_star_color_list: List[Tuple[float, float, float, float]] = []
        self.attractor_star_collection: Optional[collections.Collection] = None
        # self.setup_attractor_collections()

        self.lock = Lock()
        self.looping_function: Optional[Callable] = None

    def setup_attractor_collections(self):
        self.window_collection = collections.CircleCollection(sizes=np.array([]),
                                                              offsets=np.array([]).reshape(-1,2),
                                                              offset_transform=self.ax.transData)

        self.window_collection.set_color(self.window_color_list)
        self.window_collection.set_sizes(self.window_size_list)
        # self.window_collection.set_offsets(np.array(self.window_positions))


        self.attractor_star_collection = collections.StarPolygonCollection(numsides=5,
                                                                           rotation=0.0,
                                                                           sizes=np.array([]),
                                                                           offsets= np.array([]).reshape(-1,2),
                                                                           offset_transform=self.ax.transData)
        self.attractor_star_collection.set_color(self.attractor_star_color_list)
        self.attractor_star_collection.set_sizes(self.attractor_star_size_list)
        # self.attractor_star_collection.set_offsets(np.array(self.attractor_star_positions))

        self.ax.add_collection(self.window_collection)
        self.ax.add_collection(self.attractor_star_collection)

    def setup_data_collection(self):


        self.data_circles_collection = collections.CircleCollection(sizes=self.data_size_list,
                                                                    offsets=[],
                                                                    offset_transform=self.ax.transData)
        self.ax.add_collection(self.data_circles_collection)

    def set_looping_function(self, func:Callable):
        self.looping_function = func

    def add_data_point(self, position: Tuple[int, int], color: int = 0):
        with self.lock:
            if self.data_circles_collection is None:
                self.setup_data_collection()
            self.data_positions.append(position)
            self.data_size_list.append(math.pi*math.pow(DATA_POINT_RADIUS,2))
            if -1 < color < len(self.color_map):  # if this is an existing color in the map
                self.data_color_indices.append(color)
            else:  # if the user chose -1 or an index outside the color map, add a new color and use that.
                self.data_color_indices.append(self.add_new_color_to_list())
            self.data_color_list.append(self.color_map[self.data_color_indices[-1]])
            print(f"{self.data_positions=}\n{self.data_size_list=}\n{self.data_color_list=}")

            self.data_circles_collection.set_offsets(np.array(self.data_positions))
            self.data_circles_collection.set_sizes(self.data_size_list)
            self.data_circles_collection.set_color(self.data_color_list)

    def update_data_point_at_index_to_color(self, idx: int, color_index: int):
        if -1 < color_index < len(self.color_map):  # if this is an existing color in the map
            self.data_color_indices[idx] = color_index
        else:  # if the user chose -1 or an index outside the color map, add a new color and use that.
            self.data_color_indices[idx] = self.add_new_color_to_list()
        self.data_color_list[idx] = self.color_map[self.data_color_indices[idx]]
        self.data_circles_collection.set_color(self.data_color_list)

    def add_attractor(self, position: Tuple[int, int], color_index: int = 0):
        with self.lock:
            if self.window_collection is None:
                self.setup_attractor_collections()
            self.window_positions.append(position)
            self.window_size_list.append(math.pi * math.pow(WINDOW_RADIUS, 2))
            self.window_color_list.append(WINDOW_COLOR)
            self.window_collection.set_offsets(np.array(self.window_positions))
            self.window_collection.set_color(self.window_color_list)
            self.window_collection.set_sizes(self.window_size_list)
            print(f"{self.window_positions=}")

            # no need to update the star attractors' positions --> we're using the same positions as the windows.
            self.attractor_star_size_list.append(math.pi * math.pow(STAR_RADIUS, 2))
            if -1 < color_index < len(self.color_map):
                self.attractor_star_color_list.append(self.color_map[color_index])
            else:
                self.attractor_star_color_list.append(self.color_map[self.add_new_color_to_list()])
            self.attractor_star_collection.set_offsets(np.array(self.attractor_star_positions))
            self.attractor_star_collection.set_color(self.attractor_star_color_list)
            self.attractor_star_collection.set_sizes(self.attractor_star_size_list)

    def set_attractor_position(self, attractor_index:int, new_position:Tuple[int,int])->None:
        with self.lock:
            self.window_positions[attractor_index] = new_position
            self.window_collection.set_offsets(np.array(self.window_positions))
            self.attractor_star_collection.set_offsets(np.array(self.attractor_star_positions))

    def remove_attractor_at_index(self, attractor_index:int):
        with self.lock:
            del(self.window_positions[attractor_index])
            del(self.window_size_list[attractor_index])
            del(self.window_color_list[attractor_index])
            del(self.attractor_star_color_list[attractor_index])
            del(self.attractor_star_size_list[attractor_index])
            if len(self.window_positions) == 0:
                self.window_collection = None
                self.attractor_star_collection = None
            else:
                self.window_collection.set_offsets(np.array(self.window_positions))
                self.window_collection.set_color(self.window_color_list)
                self.window_collection.set_sizes(self.window_size_list)
                self.attractor_star_collection.set_offsets(np.array(self.attractor_star_positions))
                self.attractor_star_collection.set_color(self.attractor_star_color_list)
                self.attractor_star_collection.set_sizes(self.attractor_star_size_list)


    def add_new_color_to_list(self) -> int:
        """
        generates a new, random color and adds it to the color list
        :return: the index of the color just added to the list.
        """
        new_color: List[float] = [random.random()*0.85 for _ in range(3)]
        new_color.append(1.0)
        self.color_map.append(tuple(new_color))
        return len(self.color_map)-1

    def update_plot(self, frame):

        if self.looping_function is not None:
            self.looping_function()
        items_to_update = []
        if self.data_circles_collection is not None:
            items_to_update.append(self.data_circles_collection)
        if self.window_collection is not None:
            items_to_update.append(self.window_collection)
            items_to_update.append(self.attractor_star_collection)
        return items_to_update

    def start_animation(self):
        self.count = 0
        ani = animation.FuncAnimation(self.fig, func=self.update_plot, interval=1000, blit=True, cache_frame_data=False)
        plt.show()
        self.stopped.set()
        return ani


# if __name__ == "__main__":
    # dv = DataVisualizer("texas56.png",[(1.0, 0.0, 0.0, 1.0),(0.0, 1.0, 0.0, 1.0),(1.0, 0.5, 0.0, 1.0)])
    # dv.add_data_point((100,100),0)
    # dv.add_data_point((200,100),1)
    # dv.add_data_point((200,200),0)
    # dv.add_data_point((30,30),-1)
    # dv.update_data_point_at_index_to_color(0, 2)
    #
    # dv.add_attractor((300,300), 2)
    # dv.add_attractor((400,700), -1)
    # ani = dv.start_animation()
