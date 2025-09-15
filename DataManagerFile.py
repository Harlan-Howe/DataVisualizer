from DataVisualizerFile import DataVisualizer

class DataManager:

    def __init__(self):
        self.dv = DataVisualizer("texas56.png",[(1.0, 0.0, 0.0, 1.0),(0.0, 1.0, 0.0, 1.0),(1.0, 0.5, 0.0, 1.0)])
        self.dv.set_axis_labels("X - axis title", "Y - axis title")

        # TODO: here is where you should load up your data. An example of data points being added is shown below.
        #       (Feel free to delete these examples, when you are adding yours.)
        self.dv.add_data_point((100,100),0)
        self.dv.add_data_point((200,100),1)
        self.dv.add_data_point((200,200),0)
        self.dv.add_data_point((30,30),-1)

        # note: this is how you'll be changing the color of a data point.
        self.dv.update_data_point_at_index_to_color(0, 2)

        # TODO: here is where you should add the attractors that you wish to display (if any). These attractors have
        #       windows around them, with a radius set at the top of DataVisualizer.

        self.dv.add_attractor((300,300), 2)
        self.dv.add_attractor((400,700), -1)

        self.iteration_counter = 0

    def start(self):

        # Tell the visualizer what method should be called each time it is about to update the graph.
        self.dv.set_looping_function(self.iterate_loop)

        # Tell the visualizer to begin animating.
        ani = self.dv.start_animation()

    def iterate_loop(self):
        """
        This method gets called over and over again by the animation loop.
        :return:
        """
        print(self.iteration_counter)
        self.iteration_counter += 1

        # TODO: here is where you will execute a step of your algorithm.


if __name__ == "__main__":
    manager = DataManager()
    manager.start()