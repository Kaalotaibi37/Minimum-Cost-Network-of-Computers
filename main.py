import numpy as np
import os
import random
import sys
import time

# Packages needed for graph, MST, and generating graph image
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
from queue import PriorityQueue

# Packages needed for GUI
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg

# Memory Usage
import psutil
from subprocess import PIPE

# Subprocess
import multiprocessing


class Graph:
    """
    Graph generates a graph using the networkx package.
    The generated graph is connected, but not necessarily a complete graph.
    The number of edges and the weight on each edge is randomly determined.
    User needs to specify node count.

    Arg:
        node_cnt: number of nodes to include in graph. >= 2.
        prb: probability threshold that an edge will be generated.
        sup_cnt: number of supercomputers. Changed to only accept 2, see assert..
    """

    # node_cnt defined by user
    # Probability defaults to 0.5; supercomputer count changed to only accept 2, see assert
    def __init__(self, node_cnt, prb=0.5, sup_cnt=2):
        if node_cnt < 2:
            raise ValueError("Graph node count cannot be smaller than 2")
        self.node_cnt = node_cnt
        self.graph = self.set_graph(node_cnt, prb)
        self.nodes = list(self.graph.nodes)
        self.edges = self.set_edges(self.graph)
        self.edge_weights = nx.get_edge_attributes(self.graph, 'weight')

    # Checks whether all vertex in graph is connected, if not, get new graph
    def set_graph(self, node_cnt, prb):
        graph = nx.generators.random_graphs.erdos_renyi_graph(node_cnt, prb)
        while not nx.is_connected(graph):
            graph = nx.generators.random_graphs.erdos_renyi_graph(node_cnt, prb)
        return graph

    # Create all edges with weight.
    # Returns a list of edge tuples of format
    # (source node, target node, {'weight': weight})
    # Weight is a random integer from the interval [1, 10)
    def set_edges(self, graph):

        for e in graph.edges():
            graph[e[0]][e[1]]['weight'] = random.randint(1, 10)

        return list(nx.to_edgelist(graph))


class MST:
    """
    MST is a superclass that is inherited by other algorithms
    to generate minimum spanning tree giving graph input
    Takes account of supernodes, initializes the supernodes
    to make them all connected.

    Arg:
        instance_of_graph: an instance of the Graph class, upon which the MST is generated.
    """

    def __init__(self, instance_of_graph, sup_cnt = 2):
        self.node_cnt = instance_of_graph.node_cnt
        self.graph = instance_of_graph.graph
        self.nodes = instance_of_graph.nodes
        # Select two random supercomputer node from available nodes
        self.sup_nodes = self.set_sup_nodes(sup_cnt)
        self.edges = instance_of_graph.edges
        self.edge_weights = instance_of_graph.edge_weights

        self.node_MST = set()  # nodes included in MST
        # First connect super computers
        for sup_node in self.sup_nodes:
            self.node_MST.add(sup_node)
        self.edge_MST = set()  # edge included in MST
        for sup_node_pair in [(i, j) for i in self.sup_nodes
                              for j in self.sup_nodes if j > i]:
            self.edge_MST.add(sup_node_pair)

    def set_sup_nodes(self, sup_cnt):

        assert(sup_cnt == 2)
        sup_nodes = random.sample(self.nodes, sup_cnt)
        while not self.graph.has_edge(sup_nodes[0], sup_nodes[1]):
            sup_nodes = random.sample(self.nodes, sup_cnt)

        return sup_nodes


class Prim(MST):
    """
    Prim inherits the MST class and generates a minimum spanning tree
    using Prim's algrithm.
    Applies a PriorityQueue to quickly get the minimum item determined by weight attribute.
    Starts at a random node (one of the super nodes), cut the graph into nodes that are in MST and nodes that are not in MST.
    For every new step, generate all the edges from nodes in MST to nodes not in MST into a PriorityQueue, by adding all edges from new_node to other nodes into PriorityQueue when a new_node is gotten from new edge added.
    Pick smallest weight from PriorityQueue.

    Arg:
        instance_of_graph: an instance of the Graph class, upon which the MST is generated.
    """

    # Inherits __init__ function of MST
    def __init__(self, instance_of_graph):
        super().__init__(instance_of_graph)

    # Start applying prim's algorithm
    def prim(self):
        # Randomly select start node from one of the supernodes
        start_node = random.choice(list(self.node_MST))

        # Use priority queue to quickly get target with minimum weight.
        # Add all edges between new node added to MST and nodes not in MST.
        # Edges are in the form of (weight, (source node, target node)).
        # Continue up to when node count in MST equals total node_cnt.
        edge_pq = PriorityQueue()

        # First add all edges from super computer nodes except start_node to edge_pq
        for sup_node in self.node_MST:
            for edges in [sorted((sup_node, i)) for i in self.nodes
                          if (i != sup_node and i not in self.node_MST)]:
                if (self.graph.has_edge(*edges)):
                    edge_pq.put((self.edge_weights[((edges[0], edges[1]))],
                        (edges[0], edges[1])
                    ))

        # Start_node is first new_node
        new_node = start_node

        # Continue while loop until all nodes are included in MST
        while len(self.node_MST) < self.node_cnt:
            for edges in [sorted((new_node, i)) for i in self.nodes
                          if (i != new_node and i not in self.node_MST)]:
                if (self.graph.has_edge(*edges)):
                    edge_pq.put((
                        self.edge_weights[((edges[0], edges[1]))],
                        (edges[0], edges[1])
                    ))

            # Makes sure item exists in edge_pq otherwise will block
            assert (edge_pq.qsize() >= 1)
            _, (source_node, target_node) = edge_pq.get()
            while (source_node in self.node_MST and
                   target_node in self.node_MST):
                assert (edge_pq.qsize() >= 1)
                _, (source_node, target_node) = edge_pq.get()
            if source_node not in self.node_MST:
                self.node_MST.add(source_node)
                new_node = source_node
            else:
                self.node_MST.add(target_node)
                new_node = target_node
            self.edge_MST.add((source_node, target_node))

        return self.edge_MST


class Kruskal(MST):
    """
    Kruskal inherits the MST class and generates a minimum spanning tree
    using Kruskal's algrithm.
    Using the union-find method to speed up the process.
    First sort all the available edges by weight.
    Continuously select edge that has lowest weight.
    Check if adding this weight to already chosen edges will result in cycle, by using the find_root method.
    If not, add edge, if yes, find new edge.
    Add edge by calling the union_by_rank method.

    Arg:
        instance_of_graph: an instance of the Graph class, upon which the MST is generated.
    """

    # Inherits __init__ function of MST and make sure super nodes are accounted for
    def __init__(self, instance_of_graph):
        super().__init__(instance_of_graph)
        self.root = {self.nodes[i]: i for i in range(len(self.nodes))}
        self.rank = {i: 0 for i in self.nodes}

        for sup_node_pair in [(i, j) for i in self.sup_nodes
                              for j in self.sup_nodes if j > i]:
            self.union_by_rank(
                self.find_root(sup_node_pair[0]),
                self.find_root(sup_node_pair[1]),
            )

    # Check whether adding new path will result in cycle
    # by finding component the node belongs in
    def find_root(self, node):
        # If node's root is itself, return itself
        if self.root[node] == node:
            return node
        # Otherwise keep going upwards till root is found
        return self.find_root(self.root[node])

    # Perform union of the two trees
    # by applying path compression.
    # Smaller rank tree is attached to bigger rank tree.
    # For two trees of same rank, randomly select one tree and add rank by 1
    def union_by_rank(self, source_comp, target_comp):
        # Attach smaller tree to larger tree
        if self.rank[source_comp] > self.rank[target_comp]:
            self.root[target_comp] = self.root[source_comp]
        elif self.rank[source_comp] < self.rank[target_comp]:
            self.root[source_comp] = self.root[target_comp]
        # Two trees of same rank, randomly choose one tree and add 1 to rank
        else:
            random_comp = random.choice((source_comp, target_comp))
            if random_comp == source_comp:
                self.root[target_comp] = source_comp
                self.rank[source_comp] += 1
            else:
                self.root[source_comp] = target_comp
                self.rank[target_comp] += 1

    # Start applying kruskal's algorithm
    def kruskal(self):
        # First sort all edges by weight
        sorted_weights = sorted(self.edges,
                                key=lambda e: self.graph.get_edge_data(e[0], e[1])['weight'])

        # Continue while loop until number of edges in MST is node count - 1
        while len(self.edge_MST) < self.node_cnt - 1:
            # Get edge with lowest weight
            edge = sorted_weights.pop(0)

            # Check no cycle by finding the components the nodes belong to
            source_comp = self.find_root(edge[0])
            target_comp = self.find_root(edge[1])

            # If not the same component, there is no cycle, perform union
            if source_comp != target_comp:
                self.node_MST.add(edge[0])
                self.node_MST.add(edge[1])
                self.edge_MST.add((edge[0], edge[1]))
                self.union_by_rank(source_comp, target_comp)


## Call subprocess to determine memory usage
def memory_subprocess(algo_choice, nodes_count):

    assert(int(nodes_count) >= 2)

    # Define subprocess and parameters
    params = [
        "python",
        __file__,
        str(algo_choice),
        str(nodes_count),
    ]

    # Create subprocess
    process = psutil.Popen(params, stdout=PIPE)

    peak_mem = 0

    # Continually calculate mem use, and compare to
    # peak_mem to get peak_mem usage
    try:
        while (process.is_running() and
               process.status() != psutil.STATUS_ZOMBIE):

            if process.status() == psutil.STATUS_ZOMBIE:
                break

            # Calculate memory utilization
            mem = process.memory_info().rss / (1024 ** 2)  # MB

            # Determine peak_mem
            if mem > peak_mem:
                peak_mem = mem
            if mem == 0.0:
                break

            # Set sleep time
            time.sleep(0.01)

    except Exception as ex:
        pass

    return peak_mem


def start_GUI(timeData_p, spaceData_p, timeData_k, spaceData_k, data_p):
    """
    Function to start GUI for user interaction.
    No arguments are needed.
    Implements GUI with pysimplegui module and Tkinter.

    Arg:
        None
    """

    scroll_max = 500
    scroll_min = 1
    # Define GUI layout
    tab1_layout = [
            [sg.Slider(range=(scroll_min, scroll_max),
                       default_value=250,
                       size=(200, 5),
                       orientation="horizontal",
                       font=("Helvetica", 11),
                       enable_events=True,
                       key="hori_slider",
                       )],
            [sg.Slider(range=(scroll_max, scroll_min),
                       default_value=250,
                       size=(200, 5),
                       orientation="vertical",
                       font=("Helvetica", 12),
                       enable_events=True,
                       key="verti_slider",
                       ),
             sg.Canvas(key="-CANVAS1-"),
                ],
            ]
    tab2_layout = [
            [sg.Canvas(key="-CANVAS2-")],
            ]
    tab3_layout = [
            [
            sg.Text("Node #", size=(8, 1)),
            sg.In("6", size=(5, 1), enable_events=True, key="-NODECHART-"),
            sg.Checkbox("Best", default=False, key="-BEST-"),
            sg.Checkbox("Average", default=True, key="-AVERAGE-"),
            sg.Checkbox("Worst", default=False, key="-WORST-"),
            sg.Text("Algo #", size=(8, 1)),
            sg.In("10", size=(5, 1), enable_events=True, key="-ALGONUM-"),
            sg.Button("Start"),#, enable_events=True, key="Start"),
            ],
            [
            sg.Text(
                "K Worst:     ",
                size=(20, 1),
                key="-WORSTTIMEK-",
            ),
            sg.Text(
                "K Avg:   ",
                size=(20, 1),
                key="-AVERAGETIMEK-",
            ),
            sg.Text(
                "K Best:   ",
                size=(20, 1),
                key="-BESTTIMEK-",
            ),
            sg.Text(
                "P Worst:     ",
                size=(20, 1),
                key="-WORSTTIMEP-",
            ),
            sg.Text(
                "P Avg:   ",
                size=(20, 1),
                key="-AVERAGETIMEP-",
            ),
            sg.Text(
                "P Best:   ",
                size=(20, 1),
                key="-BESTTIMEP-",
            ),
            ],
            [sg.Canvas(key="-CANVAS3-")],
            ]
    layout = [
        [
            sg.Text("MST", size=(8, 1), key="-TYPE-"),
            sg.Text(
                "Time Com:     ",
                size=(20, 1),
                key="-TIMECOMP-",
            ),
            sg.Text(
                "Space Com:   ",
                size=(20, 1),
                key="-SPACECOMP-",
            ),
            sg.Text("Node #", size=(8, 1)),
            sg.In("6", size=(5, 1), enable_events=True, key="-NODES-"),
            sg.Text("Edge %", size=(8, 1)),
            sg.In("0.5", size=(5, 1), enable_events=True, key="-EDGES-"),
            sg.Text("Node Size", size=(8, 1)),
            sg.In("10", size=(5, 1), enable_events=True, key="-NODESIZE-"),
            sg.Text("Edge Size", size=(8, 1)),
            sg.In("5", size=(5, 1), enable_events=True, key="-EDGESIZE-"),
        ],
        [
            sg.Button("Center"),
            sg.Button("+"),
            sg.Button("-"),
            sg.Button("Graph"),
            sg.Button("Prim"),
            sg.Button("Kruskal"),
            sg.Button("Reset"),
            sg.Button("Run"),
            sg.Button("Chart"),
            sg.Button("Exit"),
            sg.Checkbox("Space", default=False, key="-SPACE-"),
        ],
        [sg.TabGroup([[sg.Tab('Graphs', tab1_layout),
                       sg.Tab('Analyses', tab2_layout),
                       sg.Tab('Analyses2', tab3_layout),
                       ]], 
                       change_submits=True,
                       key="tabgr",
                       )],
    ]

    # Create the form and show it without the plot
    window = sg.Window(
        "Minimum Cost Network of Computers for Al-Hilal Saudi Company",
        layout,
        location=(0, 0),
        size=(800, 680),
        finalize=True,
        resizable=True,
        element_justification="center",
        font="Helvetica 18",
    )

    # Function to change scale xlim and ylim
    def scale_lim(lim, scale):
        center = (lim[0] + lim[1]) / 2
        spread = lim[1] - lim[0]
        return (center - spread / scale / 2, center + spread / scale / 2)

    # Function to change scroll xlim and ylim
    def scroll_lim(lim, scroll_value, horizontal = True):
        # Hard set figure height and width to -1 to 1 
        fig_min = -1
        fig_max = 1
        
        spread = fig_max - fig_min
        current_spread = lim[1] - lim[0]
        if horizontal:
            center = -(spread / (scroll_max - scroll_min) * scroll_value + fig_min)
        else:
            center = spread / (scroll_max - scroll_min) * scroll_value + fig_min
        if horizontal:
            return center - current_spread / 2, center + current_spread / 2
        else:
            return center - current_spread / 2, center + current_spread / 2

    # Function to draw figure that is called at every while loop
    def draw_figure(canvas, figure, loc=(0, 0)):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
        cid1 = figure_canvas_agg.mpl_connect('scroll_event', click_zoom)
        cid0 = figure_canvas_agg.mpl_connect('button_press_event', onclick)
        return figure_canvas_agg

    def click_zoom(event):

        scale = 1.05
        
        if g != 0 and values["tabgr"] == "Graphs":
            # Get the present x and y limits
            xlim = ax1.get_xlim()
            ylim = ax1.get_ylim()

            # Get event location
            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location

            # Get distance from cursor to edge of figure frame
            if (xdata is not None) and (ydata is not None):
                x_left = xdata - xlim[0]
                x_right = xlim[1] - xdata
                y_top = ydata - ylim[0]
                y_bottom = ylim[1] - ydata

                if event.button == 'up':
                    # Zoom in
                    scale_factor = 1 / scale
                elif event.button == 'down':
                    # Zoom out
                    scale_factor = scale
                else:
                    # Something that should never happen
                    scale_factor = 1

                # Set new limits
                ax1.set_xlim([xdata - x_left * scale_factor,
                             xdata + x_right * scale_factor])
                ax1.set_ylim([ydata - y_top * scale_factor,
                             ydata + y_bottom * scale_factor])

                ax1.figure.canvas.draw_idle() # force re-draw the next time the GUI refreshes

            return 0

    # Matplotlib respond to mouse click
    def onclick(event):
        if g != 0 and values["tabgr"] == "Graphs":
            if event.button == 3:
                ax1_xlim = ax1.get_xlim()
                ax1_ylim = ax1.get_ylim()
                ax1.cla()
                ax1.set_axis_off()

                ax1.set_xlim(ax1_xlim[0], ax1_xlim[1])
                ax1.set_ylim(ax1_ylim[0], ax1_ylim[1])

                # Plot graph
                nx.draw_networkx(
                    choice.graph,
                    pos,
                    alpha=0.3,
                    width=0.5,
                    ax=ax1,
                    node_size=int(values["-NODESIZE-"]),
                    node_color="black",
                )
                # Plot edge weight
                if int(values["-EDGESIZE-"]) != 0:
                    nx.draw_networkx_edge_labels(
                        choice.graph,
                        pos,
                        edge_labels=choice.edge_weights,
                        ax=ax1,
                        font_size=int(values["-EDGESIZE-"]),
                    )
               # Plot MST
                if choice != g:  # choice is MST
                    nx.draw_networkx_edges(
                        choice.graph,
                        pos,
                        edgelist=list(choice.edge_MST),
                        # alpha=0.1,
                        edge_color='g',
                        width=2,
                        ax=ax1,
                    )
                    # Plot supernodes
                    nx.draw_networkx_nodes(
                        choice.graph,
                        pos,
                        nodelist=choice.sup_nodes,
                        node_color="lightgreen",
                        ax=ax1,
                    )
     
                fig_agg.draw()
            elif event.button == 1:
                if ((pos != "") and (choice != "") and 
                    (event.xdata is not None) and 
                    (event.ydata is not None)):

                    c = ClosestNode(pos)
                    if int(values["-NODES-"]) > 100:
                        node_to_highlight = c.find_neighbor((event.xdata, event.ydata))
                    else:
                        node_to_highlight = c.within_distance((event.xdata, event.ydata))

                    if node_to_highlight != -1:
                        if choice == g:
                            edge_list = [(node_to_highlight, i) 
                                    for i in choice.nodes 
                                    if (choice.graph.has_edge(
                                        node_to_highlight, i) and 
                                        node_to_highlight != i)]
                        else:
                            edge_list = [(node_to_highlight, i) 
                                        for i in choice.node_MST 
                                        if (tuple(sorted((node_to_highlight,
                                                i))) in choice.edge_MST and                                                  node_to_highlight != i)]
                        nx.draw_networkx_edges(
                            choice.graph,
                            pos,
                            edgelist=edge_list,
                            # alpha=0.1,
                            edge_color='brown',
                            width=2,
                            ax=ax1,
                        )
                        fig_agg.draw()
            return 0

    def test_not_smaller_than_zero(target):
        while (values[target] is None) or (not values[target].isdigit()) or (int(values[target]) < 0):
            values[target] = sg.popup_get_text(
                "Node pt needs to be int and not smaller than 0. Please enter new number")
            window[target].update(values[target])

    # Initialize figure, axes, and canvas setup
    fig1 = matplotlib.figure.Figure(figsize=(50, 40), dpi=100)
    ax1 = fig1.add_subplot(111)
    ax1.set_axis_off()
    fig2 = matplotlib.figure.Figure(figsize=(50, 40), dpi=100)
    ax2 = fig2.add_subplot(111)
    ax3 = ax2.twinx()
    fig3 = matplotlib.figure.Figure(figsize=(50, 40), dpi=100)
    ax4 = fig3.add_subplot(111)
    fig_agg1 = draw_figure(window["-CANVAS1-"].TKCanvas, fig1)
    fig_agg2 = draw_figure(window["-CANVAS2-"].TKCanvas, fig2)
    fig_agg3 = draw_figure(window["-CANVAS3-"].TKCanvas, fig3)

    class ClosestNode():
        def __init__(self, pos):
            self.pos = pos

        def find_neighbor(self, coord):
            closest_key = min(self.pos.keys(), 
                        key = lambda c: (self.pos[c][0] - coord[0]) ** 2 + 
                                      (self.pos[c][1] - coord[1]) ** 2)
            return closest_key

        def within_distance(self, coord):
            dist = 0.02 # can change
            target_key = self.find_neighbor(coord)
            if ((self.pos[target_key][0] - coord[0]) ** 2 + 
                    (self.pos[target_key][1] - coord[1]) ** 2) < dist:
                return target_key
            else:
                return -1

    # Initialize values to check for item is properly intialized in while loop and to use in while loop
    g = 0
    choice = ""
    start = 0
    end = 0
    pos = ""
    timeChartData_p = multiprocessing.Manager().dict()
    spaceChartData_p = multiprocessing.Manager().dict()
    timeChartData_k = multiprocessing.Manager().dict()
    spaceChartData_k = multiprocessing.Manager().dict()
    
    node_to_hightlight = -1
    fig_agg = fig_agg1
    chart_p = ""
    ax1.cla()
    ax1.set_axis_off()
    
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)

    # Continue while loop until user presses Exit button or closes window
    while True:

        # Clears axes
        ax2.cla()
        ax3.cla()
        ax4.cla()
        
        to_show = False
        redraw = False
        change_pos = False

        # If users presses Exit button or closes window, close program
        event, values = window.read()

        if event == "Exit" or event == sg.WIN_CLOSED:
            data_p.terminate()
            if chart_p != "": chart_p.terminate()
            window.close()
            break

        test_not_smaller_than_zero("-NODESIZE-")
        test_not_smaller_than_zero("-EDGESIZE-")
        test_not_smaller_than_zero("-NODECHART-")
        test_not_smaller_than_zero("-ALGONUM-")

        # When user presses Center button
        if event == "Center":
            if g == "":
                sg.popup_ok("Please get graph first before scaling")
                continue

            ax1.set_xlim(-1, 1)
            ax1.set_ylim(-1, 1)
            to_show = True
            redraw = False
        elif event == "+" or event == "-":
            if g == "":
                sg.popup_ok("Please get graph first before scaling")
                continue

            xlim = ax1.get_xlim()
            ylim = ax1.get_ylim()
            if event == "+": scale = 1.05
            elif event == "-": scale = 0.95
            ax1.set_xlim(*scale_lim(xlim, scale))
            ax1.set_ylim(*scale_lim(ylim, scale))
            to_show = True
            redraw = False
        elif event == "hori_slider":
            if g == "":
                sg.popup_ok("Please get graph first before sliding")
                continue

            xlim = ax1.get_xlim()
            ax1.set_xlim(scroll_lim(xlim, values["hori_slider"], True))
            to_show = True
            redraw = False
        elif event == "verti_slider":
            if g == "":
                sg.popup_ok("Please get graph first before sliding")
                continue

            ylim = ax1.get_ylim()
            ax1.set_ylim(scroll_lim(ylim, values["verti_slider"], False))
            to_show = True
            redraw = False
        # When user presses Graph button
        elif event == "Graph":
            try:
                # First make sure -NODES- and -EDGES- input values are correct
                if not values["-NODES-"].isdigit() or int(values["-NODES-"]) < 2:
                    sg.popup_ok("Node # needs to be int and not smaller than 2")
                    continue
                try:
                    probability = float(values["-EDGES-"])
                    assert (probability >= 0.1 and probability <= 1)
                except:
                    sg.popup_ok("Edge % needs to be float, "
                                "not smaller than 0.1 and not bigger than 1")
                    continue

                to_show = True  # show graph
                redraw = True # draw graph
                change_pos = True # change node position

                # Get graph
                nodes_count = values["-NODES-"]
                g = Graph(int(values["-NODES-"]), float(values["-EDGES-"]))
                choice = g
                window["-TYPE-"].update("Graph")
                window["-TIMECOMP-"].update("Time Comp:     ")
                window["-SPACECOMP-"].update("Space Comp:   ")
            except:
                pass
        # When user presses Prim button
        elif event == "Prim":
            try:
                # First check graph exists to apply MST to
                if g == "":
                    sg.popup_ok("Please get graph first before running Prim")
                    continue

                to_show = False  # wait for Run pressed to show MST

                # Call subprocess to check for peak memory usage
                if int(nodes_count) in spaceData_p:
                    peak_mem = spaceData_p[int(nodes_count)]
                else:
                    peak_mem = memory_subprocess("p", nodes_count)

                # Get MST with Prim's algorithm
                p = Prim(g)
                start = time.perf_counter()  # calculate time spent
                p.prim()
                end = time.perf_counter()
                choice = p
                window["-TYPE-"].update("Prim")
                window["-TIMECOMP-"].update("Time Comp: O(ElogV)")
                window["-SPACECOMP-"].update("Space Comp: O(V)")
            except:
                pass
        # When user presses Kruskal button
        elif event == "Kruskal":
            try:
                # First check graph exists to apply MST to
                if g == "":
                    sg.popup_ok("Please get graph first before running Kruskal")
                    continue

                to_show = False  # wait for Run pressed to show MST

                # Call subprocess to check for peak memory usage
                if int(nodes_count) in spaceData_k:
                    peak_mem = spaceData_k[int(nodes_count)]
                else:
                    peak_mem = memory_subprocess("k", nodes_count)

                # Get MST with Kruskal's algorithm
                k = Kruskal(g)
                start = time.perf_counter()  # calculate time spent
                k.kruskal()
                end = time.perf_counter()
                choice = k
                window["-TYPE-"].update("Kruskal")
                window["-TIMECOMP-"].update("Time Comp: O(ElogE)")
                window["-SPACECOMP-"].update("Space Comp: O(E)")
            except:
                pass
        # When user presses Reset button, deletes all MST applied
        elif event == "Reset":
            try:
                # First check graph exists
                if g == "":
                    sg.popup_ok("Please get graph first before running Kruskal")
                    continue

                to_show = True  # show graph
                redraw = True # draw graph

                # Returns image presented to graph
                choice = g
                window["-TYPE-"].update("Graph")
                window["-TIMECOMP-"].update("Time Comp:     ")
                window["-SPACECOMP-"].update("Space Comp:   ")
            except:
                pass
        elif event == "Run":# or (choice != "" and choice == g):
            try:
                to_show = True  # show graph
                redraw = True # draw graph
            except:
                pass
        elif (event == "tabgr" or event == "Chart"):
            #try:
            if values["tabgr"] == "Graphs":
                fig_agg = fig_agg1
            elif values["tabgr"] == "Analyses":
                fig_agg = fig_agg2
                # - 2 because first node_count value is 2 not 0
                nodes_p_x = list(timeData_p.keys())[:int(values["-NODES-"]) - 2]
                time_p_x = list(timeData_p.values())[:int(values["-NODES-"]) - 2]
                time_p_x = [sum(row) / len(row) for row in time_p_x]

                ax2.plot(nodes_p_x, time_p_x, '-', color='green', 
                        label="Prim Time")

                nodes_k_x = list(timeData_k.keys())[:int(values["-NODES-"]) - 2]
                time_k_x = list(timeData_k.values())[:int(values["-NODES-"]) - 2]
                time_k_x = [sum(row) / len(row) for row in time_k_x]

                ax2.plot(nodes_k_x, time_k_x, '-b', 
                        label="Kruskal Time")
                ax2.set(xlabel="nodes", ylabel="time (ms)")
                ax2.legend(loc="upper left")

                if values["-SPACE-"] == True:
                    nodes_space_p_x = list(spaceData_p.keys())[:int(values["-NODES-"]) - 2]
                    nodes_space_k_x = list(spaceData_k.keys())[:int(values["-NODES-"]) - 2]
                    space_p_x = list(spaceData_p.values())[:int(values["-NODES-"]) - 2]
                    space_k_x = list(spaceData_k.values())[:int(values["-NODES-"]) - 2]

                    ax3.plot(nodes_space_p_x, space_p_x, '-', color="red",
                            label="Prim Space")
                    ax3.plot(nodes_space_k_x, space_k_x, '-', color="brown",
                            label="Kruskal Space")
                    ax3.legend(loc=1)
                    ax3.set(ylabel="time (ms)")

                to_show = True
            elif values["tabgr"] == "Analyses2":
                fig_agg = fig_agg3
                to_show = True
               
                node_count = 500
                algo_cnt = int(values["-ALGONUM-"])
                only_time = True

                # - 2 because first node_count value is 2 not 0
                if len(timeChartData_p) >= 1 and len(timeChartData_k) >= 1:
                    n_c = int(values["-NODECHART-"]) - 2
                    nodes_p_x = list(timeChartData_p.keys())[:n_c]
                    time_p_x = list(timeChartData_p.values())[:n_c]

                    nodes_k_x = list(timeChartData_k.keys())[:n_c]
                    time_k_x = list(timeChartData_k.values())[:n_c]

                    ax4.set(xlabel="nodes", ylabel="time (ms)")

                    time_best_p_x = [min(row) for row in time_p_x]
                    time_best_k_x = [min(row) for row in time_k_x]

                    if len(time_best_k_x) >= n_c:
                        window["-BESTTIMEK-"].update(
                          f"K Best: {time_best_k_x[n_c - 1] * 1000:.5f}")
                    if len(time_best_p_x) >= n_c:
                        window["-BESTTIMEP-"].update(
                          f"P Best: {time_best_p_x[n_c - 1] * 1000:.5f}")

                    if values["-BEST-"] == True:
                        ax4.plot(nodes_p_x, time_best_p_x, '-', color='yellow',
                                label="Prim Best Time")
                        ax4.plot(nodes_k_x, time_best_k_x, '-', color="green",                             label="Kruskal Best Time")

                    time_mean_p_x = [sum(row) / len(row) for row in time_p_x]
                    time_mean_k_x = [sum(row) / len(row) for row in time_k_x]

                    if len(time_mean_k_x) >= n_c:
                        window["-AVERAGETIMEK-"].update(
                            f"K Avg: {time_mean_k_x[n_c - 1] * 1000:.5f}")
                    if len(time_mean_p_x) >= n_c:
                        window["-AVERAGETIMEP-"].update(
                            f"P Avg: {time_mean_p_x[n_c - 1] * 1000:.5f}")
                
                    if values["-AVERAGE-"] == True:
                        ax4.plot(nodes_p_x, time_mean_p_x, '-', color='orange',                             label="Prim Average Time")
                        ax4.plot(nodes_k_x, time_mean_k_x, '-b', 
                            label="Kruskal Average Time")

                    time_worst_p_x = [max(row) for row in time_p_x]
                    time_worst_k_x = [max(row) for row in time_k_x]

                    if len(time_worst_k_x) >= n_c:
                        window["-WORSTTIMEK-"].update(
                            f"K Worst: {time_worst_k_x[n_c - 1] * 1000:.5f}")
                    if len(time_worst_p_x) >= n_c:
                        window["-WORSTTIMEP-"].update(
                            f"P Worst: {time_worst_p_x[n_c - 1] * 1000:.5f}")
                
                    if values["-WORST-"] == True:
                        ax4.plot(nodes_p_x, time_worst_p_x, '-', color='red',
                                label="Prim Worst Time")
                        ax4.plot(nodes_k_x, time_worst_k_x, '-', color="purple"                             , label="Kruskal Worst Time")
                     
                    ax4.legend(loc="upper left")

            #except:
            #    pass
        elif event == "Start":
            if chart_p != "": chart_p.terminate()
            chart_p = multiprocessing.Process(name='daemon_chart',
                                        target=collectData, 
                                        args=(
                                              timeChartData_p,
                                              spaceChartData_p,
                                              timeChartData_k,
                                              spaceChartData_k,
                                              algo_cnt,
                                              node_count,
                                              only_time,
                                              ))
            chart_p.start()
 

        if redraw:
            ax1_xlim = ax1.get_xlim()
            ax1_ylim = ax1.get_ylim()
            ax1.cla()
            ax1.set_axis_off()

            ax1.set_xlim(ax1_xlim[0], ax1_xlim[1])
            ax1.set_ylim(ax1_ylim[0], ax1_ylim[1])

        # If set to show graph / MST
        if to_show:

            if fig_agg == fig_agg1 and redraw:
                # Update window text field if choice != g
                if choice != g:
                    window["-TIMECOMP-"].update(f"Time Comp: {(end - start) * 1000:.2f}")
                    window["-SPACECOMP-"].update(f"Space Comp: {peak_mem}")

                # Choose layout
                # pos = nx.nx_pydot.graphviz_layout(choice.graph)
                if change_pos:
                    pos = nx.spring_layout(choice.graph, weight="weight")
                # Plot graph
                nx.draw_networkx(
                    choice.graph,
                    pos,
                    alpha=0.3,
                    width=0.5,
                    ax=ax1,
                    node_size=int(values["-NODESIZE-"]),
                    node_color="black",
                )
                # Plot edge weight
                if int(values["-EDGESIZE-"]) != 0:
                    nx.draw_networkx_edge_labels(
                        choice.graph,
                        pos,
                        edge_labels=choice.edge_weights,
                        ax=ax1,
                        font_size=int(values["-EDGESIZE-"]),
                    )
               # Plot MST
                if choice != g:  # choice is MST
                    nx.draw_networkx_edges(
                        choice.graph,
                        pos,
                        edgelist=list(choice.edge_MST),
                        # alpha=0.1,
                        edge_color='g',
                        width=2,
                        ax=ax1,
                    )
                    # Plot supernodes
                    nx.draw_networkx_nodes(
                        choice.graph,
                        pos,
                        nodelist=choice.sup_nodes,
                        node_color="lightgreen",
                        ax=ax1,
                    )
     
            # Add the plot to the window
            fig_agg.draw()


def command_line_access():
    """
    For command line access to be used by program to calculate peak memory usage.
    No argument needed. Everything is taken from system arguments.

    Arg:
        None
    """

    # If have command line arguments, first argument is prim or kruskal
    # Second argument is node_count, must have node_count
    assert len(sys.argv) >= 2, "Please provide node count in addition to p option"

    # Get node count
    try:
        node_count = int(sys.argv[2])
        assert (node_count >= 2)
    except:
        raise ValueError("Node count has to be int and not smaller"
                         "than 2")

    # Get edge probability
    if len(sys.argv) == 3:  # edge probability not defined, use default
        edge_prob = 0.5
    else:
        try:
            edge_prob = float(sys.argv[3])
            assert (edge_prob >= 0.1 and edge_prob <= 1)
        except:
            raise ValueError("Edge probability has to be float in "
                             "interval [0.1, 1.0]")

    g = Graph(node_count, edge_prob)
    # Prim's algorithm
    if sys.argv[1].lower() == "p" or sys.argv[1].lower() == "-p":
        p = Prim(g)
        p.prim()
        return g

    # Kruskal's algorithm
    elif sys.argv[1].lower() == "k" or sys.argv[1].lower() == "-k":
        k = Kruskal(g)
        k.kruskal()
        return g


def prim_count_time(p):
    start_p = time.perf_counter()
    p.prim()
    end_p = time.perf_counter()
    return end_p - start_p


def kruskal_count_time(k):
    start_k = time.perf_counter()
    k.kruskal()
    end_k = time.perf_counter()
    return end_k - start_k
    

def collectData(timeData_p, spaceData_p, timeData_k, spaceData_k, 
                 algo_cnt = 1, node_count = 100, only_time = False):

    # Loop through node_count, then go through algo choice 'P' & 'K'
    # Smallest node_count available is 2
    for n in range(2, node_count):
        # First test for time
        g = Graph(n)

        p = Prim(g)
        result_p = []
        for c in range(algo_cnt):
            result_p.append(prim_count_time(p))
            timeData_p[n] = result_p

        k = Kruskal(g)
        result_k = []
        timeData_k[n] = []
        for c in range(algo_cnt):
            result_k.append(kruskal_count_time(k))
            timeData_k[n] = result_k

        if not only_time:
            # Then test for memory
            spaceData_p[n] = memory_subprocess("P", n)
            spaceData_k[n] = memory_subprocess("K", n)

    return timeData_p, spaceData_p, timeData_k, spaceData_k


if __name__ == "__main__":

    if len(sys.argv) == 1:
       
        node_count = 500
        timeData_p = multiprocessing.Manager().dict()
        spaceData_p = multiprocessing.Manager().dict()
        timeData_k = multiprocessing.Manager().dict()
        spaceData_k = multiprocessing.Manager().dict()
        algo_cnt = 1

        p = multiprocessing.Process(name='daemon',
                                    target=collectData, 
                                    args=(
                                          timeData_p,
                                          spaceData_p,
                                          timeData_k,
                                          spaceData_k,
                                          algo_cnt,
                                          node_count,
                                          ))
        p.start()
        
        start_GUI(timeData_p, spaceData_p, timeData_k, spaceData_k, p)

    else:
        command_line_access()


