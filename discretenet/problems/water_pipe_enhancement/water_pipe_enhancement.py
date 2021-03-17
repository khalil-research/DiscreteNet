from pathlib import Path
from typing import Union

import networkx as nx
import numpy as np
import pickle
import pyomo.environ as pyo

try:
    from importlib.resources import open_binary
except ImportError:
    from importlib_resources import open_binary

from discretenet.problem import Problem
from discretenet.generator import Generator


class WaterPipeEnhancementProblem(Problem):
    is_linear = True

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        undirected_graph: nx.MultiGraph,
        SR: list,
        C: list,
        T: list,
        name: str,
    ):
        """
        Construct a concrete Pyomo model for a Water Pipe Enhancement Problem
        :param graph: directed networkx graph, each edge represents a section of a road
            and is associated with a length representing its enhancement cost
        :param undirected_graph: undirected version of graph
        :param SR: list of sets of edges representing residential areas
        :param C: list of nodes that are critical customers
        :param T: list of nodes that are water sources
        :param name: name of the instance
        """
        super().__init__()
        self.graph = graph
        self.undirected_graph = undirected_graph
        self.SR = SR
        self.C = C
        self.T = T
        self.name = name

        self.create_model()

    def create_model(self):
        num_nodes = nx.number_of_nodes(self.graph)
        num_edges = nx.number_of_edges(self.graph)

        model = pyo.ConcreteModel()

        #  initialize variables
        model.edges = pyo.Set(initialize=list(self.graph.edges()))
        model.x = pyo.Var(model.edges, domain=pyo.Binary)
        model.y = pyo.Var(model.edges, domain=pyo.Integers)

        model.T = pyo.Set(initialize=self.T)
        model.yT = pyo.Var(model.T, domain=pyo.Integers)
        model.z = pyo.Var(domain=pyo.Integers)

        # objective function
        model.objective = pyo.Objective(
            expr=pyo.quicksum(
                edge["length"] * model.x[node1, node2]
                for node1, node2, edge in self.graph.edges(data=True)
            ),
            sense=pyo.minimize,
        )

        model.constraints = pyo.ConstraintList()
        # constraint 1:
        for node in self.graph.nodes():
            model.constraints.add(
                (model.yT[node] if node in self.T else 0)
                + pyo.quicksum(
                    model.y[node1, node2] for node1, node2 in self.graph.in_edges(node)
                )
                - pyo.quicksum(
                    model.x[node1, node2] + model.y[node1, node2]
                    for node1, node2 in self.graph.out_edges(node)
                )
                == (1 if node in self.C else 0)
            )

        # constraints 2
        visited = set()
        for node1, node2 in self.undirected_graph.edges():
            if (node1, node2) not in visited and (node2, node1) not in visited:
                model.constraints.add(
                    model.x[node1, node2] + model.x[node2, node1] <= 1
                )
                visited.add((node1, node2))
                visited.add((node2, node1))

        # constraint 3:
        for Sr in self.SR:
            cleaned_Sr = set()
            for node1, node2 in Sr:
                if (node1, node2) not in cleaned_Sr and (
                    node2,
                    node1,
                ) not in cleaned_Sr:
                    cleaned_Sr.add((node1, node2))
            model.constraints.add(
                pyo.quicksum(
                    model.x[node1, node2] + model.x[node2, node1]
                    for node1, node2 in cleaned_Sr
                )
                >= 1
            )

        # constraint 4:
        for node1, node2 in self.graph.edges():
            model.constraints.add(model.y[node1, node2] >= 0)
            model.constraints.add(
                model.y[node1, node2] - (num_edges + num_nodes) * model.x[node1, node2]
                <= 0
            )

        # constraint 5
        model.constraints.add(
            model.z + pyo.quicksum(model.yT[t] for t in self.T)
            == (num_edges + num_nodes)
        )

        # constraint 6
        model.constraints.add(
            pyo.quicksum(model.yT[t] for t in self.T)
            - pyo.quicksum(model.x[node1, node2] for node1, node2 in self.graph.edges())
            == len(self.C)
        )

        self.model = model

    def get_name(self) -> str:
        return self.name

    def get_parameters(self):
        # to be implemented
        return None


class WaterPipeEnhancementGenerator(Generator[WaterPipeEnhancementProblem]):
    def __init__(
        self,
        random_seed: int = 42,
        path_prefix: Union[str, Path] = None,
        graph_instance: str = "small_toronto",
        housing_area_rate: float = 0.01,
        housing_area_size: int = 3,
        critical_rate: float = 0.01,
        water_source_rate: float = 0.005,
    ):
        """
        Initialize the water pipe enhancement problem generator instance
        following https://doi.org/10.1145/3378393.3402246
        :param random_seed: The random seed to use
        :param path_prefix:  Path prefix to pass to instance ``save()`` methods
            during batch generation. Must be set as a public instance attribute,
            to be changed by the user after generator instantiation if desired.
        :param graph_instance: the name of the graph instance to use,
            the graphs are located under the graphs directory
        :param housing_area_rate: percentage of nodes that are center of housing areas
        :param housing_area_size: use k-hop neighbours to define each neighbourhood,
            k = r_size
        :param critical_rate: percentage of nodes that are critical customers
        :param water_source_rate: percentage of nodes that are water sources
        """
        super().__init__(random_seed, path_prefix)
        self.housing_area_rate = housing_area_rate
        self.housing_area_size = housing_area_size
        self.critical_rate = critical_rate
        self.water_source_rate = water_source_rate
        self.graph_instance = graph_instance

        with open_binary(
            "discretenet.problems.water_pipe_enhancement.graphs",
            f"{graph_instance}.gpickle",
        ) as fd:
            self.base_graph = pickle.load(fd)

        self.name = "{}_RR{}_rS{}_CR{}_TR{}".format(
            self.graph_instance,
            self.housing_area_rate,
            self.housing_area_size,
            self.critical_rate,
            self.water_source_rate,
        )

    def generate(self):
        graph = self.base_graph.copy()
        undirected_graph = graph.to_undirected()

        SR, C, T = self.generateCTR(graph)

        problem = WaterPipeEnhancementProblem(
            graph,
            undirected_graph,
            SR,
            C,
            T,
            self.name + "_%d" % self.random_seed,
        )
        return problem

    def generateCTR(self, graph):
        """
        generate:
        a set of critical customers C,
        a set of water sources T,
        a set of housing areas R and
        the set of pipes that are close enough to serve each housing area S(r) for r in R
        """
        num_nodes = nx.number_of_nodes(graph)
        C_size = max(1, int(num_nodes * self.critical_rate))
        T_size = max(1, int(num_nodes * self.water_source_rate))
        R_size = max(1, int(num_nodes * self.housing_area_rate))
        selected_nodes = np.random.choice(
            list(graph.nodes()), size=C_size + T_size + R_size
        )
        C = selected_nodes[:C_size]
        T = selected_nodes[C_size: C_size + T_size]
        R = selected_nodes[C_size + T_size:]
        for node in C:
            graph.nodes[node]["role"] = "C"
        for node in T:
            graph.nodes[node]["role"] = "T"

        SR = []
        for r in R:
            Sr = set()
            this_level = set()
            this_level.add(r)
            i = self.housing_area_size
            while i > 0:
                next_level = set()
                for node in this_level:
                    for neighbour in graph[node]:
                        graph.edges[node, neighbour, 0]["role"] = "SR"
                        edge = (node, neighbour)
                        if edge not in Sr:
                            Sr.add(edge)
                            next_level.add(neighbour)
                this_level = next_level
                i -= 1
            SR.append(Sr)

        return SR, C, T


if __name__ == "__main__":
    generator_small_toronto = WaterPipeEnhancementGenerator(
        random_seed=1,
        path_prefix="easy",
        graph_instance="small_toronto",
        housing_area_rate=0.01,
        housing_area_size=3,
        critical_rate=0.01,
        water_source_rate=0.005,
    )
    generator_small_toronto(1, 1)
