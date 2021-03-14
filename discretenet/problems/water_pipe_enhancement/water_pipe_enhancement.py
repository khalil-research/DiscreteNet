from pathlib import Path
import numpy.random as random
from typing import Union
import pyomo.environ as pyo
import networkx as nx
try:
    from importlib.resources import open_binary
except ImportError:
    from importlib_resources import open_binary
import pickle

from discretenet.problem import Problem
from discretenet.generator import Generator


class WaterPipeEnhancementProblem(Problem):
    def __init__(self, graph, undirected_graph, SR, C, T, name):
        self.model = None
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
        # model.add_constr(z >= 0)

        # constraint 6
        model.constraints.add(
            pyo.quicksum(model.yT[t] for t in self.T)
            - pyo.quicksum(model.x[node1, node2] for node1, node2 in self.graph.edges())
            == len(self.C)
        )

        self.model = model

    def is_linear(self) -> bool:
        """
        Boolean indicating if the model objective and constraints are linear.

        Must be set when creating a concrete problem class. ::

            class MyProblem(Problem):
                is_linear = True

                ...
        """
        return True

    def get_name(self) -> str:
        """
        Return a string name for the model instance, based on problem parameters

        Use for saving parts of the model, but most not include path or extension.

        :return: Model instance name
        """
        return self.name


class WaterPipeEnhancementGenerator(Generator):
    def __init__(
            self,
            random_seed: int = 42,
            path_prefix: Union[str, Path] = None,
            graph_instance="small_toronto",
            housing_area_rate=0.01,
            housing_area_size=3,
            critical_rate=0.01,
            water_source_rate=0.005,
    ):
        """
        Initialize the water pipe enhancement problem generator instance following https://doi.org/10.1145/3378393.3402246
        :param random_seed: The random seed to use
        :param path_prefix:  Path prefix to pass to instance ``save()`` methods
            during batch generation. Must be set as a public instance attribute,
            to be changed by the user after generator instantiation if desired.
        :param graph_instance: the name of the graph instance to use, the graphs are located under the graphs directory
        :param housing_area_rate: percentage of nodes that are center of housing areas
        :param housing_area_size: use k-hop neighbours to define each neighbourhood, k = r_size
        :param critical_rate: percentage of nodes that are critical customers
        :param water_source_rate: percentage of nodes that are water sources
        """
        super().__init__(random_seed, path_prefix)
        self.housing_area_rate = housing_area_rate
        self.housing_area_size = housing_area_size
        self.critical_rate = critical_rate
        self.water_source_rate = water_source_rate
        self.graph_instance = graph_instance
        self.SR = None
        self.C = None
        self.T = None
        self.name = None

        with open_binary(
            "discretenet.problems.water_pipe_enhancement.graphs",
            f"{graph_instance}.gpickle"
        ) as fd:
            self.graph = pickle.load(fd)
        self.undirected_graph = self.graph.to_undirected()

    def generate(self):

        self.generateCTR()
        self.name = "{}_RR{}_rS{}_CR{}_TR{}".format(
            self.graph_instance,
            self.housing_area_rate,
            self.housing_area_size,
            self.critical_rate,
            self.water_source_rate,
        )
        problem = WaterPipeEnhancementProblem(
            self.graph,
            self.undirected_graph,
            self.SR,
            self.C,
            self.T,
            self.name + "_%d" % self.random_seed,
        )
        return problem

    def generateCTR(self):
        """
        generate:
        a set of critical customers C,
        a set of water sources T,
        a set of housing areas R and
        the set of pipes that are close enough to serve each housing area S(r) for r in R
        """
        num_nodes = nx.number_of_nodes(self.graph)
        C_size = max(1, int(num_nodes * self.critical_rate))
        T_size = max(1, int(num_nodes * self.water_source_rate))
        R_size = max(1, int(num_nodes * self.housing_area_rate))
        selected_nodes = random.choice(
            list(self.graph.nodes()), size=C_size + T_size + R_size
        )
        C = selected_nodes[:C_size]
        T = selected_nodes[C_size: C_size + T_size]
        R = selected_nodes[C_size + T_size:]
        for node in C:
            self.graph.nodes[node]["role"] = "C"
        for node in T:
            self.graph.nodes[node]["role"] = "T"

        SR = []
        for r in R:
            Sr = set()
            this_level = set()
            this_level.add(r)
            i = self.housing_area_size
            while i > 0:
                next_level = set()
                for node in this_level:
                    for neighbour in self.graph[node]:
                        self.graph.edges[node, neighbour, 0]["role"] = "SR"
                        edge = (node, neighbour)
                        if edge not in Sr:
                            Sr.add(edge)
                            next_level.add(neighbour)
                this_level = next_level
                i -= 1
            SR.append(Sr)

        self.SR = SR
        self.C = C
        self.T = T


if __name__ == "__main__":
    generator_small_toronto = WaterPipeEnhancementGenerator(
        random_seed=2,
        path_prefix="easy",
        graph_instance="small_toronto",
        housing_area_rate=0.01,
        housing_area_size=3,
        critical_rate=0.01,
        water_source_rate=0.005,
    )
    generator_small_toronto(2, 1)
