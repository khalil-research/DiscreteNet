from pathlib import Path
import random
from typing import Union, Optional

import pyomo.environ as pyo
import networkx as nx

try:
    from importlib.resources import open_text
except ImportError:
    from importlib_resources import open_text

from discretenet.problem import Problem
from discretenet.generator import Generator


class GISPProblem(Problem):
    is_linear = True

    def __init__(self, graph: nx.graph, E2: list, name: str):
        """
        Construct a concrete Pyomo model for a Generalized Independent Set Problem

        :param graph: undirected networkx graph, each node is associated with a revenue
            and each edge is associated with a cost
        :param E2: list of removable edges
        :param name: name of the instance
        """
        super().__init__()
        self.graph = graph
        self.E2 = E2
        self.name = name

        # create concrete pyomo model
        self.create_model()

    def create_model(self):
        model = pyo.ConcreteModel()
        #  initialize variables and cost functions
        coeffs = {}
        model.nodes = pyo.Set(initialize=list(self.graph.nodes()))
        model.x = pyo.Var(model.nodes, domain=pyo.Binary)
        for node in self.graph.nodes():
            coeffs[node] = self.graph.nodes[node]["revenue"]

        model.E2 = pyo.Set(initialize=self.E2)
        model.y = pyo.Var(model.E2, domain=pyo.Binary)
        for node1, node2 in self.E2:
            coeffs[node1, node2] = -1 * self.graph[node1][node2]["cost"]

        #  objective function
        model.objective = pyo.Objective(
            expr=pyo.quicksum(
                coeffs[node] * model.x[node] for node in self.graph.nodes()
            )
            + pyo.quicksum(
                coeffs[node1, node2] * model.y[node1, node2] for node1, node2 in self.E2
            ),
            sense=pyo.maximize,
        )

        #  add constraints
        model.constraints = pyo.ConstraintList()
        for node1, node2, edge in self.graph.edges(data=True):
            if (node1, node2) not in self.E2:
                model.constraints.add(model.x[node1] + model.x[node2] <= 1)
            else:
                model.constraints.add(
                    model.x[node1] + model.x[node2] - model.y[node1, node2] <= 1
                )

        self.model = model

    def get_name(self) -> str:
        return self.name

    def get_parameters(self):
        return {
            "graph": self.graph,
            "E2": self.E2,
            "name": self.name,
        }


class GISPGenerator(Generator[GISPProblem]):
    def __init__(
        self,
        random_seed: int = 42,
        path_prefix: Union[str, Path] = None,
        which_set: str = "SET2",
        graph_instance: Optional[str] = None,
        min_n: Optional[int] = 100,
        max_n: Optional[int] = 100,
        er_prob: Optional[float] = 0.1,
        set_param: float = 100.0,
        alpha: float = 0.75,
    ):
        """
        Initialize the Generalized Independent Set Problem generator instance
        following https://doi.org/10.1016/j.ejor.2016.11.050

        :param random_seed: The random seed to use
        :param path_prefix:  Path prefix to pass to instance ``save()`` methods
            during batch generation. Must be set as a public instance attribute,
            to be changed by the user after generator instantiation if desired.
        :param which_set: can take values “SET1” or “SET2". These are two types
            of models for assigning costs/revenues to edges/nodes in the generated
            instances.
        :param graph_instance: path to .clq file of DIMACS graph or do not provide
            it and a random graph will be generated.
        :param min_n/max_n: min/max number of nodes a randomly generated
            graph can have, if not using a DIMACS graph
        :param er_prob: the probability of an edge existing in the randomly
            generated graph, if not using a DIMACS graph.
        :param set_param: a value that affects the node revenues and/or edge costs
        :param alpha: probability of an edge being in the set of edges which are
            legal to remove (at a cost)
        """

        super().__init__(random_seed, path_prefix)
        self.which_set = which_set
        self.graph_instance = graph_instance
        self.min_n = min_n
        self.max_n = max_n
        self.er_prob = er_prob
        self.set_param = set_param
        self.alpha = alpha

        if self.graph_instance is None:
            # Generate random graph
            num_nodes = random.randint(self.min_n, self.max_n)
            self.base_graph = nx.erdos_renyi_graph(
                n=num_nodes, p=self.er_prob, seed=self.random_seed
            )
            self.name = "er_n=%d_m=%d_p=%.2f_%s_setparam=%.2f_alpha=%.2f" % (
                num_nodes,
                nx.number_of_edges(self.base_graph),
                self.er_prob,
                self.which_set,
                self.set_param,
                self.alpha,
            )
        else:
            self.dimacs_to_nx()
            self.name = "%s_%s_%g_%g" % (
                self.graph_instance,
                self.which_set,
                self.alpha,
                self.set_param,
            )

    def dimacs_to_nx(self):
        g = nx.Graph()
        with open_text("discretenet.problems.gisp.graphs", self.graph_instance) as f:
            for line in f:
                arr = line.split()
                if line[0] == "e":
                    g.add_edge(int(arr[1]), int(arr[2]))
        self.base_graph = g

    def generate(self):
        # create a copy of the graph to randomize
        graph = self.base_graph.copy()

        # Generate node revenues and edge costs
        graph = self.generate_revs_costs(graph)

        # Generate the set of removable edges
        E2 = self.generate_E2(graph)

        # Construct Problem
        problem = GISPProblem(graph, E2, self.name + "_%d" % self.random_seed)

        return problem

    def generate_revs_costs(self, graph):
        if self.which_set == "SET1":
            for node in graph.nodes():
                graph.nodes[node]["revenue"] = random.randint(1, 100)
            for u, v, edge in graph.edges(data=True):
                edge["cost"] = (
                    graph.nodes[u]["revenue"] + graph.nodes[v]["revenue"]
                ) / float(self.set_param)
        elif self.which_set == "SET2":
            for node in graph.nodes():
                graph.nodes[node]["revenue"] = float(self.set_param)
            for u, v, edge in graph.edges(data=True):
                edge["cost"] = 1.0
        return graph

    def generate_E2(self, graph):
        E2 = []
        for edge in graph.edges():
            if random.random() <= self.alpha:
                E2.append(edge)
        return E2


if __name__ == "__main__":
    generator_existing_graph = GISPGenerator(
        random_seed=1,
        which_set="SET2",
        graph_instance="C125.9.clq",
        set_param=100.0,
        alpha=0.60,
    )
    generator_existing_graph(1, 1)
