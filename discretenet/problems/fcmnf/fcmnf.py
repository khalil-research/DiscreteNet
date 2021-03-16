import random
from pathlib import Path
from typing import Union
import pyomo.environ as pyo
import networkx as nx

from discretenet.problem import Problem
from discretenet.generator import Generator


class FCMNFProblem(Problem):
    is_linear = True

    def __init__(self, g, num_commodities, od_list, name):
        super().__init__()
        self.name = name

        model = pyo.ConcreteModel()

        # initialize variables and coefficients
        model.edges = pyo.Set(initialize=list(g.edges()))
        model.K = pyo.Set(initialize=range(num_commodities))

        def DK_init(m):
            return ((node1, node2, k) for node1, node2 in m.edges for k in m.K)

        model.DK = pyo.Set(dimen=3, initialize=DK_init)

        model.y = pyo.Var(model.edges, domain=pyo.Binary)
        model.x = pyo.Var(model.DK, domain=pyo.Binary)

        # objective function
        pyo.Objective(
            expr=pyo.quicksum(
                edge["length"] * model.x[node1, node2]
                for node1, node2, edge in g.edges(data=True)
            ),
            sense=pyo.minimize,
        )

        model.objective = pyo.Objective(
            expr=pyo.quicksum(
                edge["fixed_cost"] * model.y[node1, node2]
                for node1, node2, edge in g.edges(data=True)
            )
            + pyo.quicksum(
                pyo.quicksum(
                    edge["var_cost"] * od_list[k][2] * model.x[node1, node2, k]
                    for node1, node2, edge in g.edges(data=True)
                )
                for k in range(num_commodities)
            ),
            sense=pyo.minimize,
        )

        model.constraint1 = pyo.ConstraintList()
        # constraint 1
        for k in range(num_commodities):
            for node in g.nodes():
                rhs = 0.0
                if od_list[k][0] == node:
                    rhs = 1.0
                elif od_list[k][1] == node:
                    rhs = -1.0
                model.constraint1.add(
                    pyo.quicksum(
                        model.x[node, successor, k] for successor in g.successors(node)
                    )
                    - pyo.quicksum(
                        model.x[predecessor, node, k]
                        for predecessor in g.predecessors(node)
                    )
                    - rhs
                    == 0
                )

        model.constraint2 = pyo.ConstraintList()
        # constraint 2
        for node1, node2, edge in g.edges(data=True):
            model.constraint2.add(
                pyo.quicksum(
                    od_list[k][2] * model.x[node1, node2, k]
                    for k in range(num_commodities)
                )
                - edge["cap"] * model.y[node1, node2]
                <= 0
            )

        self.model = model

    def get_name(self):
        return self.name

    def get_parameters(self):
        # to be implemented
        return None


class FCMNFGenerator(Generator[FCMNFProblem]):
    def __init__(
        self,
        random_seed: int = 42,
        path_prefix: Union[str, Path] = None,
        min_n=100,
        max_n=100,
        er_prob=0.1,
        variable_costs_range_lower=11,
        variable_costs_range_upper=50,
        commodities_quantities_range_lower=10,
        commodities_quantities_range_upper=100,
        fixed_to_variable_ratio=100,
        edge_upper=200,
        num_commodities=200,
    ):
        """
        Initialize the Fixed-charge Multi-commodity Network Flow Problem
        generator instance following https://doi.org/10.1287/ijoc.1090.0348
        :param random_seed: The random seed to use
        :param path_prefix: Path prefix to pass to instance ``save()`` methods
            during batch generation. Must be set as a public instance attribute,
            to be changed by the user after generator instantiation if desired.
        :param min_n/max_n: min/max number of nodes the generated graph can have
        :param er_prob: the probability of an edge existing in the generated graph
        :param variable_costs_range_lower: edge variable cost lower bound
        :param variable_costs_range_upper: edge variable cost upper bound
        :param commodities_quantities_range_lower: quantity of each commodity lower bound
        :param commodities_quantities_range_upper: quantity of each commodity upper bound
        :param fixed_to_variable_ratio:
            the ratio of an edge's fixed cost to its variable cost
        :param edge_upper: number of commodities that can fit on an edge
        :param num_commodities: number of commodities
        """
        super().__init__(random_seed, path_prefix)

        self.variable_costs_range_lower = variable_costs_range_lower
        self.variable_costs_range_upper = variable_costs_range_upper
        self.commodities_quantities_range_lower = commodities_quantities_range_lower
        self.commodities_quantities_range_upper = commodities_quantities_range_upper
        self.fixed_to_variable_ratio = fixed_to_variable_ratio
        self.fixed_costs_range = (
            fixed_to_variable_ratio * variable_costs_range_lower,
            fixed_to_variable_ratio * variable_costs_range_upper,
        )
        self.edge_upper = edge_upper
        self.num_commodities = num_commodities
        self.od_list = None
        self.graph = None

        # Generate random graph
        num_nodes = random.randint(min_n, max_n)
        self.base_graph = nx.erdos_renyi_graph(
            n=num_nodes, p=er_prob, seed=self.random_seed, directed=True
        )
        self.name = "er_n=%d_m=%d_p=%.2f" % (
            num_nodes,
            nx.number_of_edges(self.base_graph),
            er_prob,
        )

    def generate(self):
        self.graph = self.base_graph.copy()

        self.generate_random_vars()
        problem = FCMNFProblem(
            self.graph,
            self.num_commodities,
            self.od_list,
            self.name + "_%d" % self.random_seed,
        )
        return problem

    def generate_random_vars(self):

        for u, v, edge in self.graph.edges(data=True):
            edge["var_cost"] = random.randint(
                self.variable_costs_range_lower, self.variable_costs_range_upper
            )
            edge["fixed_cost"] = random.randint(
                self.fixed_costs_range[0], self.fixed_costs_range[1]
            )
            edge["cap"] = random.randint(1, self.edge_upper) * random.randint(
                self.commodities_quantities_range_lower,
                self.commodities_quantities_range_upper,
            )
        # generate o-d pairs + quantity
        od_list = []
        n = nx.number_of_nodes(self.graph)
        while len(od_list) < self.num_commodities:
            i, j = random.randint(0, n - 1), random.randint(0, n - 1)
            if i != j and nx.has_path(self.graph, i, j):
                od_list += [
                    (
                        i,
                        j,
                        random.randint(
                            self.commodities_quantities_range_lower,
                            self.commodities_quantities_range_upper,
                        ),
                    )
                ]

        self.od_list = od_list


if __name__ == "__main__":
    generator = FCMNFGenerator(
        random_seed=0,
        path_prefix="easy",
        min_n=100,
        max_n=100,
        er_prob=0.1,
        variable_costs_range_lower=11,
        variable_costs_range_upper=50,
        commodities_quantities_range_lower=10,
        commodities_quantities_range_upper=100,
        fixed_to_variable_ratio=100,
        edge_upper=200,
        num_commodities=200,
    )
    generator(1, 1)
