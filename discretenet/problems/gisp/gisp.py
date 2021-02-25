from abc import ABC, abstractmethod
from pathlib import Path
import pickle
import random
from typing import Dict, Union, List
from joblib import Parallel, delayed
import numpy as np

from pyomo.core.expr.current import identify_variables, decompose_term
import pyomo.environ as pyo
import networkx as nx

from discretenet.problem import Problem
from discretenet.generator import Generator


class GISPProblem(Problem):

    def __init__(self, graph, E2, name):
        """
        Generate a concrete Pyomo model of the problem

        By the end of initialization, the model must be stored in ``self.model``.
        After initialization, the model is assumed to be immutable.
        """
        self.model = None
        self.name = name
        self.graph = graph
        self.E2 = list(E2)

        # create concrete pyomo model
        self.create_model()

    def create_model(self):
        model = pyo.ConcreteModel()
        #  initialize variables and cost functions
        coeffs = {}
        model.nodes = pyo.Set(initialize=list(self.graph.nodes()))
        model.x = pyo.Var(model.nodes, domain=pyo.Binary)
        for node in self.graph.nodes():
            # model.x[node] = pyo.Var(domain=pyo.Binary)
            coeffs[node] = self.graph.nodes[node]['revenue']

        model.E2 = pyo.Set(initialize=self.E2)
        model.y = pyo.Var(model.E2, domain=pyo.Binary)
        for node1, node2 in self.E2:
            # model.y[node1, node2] = pyo.Var(domain=pyo.Binary)
            coeffs[node1, node2] = -1 * self.graph[node1][node2]['cost']

        #  objective function
        model.objective = pyo.Objective(
            expr=pyo.quicksum(coeffs[node] * model.x[node] for node in self.graph.nodes()) + pyo.quicksum(
                coeffs[node1, node2] * model.y[node1, node2] for node1, node2 in self.E2), sense=pyo.maximize)

        #  add constraints
        model.constraints = pyo.ConstraintList()
        for node1, node2, edge in self.graph.edges(data=True):
            if (node1, node2) not in self.E2:
                model.constraints.add(model.x[node1] + model.x[node2] <= 1)
            else:
                model.constraints.add(model.x[node1] + model.x[node2] - model.y[node1, node2] <= 1)

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


class GISPGenerator(Generator):

    def __init__(self, random_seed: int = 42, path_prefix: Union[str, Path] = None, which_set='SET2',
                 graph_instance=None, min_n=100, max_n=100, er_prob=0.1, set_param=100.0, alpha=0.75):
        """
        Initialize the generator instance

        Concrete implementations should call `super().__init__(random_seed, path_prefix)`
        to ensure the random seed is properly initialized.

        In concrete generators, should take callables used to sample the parameters
        used to instantiate a :class:`discretenet.problem.Problem`. These callables
        should be set as instance attributes, and called in ``generate()``.

        For example, a callable returning a number from Normal(0, 1). The range of
        values returned by the callable is expected to change in different
        :class:`Generator` instances to set the difficulty (for example, in a
        harder set the variable may be sampled from Normal(0, 10)).

        :param random_seed: The random seed to use
        :param path_prefix:  Path prefix to pass to instance ``save()`` methods
            during batch generation. Must be set as a public instance attribute,
            to be changed by the user after generator instantiation if desired.
        """

        super().__init__(random_seed, path_prefix)
        random.seed(random_seed)  # should be moved into set_seed()
        self.problem = None
        self.which_set = which_set
        self.graph_instance = graph_instance
        self.min_n = min_n
        self.max_n = max_n
        self.er_prob = er_prob
        self.set_param = set_param
        self.alpha = alpha
        self.E2 = set()

        if self.graph_instance is None:
            # Generate random graph
            num_nodes = random.randint(self.min_n, self.max_n)
            self.graph = nx.erdos_renyi_graph(n=num_nodes, p=self.er_prob, seed=self.random_seed)
            self.name = ("er_n=%d_m=%d_p=%.2f_%s_setparam=%.2f_alpha=%.2f" % (
                num_nodes, nx.number_of_edges(self.graph), self.er_prob, self.which_set, self.set_param, self.alpha))
        else:
            self.dimacs_to_nx()
            instance_name = self.graph_instance.split('/')[1]
            self.name = ("%s_%s_%g_%g" % (instance_name, self.which_set, self.alpha, self.set_param))

    def dimacs_to_nx(self):
        g = nx.Graph()
        with open(self.graph_instance, 'r') as f:
            for line in f:
                arr = line.split()
                if line[0] == 'e':
                    g.add_edge(int(arr[1]), int(arr[2]))
        self.graph = g

    def problem(self) -> Problem:
        """
        A :class:`discretenet.problem.Problem` instance generated by this generator.
        Must be set when creating a concrete generator. ::
            class MyGenerator(Generator):
                problem = MyProblem
                ...
        """
        return self.problem

    def generate(self) -> Problem:
        """
        Generate and return a single :class:`discretenet.problem.Problem` instance

        Should sample for parameters from the defined parameter generator callables
        (from the ``__init__()`` method), instantiate an instance with those parameters,
        and return it.

        :return: An initialized concrete ``Problem`` instance
        """

        # Generate node revenues and edge costs
        self.generate_revs_costs()

        # Generate the set of removable edges
        self.generate_E2()

        # Construct Problem
        problem = GISPProblem(self.graph, self.E2, self.name+"_%d"%self.random_seed)

        # Update seed
        self.set_seed(self.random_seed+1)

        return problem

    def generate_revs_costs(self):
        if self.which_set == 'SET1':
            for node in self.graph.nodes():
                self.graph.nodes[node]['revenue'] = random.randint(1, 100)
            for u, v, edge in self.graph.edges(data=True):
                edge['cost'] = (self.graph.node[u]['revenue'] + self.graph.node[v]['revenue']) / float(self.set_param)
        elif self.which_set == 'SET2':
            for node in self.graph.nodes():
                self.graph.nodes[node]['revenue'] = float(self.set_param)
            for u, v, edge in self.graph.edges(data=True):
                edge['cost'] = 1.0

    def generate_E2(self):
        for edge in self.graph.edges():
            if random.random() <= self.alpha:
                self.E2.add(edge)




if __name__ == "__main__":

    # generator = GISPGenerator(which_set='SET2', min_n=10, max_n=100, er_prob=0.1, set_param=100.0, alpha=0.75)
    # generator(5, 1)

    generator_existing_graph = GISPGenerator(random_seed=1, which_set='SET2', graph_instance="DIMACS_subset_ascii/C125.9.clq", set_param=100.0, alpha=0.75)
    generator_existing_graph(2, 1)