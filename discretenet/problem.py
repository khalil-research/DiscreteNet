from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from typing import Dict, Union

from pyomo.core.expr.current import identify_variables, decompose_term
import pyomo.environ as pyo
import networkx as nx


class Problem(ABC):
    model: pyo.ConcreteModel = None

    @property
    @abstractmethod
    def is_linear(self) -> bool:
        """
        Boolean indicating if the model objective and constraints are linear.

        Must be set when creating a concrete problem class. ::

            class MyProblem(Problem):
                is_linear = True

                ...
        """
        pass

    @abstractmethod
    def __init__(self):
        """
        Generate a concrete Pyomo model of the problem

        By the end of initialization, the model must be stored in ``self.model``.
        After initialization, the model is assumed to be immutable.
        """

        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return a string name for the model instance, based on problem parameters

        Use for saving parts of the model, but most not include path or extension.

        :return: Model instance name
        """

        pass

    def get_features(self) -> Dict[str, float]:
        pass

    def get_variable_constraint_graph(self) -> nx.Graph:
        """
        Construct a bipartite Variable Constraint Graph of the problem instance

        One node set is the set of problem variables, the other is the set of
        problem constraints. Nodes are connected by edges if a variable
        participates in a constraint. If the interaction is linear, the edge
        data will have a ``coeff`` attribute with the coefficient, otherwise
        there will be no ``coeff`` edge attribute.

        :return: A bipartite variable constraint graph
        """
        constraint_to_vars = {}
        variables = set()

        for c in self.model.component_objects(pyo.Constraint):
            constr_name = c.get_name()
            constraint_to_vars[constr_name] = []

            is_linear, var_list = decompose_term(c.body)

            if not is_linear:
                for v in identify_variables(c.body):
                    constraint_to_vars[constr_name].append((v.getname(), None))
                    variables.add(v.getname())
            else:
                for coeff, v in var_list:
                    if v is None:
                        # Constant term
                        continue

                    constraint_to_vars[constr_name].append((v.getname(), coeff))
                    variables.add(v.getname())

        G = nx.Graph()

        for v in variables:
            G.add_node(v)

        for constr_name, _vars in constraint_to_vars.items():
            G.add_node(constr_name)

            for v, coeff in _vars:
                edge_args = {}

                if coeff:
                    edge_args["coeff"] = coeff

                G.add_edge(c, v, **edge_args)

        return G

    def __build_full_path(self, extension: str, path_prefix: Union[str, Path]):
        filename = self.get_name() + extension

        if path_prefix:
            if isinstance(path_prefix, str):
                path_prefix = Path(path_prefix).resolve()

            filename = str(path_prefix.joinpath(filename))

        return filename

    def save(self, path_prefix: Union[str, Path] = None) -> None:
        if self.is_linear:
            filename = self.__build_full_path(".mps", path_prefix)
        else:
            filename = self.__build_full_path(".gms", path_prefix)

        self.model.write(filename)

    def save_graph(self, path_prefix: str = None) -> None:
        pass

    def dump(self, path_prefix: Union[str, Path] = None) -> None:
        """
        Dump the entire :class:`Problem` instance as a pickle object

        Filename is given by ``self.get_name()``.

        :param path_prefix: Folder to save to, default to current directory
        """

        filename = self.__build_full_path(".pkl", path_prefix)

        with open(filename, "wb+") as fd:
            pickle.dump(self, fd)

    @classmethod
    def load(cls, pkl_path: str) -> "Problem":
        """
        Load a problem instance from a pickle file saved by ``dump()``

        :param pkl_path: Path to pickle file
        :return: Problem instance
        """

        with open(pkl_path, "rb") as fd:
            obj = pickle.load(fd)

        return obj
