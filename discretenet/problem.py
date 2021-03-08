from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from typing import Dict, Union, Type, TypeVar

from pyomo.core.expr.current import identify_variables, decompose_term
import pyomo.environ as pyo
import networkx as nx

T = TypeVar("T", bound="Problem")


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
        The model objective must be stored in ``self.model.OBJ``.

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

    @staticmethod
    def _get_variable_domain(var: pyo.Var) -> str:
        """
        Return the type of the given variable: continuous, integer, or binary.

        Variable domain does not refer to its bounds, as those can be accessed
        directly. Only default Pyomo Virtual Sets are checked against, variables
        should be bounded using constraints rather than adding new Sets.

        :param var: Pyomo Variable
        :return: str, one of "continuous", "integer", or "binary"
        """
        domain = var.domain

        # Virtual sets aren't hashable, so have to use lists instead of sets
        continuous_domains = [
            pyo.Any,
            pyo.Reals,
            pyo.PositiveReals,
            pyo.NonPositiveReals,
            pyo.NegativeReals,
            pyo.NonNegativeReals,
            pyo.PercentFraction,
            pyo.UnitInterval,
        ]
        integer_domains = [
            pyo.Integers,
            pyo.PositiveIntegers,
            pyo.NonPositiveIntegers,
            pyo.NegativeIntegers,
            pyo.NonNegativeIntegers,
        ]
        binary_domains = [pyo.Boolean, pyo.Binary]

        if domain in continuous_domains:
            return "continuous"
        elif domain in integer_domains:
            return "integer"
        elif domain in binary_domains:
            return "binary"

        raise ValueError("Unrecognized variable domain")

    def get_variable_constraint_graph(self) -> nx.Graph:
        """
        Construct a bipartite Variable Constraint Graph of the problem instance

        One node set is the set of problem variables, the other is the set of
        problem constraints. Nodes are connected by edges if a variable
        participates in a constraint. If the interaction is linear, the edge
        data will have a ``coeff`` attribute with the coefficient, otherwise
        there will be no ``coeff`` edge attribute. Variable nodes have a ``type``
        attribute of "variable", and constraints "constraints".

        Pyomo constraints can be modelled generically to have upper and lower
        bounds. For the VCG, all constraints are converted to be either upper
        bounded (constr <= bound), or equality bounded (constr == bound). The
        latter form could be split into two separate upper bounded constraints,
        but is left as-is instead. Constraint nodes have a ``kind`` attribute,
        one of ``leq`` or ``eq``, indicating whether the constraint is upper
        bounded or equality bounded. Constraint nodes also have a ``bound``
        attribute, with the numerical bound.

        Variable nodes have a ``domain`` attribute, one of "continuous", "integer",
        or "binary" depending on the domain (a Pyomo Virtual Set) of the variable.
        If variables need to be bounded, constraints should be added rather than
        creating a new Pyomo Set.

        If the objective function is linear, variables which participate in the
        objective function have their objective coefficient in the ``obj_coeff``
        node attribute. Since objectives can be minimized or maximized, the
        objective is converted to minimizing sense for the VCG, and objective
        coefficients negated as necessary.

        :return: A bipartite variable constraint graph
        """

        G = nx.Graph()

        for constr in self.model.component_objects(pyo.Constraint):
            constr_name = constr.getname()
            is_linear, var_list = decompose_term(constr.body)

            # All constraints in the VCG will either be <= or == bounded
            if constr.has_lb() and not constr.has_ub():  # constr >= b
                multiplier = -1
                bound = constr.lower()
                kind = "leq"
            elif not constr.has_lb() and constr.has_ub():  # constr <= b
                multiplier = 1
                bound = constr.upper()
                kind = "leq"
            elif constr.lower() == constr.upper():  # constr == b
                multiplier = 1
                bound = constr.upper()
                kind = "eq"
            else:
                # lb <= constr <= ub
                raise NotImplementedError(
                    "Double-bounded constraints are not supported"
                )

            bound *= multiplier

            G.add_node(constr_name, type="constraint", kind=kind)

            if not is_linear:
                # Pyomo currently doesn't support extracting coefficients
                # for nonlinear constraints
                for var in identify_variables(constr.body):
                    # Graph nodes are sets, so this is fine
                    G.add_node(
                        var.getname(),
                        type="variable",
                        domain=self._get_variable_domain(var),
                    )

                    # No coeff attribute for non-linear constraints
                    G.add_edge(constr_name, var.getname())
            else:
                for coeff, var in var_list:
                    if var is None:
                        # Constant term - subtract from bound
                        bound -= coeff * multiplier
                        continue

                    G.add_node(
                        var.getname(),
                        type="variable",
                        domain=self._get_variable_domain(var),
                    )

                    G.add_edge(constr_name, var.getname(), coeff=coeff)

            # Add attributes to the constraint node
            G.nodes[constr_name]["bound"] = bound

        # Tag variable nodes with their coefficient in the objective
        is_objective_linear, objective_var_list = decompose_term(self.model.OBJ.expr)
        objective_multiplier = 1 if self.model.OBJ.is_minimizing() else -1

        if is_objective_linear:
            for coeff, var in objective_var_list:
                if var is None:
                    # It's technically possible to add a constant term to the objective
                    continue

                # All variables should be in the VCG by now, but just in case
                G.add_node(
                    var.getname(),
                    type="variable",
                    domain=self._get_variable_domain(var),
                )
                G.nodes[var.getname()]["obj_coeff"] = coeff * objective_multiplier

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

        self.model.write(filename, io_options={"symbolic_solver_labels": True})

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
    def load(cls: Type[T], pkl_path: str) -> T:
        """
        Load a problem instance from a pickle file saved by ``dump()``

        :param pkl_path: Path to pickle file
        :return: Problem instance
        """

        with open(pkl_path, "rb") as fd:
            obj = pickle.load(fd)

        return obj
