from abc import ABC, abstractmethod
from functools import lru_cache
import json
from pathlib import Path
import pickle
from typing import Any, Dict, Generator, Union, Type, TypeVar

from pyomo.core.base.constraint import IndexedConstraint, _GeneralConstraintData
from pyomo.core.expr.current import identify_variables, decompose_term
import pyomo.environ as pyo
import networkx as nx
import numpy as np
from scipy.stats import variation

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
        The model objective must be stored in ``self.model.objective``.

        There should be no randomness in model initialization - given the
        same parameters, an identical model should be created.

        After initialization, the model is assumed to be immutable.
        """

        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return a string name for the model instance, based on problem parameters

        Use for saving parts of the model, but must not include path or extension.

        :return: Model instance name
        """

        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Return a dictionary of parameters used to construct the instance

        It should be possible to re-initialize an identical instance by passing
        these parameters to the :meth:`~__init__` method.

        :return: A dictionary of problem parameters
        """
        pass

    @lru_cache(maxsize=None)
    def get_features(self) -> Dict[str, float]:
        """
        Return a dictionary of computed features for the problem instance

        These features are computed using the Variable Constraint Graph from
        :meth:`~get_variable_constraint_graph`. Note
        that:
            - The VCG considers two kinds of constraints: inequality (<=) and
              equality (==).
            - All inequality constraints in the problem definition are converted
              to <= form, even if defined in >= form. This means that coefficients
              for >= constraints will have their sign flipped, which will affect
              the numbers returned here.
            - Equality constraints are left as-is, and their coefficients are
              only counted a single time rather than converting the problem to
              standard form.
            - Coefficients are not present for non-linear constraints, hence
              many of these statistics may not make sense for non-linear problems.

        Description of features:
        1.  ``num_variables``: Number of variables
        2.  ``num_constraints``: Number of constraints. Inequality and equality
            constraints are each counted once. If interested in this number being
            in standard form, use
            ``num_inequality_constraints + 2 * num_equality_constraints``.
        3.  ``num_inequality_constraints``: Number of inequality constraints
        4.  ``num_equality_constraints``: Number of equality constraints
        5.  ``num_linear_constraints``: Number of linear constraints
        6.  ``num_nonlinear_constraints``: Number of nonlinear constraints
        7.  ``num_vcg_edges``: Number of edges in the Variable Constraint Graph
            (total number of times variables participate in constraints)
        8.  ``num_linear_vcg_edges``: Number of edges in the VCG where the constraint
            is linear
        9.  ``num_nonlinear_vcg_edges``: Number of edges in the VCG where the
            constraint is nonlinear
        10.  ``num_binary_variables``: Number of binary variables
        11.  ``num_integer_variable``: Number of integer variables
        12. ``num_continuous_variables``: Number of continuous variables
        13. ``num_non_continuous_variables``: Number of non-continuous (binary and
            integer) variables
        14. ``fraction_binary_variables``: Fraction of variables that are binary
        15. ``fraction_integer_variables``: Fraction of variables that are integer
        16. ``fraction_continuous_variables``: Fraction of variables that are continuous
        17. ``fraction_non_continuous_variables``: Fraction of variables that are
            not continuous
        18. ``vcg_variable_node_degree_mean``: Mean node degree for variable nodes
        19. ``vcg_variable_node_degree_median``: Median node degree for variable nodes
        20. ``vcg_variable_node_degree_cv``: Coefficient of variation of node degree for
            variable nodes
        21. ``vcg_variable_node_degree_p90p10``: Percentile ration p90/p10 of node degree
            for variable nodes
        22. ``vcg_continuous_variable_node_degree_mean``: Mean node degree for continuous
            variable nodes
        23. ``vcg_continuous_variable_node_degree_median``: Median node degree for
            continuous variable nodes
        24. ``vcg_continuous_variable_node_degree_cv``: Coefficient of variation of node
            degree for continuous variable nodes
        25. ``vcg_continuous_variable_node_degree_p90p10``: Percentile ration p90/p10
            of node degree for continuous variable nodes
        26. ``vcg_non_continuous_variable_node_degree_mean``: Mean node degree for
            non-continuous variable nodes
        27. ``vcg_non_continuous_variable_node_degree_median``: Median node degree for
            non-continuous variable nodes
        28. ``vcg_non_continuous_variable_node_degree_cv``: Coefficient of variation of
            node degree for non-continuous variable nodes
        29. ``vcg_non_continuous_variable_node_degree_p90p10``: Percentile ration p90/p10
            of node degree for non-continuous variable nodes
        30. ``vcg_constraint_node_degree_mean``: Mean node degree for constraint nodes
            with respect to all variables
        31. ``vcg_constraint_node_degree_median``: Median node degree for constraint nodes
            with respect to all variables
        32. ``vcg_constraint_node_degree_cv``: Coefficient of variation of node degree for
            constraint nodes with respect to all variables
        33. ``vcg_constraint_node_degree_p90p10``: Percentile ration p90/p10 of node
            degree for constraint nodes with respect to all variables
        34. ``vcg_continuous_constraint_node_degree_mean``: Mean node degree for
            constraint nodes with respect to continuous variables
        35. ``vcg_continuous_constraint_node_degree_median``: Median node degree for
            constraint nodes with respect to continuous variables
        36. ``vcg_continuous_constraint_node_degree_cv``: Coefficient of variation of
            node degree for constraint nodes with respect to continuous variables
        37. ``vcg_continuous_constraint_node_degree_p90p10``: Percentile ration p90/p10
            of node degree for constraint nodes with respect to continuous variables
        38. ``vcg_non_continuous_constraint_node_degree_mean``: Mean node degree for
            constraint nodes with respect to non-continuous variables
        39. ``vcg_non_continuous_constraint_node_degree_median``: Median node degree for
            constraint nodes with respect to non-continuous variables
        40. ``vcg_non_continuous_constraint_node_degree_cv``:  Coefficient of variation of
            node degree for constraint nodes with respect to non-continuous variables
        41. ``vcg_non_continuous_constraint_node_degree_p90p10``: Percentile ration
            p90/p10 of node degree for constraint nodes with respect to non-continuous
            variables
        42. ``variable_coefficient_sum_mean``: Mean of a vector where the ith element
            is the sum of coefficients for the ith variable over all constraints, with
            respect to all variables
        43. ``variable_coefficient_sum_cv``: Coefficient of variation of a vector where
            the ith element is the sum of coefficients for the ith variable over all
            constraints, with respect to all variables
        44. ``continuous_variable_coefficient_sum_mean``: As above, with respect to
            continuous variables
        45. ``continuous_variable_coefficient_sum_cv``: As above, with respect to
            continuous variables
        46. ``non_continuous_variable_coefficient_sum_mean``: As above, with respect to
            non-continuous variables
        47. ``non_continuous_variable_coefficient_sum_cv``: As above, with respect to
            non-continuous variables
        48. ``constraint_coefficient_sum_mean``: Mean of a vector where the ith
            element is the sum of coefficients within the ith constraint, with respect to
            all variables
        49. ``constraint_coefficient_sum_cv``: Coefficient of variation of a vector where
            the ith element is the sum of coefficients within the ith constraint, with
            respect to all variables
        50. ``continuous_constraint_coefficient_sum_mean``: As above, with respect to
            continuous variables
        51. ``continuous_constraint_coefficient_sum_cv``: As above, with respect to
            continuous variables
        52. ``non_continuous_constraint_coefficient_sum_mean``: As above, with respect to
            non-continuous variables
        53. ``non_continuous_constraint_coefficient_sum_cv``: As above, with respect to
            non-continuous variables
        54. ``normalized_constraint_coefficient_mean``: Mean of normalized constraint
            coefficients, where each coefficient is normalized by the constraint’s bound
        55. ``normalized_constraint_coefficient_cv``: Coefficient of variation of
            normalized constraint coefficients, where each coefficient is normalized by
            the constraint’s bound
        56. ``continuous_normalized_constraint_coefficient_mean``: As above, with respect
            to continuous variables
        57. ``continuous_normalized_constraint_coefficient_cv``: As above, with respect to
            continuous variables
        58. ``non_continuous_normalized_constraint_coefficient_mean``: As above, with
            respect to non-continuous variables
        59. ``non_continuous_normalized_constraint_coefficient_cv``:  As above, with
            respect to non-continuous variables
        60. ``abs_objective_function_coefficients_mean``: Mean of absolute objective
            function coefficients
        61. ``abs_objective_function_coefficients_stddev``: Standard deviation of
            absolute objective function coefficients
        62. ``abs_objective_function_continuous_coefficients_mean``: As above, with
            respect to continuous variables in the objective function
        63. ``abs_objective_function_continuous_coefficients_stddev``: As above, with
            respect to continuous variables in the objective function
        64. ``abs_objective_function_non_continuous_coefficients_mean``: As above, with
            respect to non-continuous variables in the objective function
        65. ``abs_objective_function_non_continuous_coefficients_stddev``: As above, with
            respect to non-continuous variables in the objective function
        66. ``normalized_abs_objective_function_coefficients_mean``: Mean of absolute
            objective function coefficients, normalized by number of constraints each
            variable participates in
        67. ``normalized_abs_objective_function_coefficients_stddev``: Standard deviation
            of absolute objective function coefficients, normalized by number of
            constraints each variable participates in
        68. ``normalized_abs_objective_function_continuous_coefficients_mean``: As above,
            with respect to continuous variables in the objective function
        69. ``normalized_abs_objective_function_continuous_coefficients_stddev``: As
            above, with respect to continuous variables in the objective function
        70. ``normalized_abs_objective_function_non_continuous_coefficients_mean``: As
            above, with respect to non-continuous variables in the objective function
        71. ``normalized_abs_objective_function_non_continuous_coefficients_stddev``: As
            above, with respect to non-continuous variables in the objective function
        72. ``sqrt_normalized_abs_objective_function_coefficients_mean``: Mean of absolute
            objective function coefficients, normalized by square root of the number of
            constraints each variable participates in
        73. ``sqrt_normalized_abs_objective_function_coefficients_stddev``: Standard
            deviation of absolute objective function coefficients, normalized by square
            root of the number of constraints each variable participates in
        74. ``sqrt_normalized_abs_objective_function_continuous_coefficients_mean``: As
            above, with respect to continuous variables in the objective function
        75. ``sqrt_normalized_abs_objective_function_continuous_coefficients_stddev``:
            As above, with respect to continuous variables in the objective function
        76. ``sqrt_normalized_abs_objective_function_non_continuous_coefficients_mean``:
            As above, with respect to non-continuous variables in the objective function
        77. ``sqrt_normalized_abs_objective_function_non_continuous_coefficients_stddev``:
            As above, with respect to non-continuous variables in the objective function
        78. ``leq_constraint_bounds_mean``: Mean constraint bound for <= constraints
        79. ``leq_constraint_bounds_stddev``: Standard deviation of constraint bound
            for <= constraints
        80. ``eq_constraint_bounds_mean``: Mean constraint bound for == constraints
        81. ``eq_constraint_bounds_stddev``: Standard deviation of constraint bound
            for == constraints

        :return: A dictionary of computed features
        """
        features: Dict[str, float] = {}
        vcg = self.get_variable_constraint_graph()

        variable_nodes = [
            (name, data)
            for name, data in vcg.nodes(data=True)
            if data["type"] == "variable"
        ]

        continuous_variable_nodes = [
            (name, data)
            for name, data in variable_nodes
            if data["domain"] == "continuous"
        ]
        non_continuous_variable_nodes = [
            (name, data)
            for name, data in variable_nodes
            if data["domain"] != "continuous"
        ]

        constraint_nodes = [
            (name, data)
            for name, data in vcg.nodes(data=True)
            if data["type"] == "constraint"
        ]

        features["num_variables"] = len(variable_nodes)
        features["num_constraints"] = len(constraint_nodes)
        features["num_inequality_constraints"] = len(
            [name for name, data in constraint_nodes if data["kind"] == "leq"]
        )
        features["num_equality_constraints"] = len(
            [name for name, data in constraint_nodes if data["kind"] == "eq"]
        )
        features["num_linear_constraints"] = len(
            [name for name, data in constraint_nodes if data["is_linear"]]
        )
        features["num_nonlinear_constraints"] = len(
            [name for name, data in constraint_nodes if not data["is_linear"]]
        )

        features["num_vcg_edges"] = len(vcg.edges)
        features["num_linear_vcg_edges"] = len(
            [1 for n1, n2, data in vcg.edges(data=True) if data["is_linear"]]
        )
        features["num_nonlinear_vcg_edges"] = len(
            [1 for n1, n2, data in vcg.edges(data=True) if not data["is_linear"]]
        )
        features["num_binary_variables"] = len(
            [
                name
                for name, data in vcg.nodes(data=True)
                if (data["type"] == "variable" and data["domain"] == "binary")
            ]
        )
        features["num_integer_variables"] = len(
            [
                name
                for name, data in vcg.nodes(data=True)
                if (data["type"] == "variable" and data["domain"] == "integer")
            ]
        )
        features["num_continuous_variables"] = len(continuous_variable_nodes)
        features["num_non_continuous_variables"] = (
            features["num_binary_variables"] + features["num_integer_variables"]
        )

        features["fraction_binary_variables"] = (
            features["num_binary_variables"] / features["num_variables"]
        )
        features["fraction_integer_variables"] = (
            features["num_integer_variables"] / features["num_variables"]
        )
        features["fraction_continuous_variables"] = (
            features["num_continuous_variables"] / features["num_variables"]
        )
        features["fraction_non_continuous_variables"] = (
            features["num_non_continuous_variables"] / features["num_variables"]
        )

        # VCG Variable Node Degree Statistics - computed with respect to
        # all, only continuous, and only non-continuous variables
        for variable_type, nodes in [
            ("all", variable_nodes),
            ("continuous", continuous_variable_nodes),
            ("non_continuous", non_continuous_variable_nodes),
        ]:
            degrees = []
            for node_name, _ in nodes:
                degrees.append(vcg.degree[node_name])

            if variable_type == "all":
                feature_prefix = "vcg_variable_node_degree"
            else:
                feature_prefix = f"vcg_{variable_type}_variable_node_degree"

            if degrees:
                features[f"{feature_prefix}_mean"] = np.mean(degrees)
                features[f"{feature_prefix}_median"] = np.median(degrees)
                features[f"{feature_prefix}_cv"] = variation(degrees)
                features[f"{feature_prefix}_p90p10"] = np.percentile(
                    degrees, 90
                ) / np.percentile(degrees, 10)
            else:
                features[f"{feature_prefix}_mean"] = 0.0
                features[f"{feature_prefix}_median"] = 0.0
                features[f"{feature_prefix}_cv"] = 0.0
                features[f"{feature_prefix}_p90p10"] = 0.0

        # VCG Constraint Node Degree Statistics - computed with respect to
        # all, only continuous, and only non-continuous variables. This is
        # a bit trickier than above, since we have to actually copy the graph
        # and remove nodes. Note that equality constraints are single nodes,
        # the problem is not in standard form.
        for variable_type, nodes_to_remove in [
            ("all", []),
            ("continuous", non_continuous_variable_nodes),
            ("non_continuous", continuous_variable_nodes),
        ]:
            vcg_copy = vcg.copy()
            nodes = [node_name for node_name, _ in nodes_to_remove]
            vcg_copy.remove_nodes_from(nodes)

            remaining_variable_nodes = [
                node_name
                for node_name, data in vcg_copy.nodes(data=True)
                if data["type"] == "variable"
            ]

            degrees = []
            for node_name in remaining_variable_nodes:
                degrees.append(vcg_copy.degree[node_name])

            if variable_type == "all":
                feature_prefix = "vcg_constraint_node_degree"
            else:
                feature_prefix = f"vcg_{variable_type}_constraint_node_degree"

            if degrees:
                features[f"{feature_prefix}_mean"] = np.mean(degrees)
                features[f"{feature_prefix}_median"] = np.median(degrees)
                features[f"{feature_prefix}_cv"] = variation(degrees)
                features[f"{feature_prefix}_p90p10"] = np.percentile(
                    degrees, 90
                ) / np.percentile(degrees, 10)
            else:
                features[f"{feature_prefix}_mean"] = 0.0
                features[f"{feature_prefix}_median"] = 0.0
                features[f"{feature_prefix}_cv"] = 0.0
                features[f"{feature_prefix}_p90p10"] = 0.0

        # Variable coefficient statistics
        for variable_type, nodes in [
            ("all", variable_nodes),
            ("continuous", continuous_variable_nodes),
            ("non_continuous", non_continuous_variable_nodes),
        ]:
            coefficient_sums = []
            for node_name, _ in nodes:
                coeff_sum = 0.0
                for var_node, constr_node, data in vcg.edges([node_name], data=True):
                    if data["is_linear"]:
                        coeff_sum += data["coeff"]
                coefficient_sums.append(coeff_sum)

            if variable_type == "all":
                feature_prefix = "variable_coefficient_sum"
            else:
                feature_prefix = f"{variable_type}_variable_coefficient_sum"

            if coefficient_sums:
                features[f"{feature_prefix}_mean"] = np.mean(coefficient_sums)
                features[f"{feature_prefix}_cv"] = variation(coefficient_sums)
            else:
                features[f"{feature_prefix}_mean"] = 0.0
                features[f"{feature_prefix}_cv"] = 0.0

        # Constraint coefficient statistics - only for linear constraints
        for variable_type, nodes_to_remove in [
            ("all", []),
            ("continuous", non_continuous_variable_nodes),
            ("non_continuous", continuous_variable_nodes),
        ]:
            vcg_copy = vcg.copy()
            nodes = [node_name for node_name, _ in nodes_to_remove]
            vcg_copy.remove_nodes_from(nodes)

            coefficient_sums = []
            for node_name, node_data in vcg.nodes(data=True):
                if node_data["type"] != "constraint":
                    continue

                coeff_sum = 0.0
                for constr_node, var_node, data in vcg.edges([node_name], data=True):
                    if data["is_linear"]:
                        coeff_sum += data["coeff"]

                coefficient_sums.append(coeff_sum)

            if variable_type == "all":
                feature_prefix = "constraint_coefficient_sum"
            else:
                feature_prefix = f"{variable_type}_constraint_coefficient_sum"

            if coefficient_sums:
                features[f"{feature_prefix}_mean"] = np.mean(coefficient_sums)
                features[f"{feature_prefix}_cv"] = variation(coefficient_sums)
            else:
                features[f"{feature_prefix}_mean"] = 0.0
                features[f"{feature_prefix}_cv"] = 0.0

        # Distribution of normalized constraint variable coefficients
        for variable_type, nodes in [
            ("all", variable_nodes),
            ("continuous", continuous_variable_nodes),
            ("non_continuous", non_continuous_variable_nodes),
        ]:
            if variable_type == "all":
                feature_prefix = "normalized_constraint_coefficient"
            else:
                feature_prefix = f"{variable_type}_normalized_constraint_coefficient"

            normalized_coeffs = []

            for node_name, _ in nodes:
                for var_node_name, constraint_node_name, data in vcg.edges(
                    [node_name], data=True
                ):
                    if not data["is_linear"]:
                        continue

                    constraint_bound = vcg.nodes[constraint_node_name]["bound"]

                    if constraint_bound == 0:
                        continue

                    normalized_coeffs.append(data["coeff"] / constraint_bound)

            if normalized_coeffs:
                features[f"{feature_prefix}_mean"] = np.mean(normalized_coeffs)
                features[f"{feature_prefix}_cv"] = variation(normalized_coeffs)
            else:
                features[f"{feature_prefix}_mean"] = 0.0
                features[f"{feature_prefix}_cv"] = 0.0

        # Objective function features
        is_objective_linear, objective_var_list = decompose_term(
            self.model.objective.expr
        )

        objective_coefficients = [
            (abs(coeff), var) for coeff, var in objective_var_list if var is not None
        ]
        continuous_objective_coefficients = [
            (abs(coeff), var)
            for coeff, var in objective_var_list
            if var is not None and self._get_variable_domain(var) == "continuous"
        ]
        non_continuous_objective_coefficients = [
            (abs(coeff), var)
            for coeff, var in objective_var_list
            if var is not None and self._get_variable_domain(var) != "continuous"
        ]

        # Absolute objective function coefficients
        for variable_type, coeff_data in [
            ("all", objective_coefficients),
            ("continuous", continuous_objective_coefficients),
            ("non_continuous", non_continuous_objective_coefficients),
        ]:
            if variable_type == "all":
                feature_prefix = "abs_objective_function_coefficients"
            else:
                feature_prefix = f"abs_objective_function_{variable_type}_coefficients"

            coeffs = [coeff for coeff, var in coeff_data]

            if coeffs:
                features[f"{feature_prefix}_mean"] = float(np.mean(coeffs))
                features[f"{feature_prefix}_stddev"] = float(np.std(coeffs))
            else:
                features[f"{feature_prefix}_mean"] = 0.0
                features[f"{feature_prefix}_stddev"] = 0.0

        # Normalized absolute objective function coefficients
        # Normalized by the number of constraints each variable participates in
        for variable_type, coeff_data in [
            ("all", objective_coefficients),
            ("continuous", continuous_objective_coefficients),
            ("non_continuous", non_continuous_objective_coefficients),
        ]:
            if variable_type == "all":
                feature_prefix = "normalized_abs_objective_function_coefficients"
            else:
                feature_prefix = (
                    f"normalized_abs_objective_function_{variable_type}_coefficients"
                )

            coeffs = []
            for coeff, var in coeff_data:
                num_constraints = len(vcg.edges([var.getname()]))
                if num_constraints == 0:
                    continue

                coeffs.append(coeff / num_constraints)

            if coeffs:
                features[f"{feature_prefix}_mean"] = float(np.mean(coeffs))
                features[f"{feature_prefix}_stddev"] = float(np.std(coeffs))
            else:
                features[f"{feature_prefix}_mean"] = 0.0
                features[f"{feature_prefix}_stddev"] = 0.0

        # Square root normalized absolute objective function coefficients
        for variable_type, coeff_data in [
            ("all", objective_coefficients),
            ("continuous", continuous_objective_coefficients),
            ("non_continuous", non_continuous_objective_coefficients),
        ]:
            if variable_type == "all":
                feature_prefix = "sqrt_normalized_abs_objective_function_coefficients"
            else:
                feature_prefix = (
                    f"sqrt_normalized_abs_objective_function_{variable_type}"
                    "_coefficients"
                )

            coeffs = []
            for coeff, var in coeff_data:
                num_constraints = len(vcg.edges([var.getname()]))
                if num_constraints == 0:
                    continue

                coeffs.append(coeff / np.sqrt(num_constraints))

            if coeffs:
                features[f"{feature_prefix}_mean"] = float(np.mean(coeffs))
                features[f"{feature_prefix}_stddev"] = float(np.std(coeffs))
            else:
                features[f"{feature_prefix}_mean"] = 0.0
                features[f"{feature_prefix}_stddev"] = 0.0

        # Constraint bound features
        leq_constraint_bounds = [
            data["bound"] for name, data in constraint_nodes if data["kind"] == "leq"
        ]
        eq_constraint_bounds = [
            data["bound"] for name, data in constraint_nodes if data["kind"] == "eq"
        ]

        if leq_constraint_bounds:
            features["leq_constraint_bounds_mean"] = np.mean(leq_constraint_bounds)
            features["leq_constraint_bounds_stddev"] = np.std(leq_constraint_bounds)
        else:
            features["leq_constraint_bounds_mean"] = 0.0
            features["leq_constraint_bounds_stddev"] = 0.0

        if eq_constraint_bounds:
            features["eq_constraint_bounds_mean"] = np.mean(eq_constraint_bounds)
            features["eq_constraint_bonds_stddev"] = np.std(eq_constraint_bounds)
        else:
            features["eq_constraint_bounds_mean"] = 0.0
            features["eq_constraint_bonds_stddev"] = 0.0
        return features

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

    def __yield_constraints(self) -> Generator[_GeneralConstraintData, None, None]:
        """
        Yield the constraints within a model. Currently supports:
        - ``pyo.Constraint``s defined as model properties
        - Constraints within ``IndexedConstraint``s
        - Constraints within ``ConstraintList``s (by nature of ``ConstraintList``
          subclassing ``IndexedConstraint``)

        TODO: The return type for this is ``_GeneralConstraintData``, because
         that's what a ``ConstraintList`` will yield directly. There is no
         public class we can use in its place, but this is an unstable
         annotation relying on a private Pyomo class.
        """

        for x in self.model.component_objects(pyo.Constraint):
            if isinstance(x, _GeneralConstraintData):
                yield x
            elif isinstance(x, IndexedConstraint):
                # Iterating over an IndexedConstraint yields the index
                for idx in x:
                    yield x[idx]

    @lru_cache(maxsize=None)
    def get_variable_constraint_graph(self) -> nx.Graph:
        """
        Construct a bipartite Variable Constraint Graph of the problem instance

        One node set is the set of problem variables, the other is the set of
        problem constraints. Nodes are connected by edges if a variable
        participates in a constraint.

        Variable nodes have the following attributes:
        - ``type``: "variable"
        - ``domain``: One of "continuous", "integer", or "binary"
        - ``obj_coeff``: The coefficient of the variable in the objective function.
          Only present the objective is linear and the variable participates in it.

        Constraint nodes have the following attributes:
        - ``type``: "constraint"
        - ``kind``: "leq" or "eq", indicating inequality or equality constraint
        - ``original_kind``: One of "leq", "geq", or "eq" indicating whether the
          constraint was <=, >=, or == prior to being transformed to one of <= or ==
        - ``is_linear``: Whether the constraint is linear

        Edges have the following attributes:
        - ``is_liner``: Whether the corresponding constraint is linear
        - ``coeff``: Contains the coefficient of the variable in the constraint.
          Only present if ``is_linear`` is True.

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

        for constr in self.__yield_constraints():
            constr_name = constr.getname()
            is_linear, var_list = decompose_term(constr.body)

            # All constraints in the VCG will either be <= or == bounded
            if constr.has_lb() and not constr.has_ub():  # constr >= b
                multiplier = -1
                bound = constr.lower()
                kind = "leq"
                original_kind = "geq"
            elif not constr.has_lb() and constr.has_ub():  # constr <= b
                multiplier = 1
                bound = constr.upper()
                kind = original_kind = "leq"
            elif constr.lower() == constr.upper():  # constr == b
                multiplier = 1
                bound = constr.upper()
                kind = original_kind = "eq"
            else:
                # lb <= constr <= ub
                raise NotImplementedError(
                    "Double-bounded constraints are not supported"
                )

            bound *= multiplier

            G.add_node(
                constr_name,
                type="constraint",
                kind=kind,
                original_kind=original_kind,
                is_linear=is_linear,
            )

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
                    G.add_edge(constr_name, var.getname(), is_linear=False)
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

                    G.add_edge(constr_name, var.getname(), is_linear=True, coeff=coeff)

            # Add attributes to the constraint node
            G.nodes[constr_name]["bound"] = bound

        # Tag variable nodes with their coefficient in the objective
        is_objective_linear, objective_var_list = decompose_term(
            self.model.objective.expr
        )
        objective_multiplier = 1 if self.model.objective.is_minimizing() else -1

        if is_objective_linear:
            for coeff, var in objective_var_list:
                if var is None:
                    # It's technically possible to add a constant term to the objective
                    continue

                # If the variable is not in the VCG by now, that means it was never
                # part of a constraint
                if var.getname() not in G.nodes:
                    raise ValueError(
                        f"Variable {var.getname()} appears in the objective "
                        "function without participating in any constraints"
                    )

                G.nodes[var.getname()]["obj_coeff"] = coeff * objective_multiplier

        return G

    def __build_full_path(
        self, extension: str, path_prefix: Union[str, Path], extra_params: str = ""
    ) -> str:
        """
        Build a filename, creating directories so it can be immediately
        written to if necessary
        """

        filename = self.get_name() + extra_params + extension

        if path_prefix:
            if isinstance(path_prefix, str):
                path_prefix = Path(path_prefix).resolve()
        else:
            path_prefix = Path.cwd()

        filename = path_prefix.joinpath(filename)

        filename = Path(filename)

        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)

        return str(filename)

    def save(
        self, path_prefix: Union[Path, str] = None, params=False, features=False
    ) -> None:
        """
        Save the associated model, problem parameters, and problem features

        The model will be saved to a .mps file if it is linear, and .gms
        otherwise.

        Parameters and features will be saved to json files.

        :param path_prefix: Folder to save to
        :param params: Whether to save model parameters to a pickle file, default False
        :param features: Whether to save computed features to a json file, default False.
               This can be very slow for large models.
        """
        if self.is_linear:
            model_filename = self.__build_full_path(".mps", path_prefix)
        else:
            model_filename = self.__build_full_path(".gms", path_prefix)

        self.model.write(model_filename, io_options={"symbolic_solver_labels": True})

        if params:
            params_filename = self.__build_full_path(
                ".pkl", path_prefix, extra_params="_parameters"
            )

            parameters = self.get_parameters()

            with open(params_filename, "wb+") as fd:
                pickle.dump(parameters, fd)

        if features:
            features_filename = self.__build_full_path(
                ".json", path_prefix, extra_params="_features"
            )

            features = self.get_features()

            with open(features_filename, "w+") as fd:
                json.dump(features, fd)

    def save_graph(self, path_prefix: str = None) -> None:
        pass

    def dump(self, path_prefix: Union[Path, str] = None) -> None:
        """
        Dump the entire :class:`Problem` instance as a pickle object

        Filename is given by ``self.get_name()``.

        :param path_prefix: Folder to save to, default to current directory
        """

        filename = self.__build_full_path(".pkl", path_prefix)

        with open(filename, "wb+") as fd:
            pickle.dump(self, fd)

    @classmethod
    def load(cls: Type[T], pkl_path: Union[Path, str]) -> T:
        """
        Load a problem instance from a pickle file saved by ``dump()``

        :param pkl_path: Path to pickle file
        :return: Problem instance
        """

        with open(pkl_path, "rb") as fd:
            obj = pickle.load(fd)

        return obj

    @staticmethod
    def from_params(cls: Type[T], pkl_path: Union[Path, str]) -> T:
        """
        Instantiate a problem instance from the supplied
        pickled arguments

        Usage: ::
            class MyProblem(Problem):
                ...

            my_problem = MyProblem(**kwargs)

            my_problem.save(params=True, features=False)

            ...

            loaded_problem = Problem.from_params(MyProblem, "path_to_parameters.pkl")

        :param cls: Class to instantiate. This is necessary due to
            generic types being hints only in Python.
        :param pkl_path: Path to pickle file
        :return: Problem instance
        """

        with open(pkl_path, "rb") as fd:
            params = pickle.load(fd)

        return cls(**params)
