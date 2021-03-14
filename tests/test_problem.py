from unittest.mock import MagicMock
import pytest

import pyomo.environ as pyo
import networkx as nx

from discretenet.problem import Problem


class MockProblem(Problem):
    is_linear = True

    def __init__(self):
        super().__init__()

        self.model = MagicMock()

    def get_name(self) -> str:
        return "mock_problem"

    def get_parameters(self):
        return {}


class LinearProblem(Problem):
    """
    A simple linear problem for testing purposes. Desirable
    properties:
        - Multiple variable types (cont, binary, integer)
        - Linear constraints and objective function
        - Constraints in all directions (<=, ==, >=)
    """

    is_linear = True

    def __init__(self):
        super().__init__()

        model = pyo.ConcreteModel()

        # Set variable with integer domain
        model.x = pyo.Var([1, 2], domain=pyo.NonNegativeIntegers)
        # Singleton variable with continuous domain
        model.y = pyo.Var(domain=pyo.NonNegativeReals)
        # Singleton variable with binary domain
        model.z = pyo.Var(domain=pyo.Boolean)

        model.constr1 = pyo.Constraint(expr=2 * model.x[1] + 3 * model.x[2] <= 10)
        model.constr2 = pyo.Constraint(expr=model.x[1] + model.x[2] + model.y + 1 >= 2)
        model.constr3 = pyo.Constraint(expr=model.y + model.z == 3)

        # Maximizing objective, to make sure the coeffs are flipped
        model.OBJ = pyo.Objective(
            expr=-(1 * model.x[1] + 2 * model.x[2] + 3 * model.y + 4 * model.z),
            sense=pyo.maximize,
        )
        self.model = model

    def get_name(self) -> str:
        return "linear_problem"

    def get_parameters(self):
        return {}


class NonlinearProblem(Problem):
    """
    A simple nonlinear problem for testing purposes.
    """

    is_linear = True

    def __init__(self):
        super().__init__()

        model = pyo.ConcreteModel()

        model.x = pyo.Var([1, 2], domain=pyo.Reals)

        # Nonlinear constraint
        model.constr1 = pyo.Constraint(
            expr=2 * model.x[1] * model.x[1] + model.x[2] <= 10
        )

        # Nonlinear objective function
        model.OBJ = pyo.Objective(
            expr=model.x[1] * model.x[1] + model.x[2] * model.x[2]
        )
        self.model = model

    def get_name(self) -> str:
        return "linear_problem"

    def get_parameters(self):
        return {}


@pytest.fixture
def vcg() -> nx.Graph:
    problem = LinearProblem()
    return problem.get_variable_constraint_graph()


@pytest.fixture
def nonlinear_vcg() -> nx.Graph:
    problem = NonlinearProblem()
    return problem.get_variable_constraint_graph()


class TestLinearVCG:
    """
    Test variable constraint graph creation
    """

    def test_variable_nodes_labeled(self, vcg):
        expected_var_nodes = ["x[1]", "x[2]", "y", "z"]
        actual_var_nodes = [
            node_name
            for node_name, node_data in vcg.nodes(data=True)
            if node_data["type"] == "variable"
        ]
        assert sorted(actual_var_nodes) == expected_var_nodes

    def test_variable_node_domains(self, vcg):
        # Domain types
        assert vcg.nodes["x[1]"]["domain"] == "integer"
        assert vcg.nodes["x[2]"]["domain"] == "integer"
        assert vcg.nodes["y"]["domain"] == "continuous"
        assert vcg.nodes["z"]["domain"] == "binary"

    def test_variable_node_obj_coeffs(self, vcg):
        # Objective function coefficients
        assert vcg.nodes["x[1]"]["obj_coeff"] == 1
        assert vcg.nodes["x[2]"]["obj_coeff"] == 2
        assert vcg.nodes["y"]["obj_coeff"] == 3
        assert vcg.nodes["z"]["obj_coeff"] == 4

    def test_constraint_nodes_labelled(self, vcg):
        expected_constr_nodes = ["constr1", "constr2", "constr3"]
        actual_constr_nodes = [
            node_name
            for node_name, node_data in vcg.nodes(data=True)
            if node_data["type"] == "constraint"
        ]
        assert sorted(expected_constr_nodes) == actual_constr_nodes

    def test_constraint_node_kinds(self, vcg):
        # constr1 and constr2 are leq (after adjustment), constr3 is eq
        assert vcg.nodes["constr1"]["kind"] == "leq"
        assert vcg.nodes["constr2"]["kind"] == "leq"
        assert vcg.nodes["constr3"]["kind"] == "eq"

    def test_constraint_node_bounds(self, vcg):
        # Bounds - constr2 should have its bounds adjusted to (2 - 1)*(-1) = -1
        assert vcg.nodes["constr1"]["bound"] == 10
        assert vcg.nodes["constr2"]["bound"] == -1
        assert vcg.nodes["constr3"]["bound"] == 3

    def test_edges(self, vcg):
        expected_edges = {
            "constr1": [("x[1]", 2), ("x[2]", 3)],
            "constr2": [("x[1]", 1), ("x[2]", 1), ("y", 1)],
            "constr3": [("y", 1), ("z", 1)],
        }

        assert len(vcg.edges) == 7

        for constraint, variables in expected_edges.items():
            for variable, coeff in variables:
                assert vcg[constraint][variable]["coeff"] == coeff


class TestNonlinearVCG:
    def test_variable_nodes_labelled(self, nonlinear_vcg):
        expected_var_nodes = ["x[1]", "x[2]"]
        actual_var_nodes = [
            node_name
            for node_name, node_data in nonlinear_vcg.nodes(data=True)
            if node_data["type"] == "variable"
        ]
        assert sorted(actual_var_nodes) == expected_var_nodes

    def test_variable_node_domains(self, nonlinear_vcg):
        assert nonlinear_vcg.nodes["x[1]"]["domain"] == "continuous"
        assert nonlinear_vcg.nodes["x[2]"]["domain"] == "continuous"

    def test_no_edge_coeffs(self, nonlinear_vcg):
        """
        Nonlinear edges aren't annotated with coefficients, since Pyomo
        can't extract them
        """

        assert "coeff" not in nonlinear_vcg["constr1"]["x[1]"]
        assert "coeff" not in nonlinear_vcg["constr1"]["x[2]"]

    def test_no_obj_coeff_for_variable_nodes(self, nonlinear_vcg):
        assert "obj_coeff" not in nonlinear_vcg.nodes["x[1]"]
        assert "obj_coeff" not in nonlinear_vcg.nodes["x[2]"]


class TestSavingLoading:
    def test_save_linear_model(self, tmp_path):
        problem = MockProblem()
        problem.save(tmp_path)

        problem.model.write.assert_called_with(str(tmp_path / "mock_problem.mps"))

        params_path = tmp_path / "mock_problem_parameters.json"
        assert not params_path.exists()

        features_path = tmp_path / "mock_problem_features.json"
        assert not features_path.exists()

    def test_save_nonlinear_problem(self, tmp_path):
        problem = MockProblem()
        problem.is_linear = False
        problem.save(tmp_path)

        problem.model.write.assert_called_with(str(tmp_path / "mock_problem.gms"))

    def test_save_extra_files(self, tmp_path):
        problem = LinearProblem()
        problem.save(tmp_path, model_only=False)

        params_path = tmp_path / "linear_problem_parameters.json"
        assert params_path.exists()

        features_path = tmp_path / "linear_problem_features.json"
        assert features_path.exists()
