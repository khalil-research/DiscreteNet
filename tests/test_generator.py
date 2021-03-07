from unittest.mock import MagicMock

import pyomo.environ as pyo
import numpy as np

from discretenet.generator import Generator
from discretenet.problem import Problem


class FakeProblem(Problem):
    """
    A fake problem, with an actual pyomo Concrete Model. Pickle doesn't
    work nicely with Mock objects, so this is used to test parallel
    behaviour.
    """

    is_linear = True

    def __init__(self, obj_coeffs: np.array, constr_coeffs: np.array):
        super().__init__()
        self.n = len(obj_coeffs)
        self.obj_coeffs = obj_coeffs
        self.constr_coeffs = constr_coeffs

        self.model = pyo.ConcreteModel()

    def get_name(self):
        return f"fake_problem_{self.n}"


class FakeGenerator(Generator[FakeProblem]):
    def __init__(self, n=5, random_seed=42, path_prefix="fake"):
        super().__init__(random_seed, path_prefix)
        self.n = n

    def generate(self):
        obj_coeffs = np.random.randint(10, size=self.n)
        constr_coeffs = np.random.randint(10, size=self.n)
        return FakeProblem(obj_coeffs, constr_coeffs)


class MockProblem(Problem):
    is_linear = True

    def __init__(self):
        super().__init__()
        self.model = MagicMock()

    def get_name(self):
        return "mock_problem"


class MockGenerator(Generator[MockProblem]):
    def __init__(self, random_seed=42, path_prefix="mock"):
        super().__init__(random_seed, path_prefix)

    def generate(self):
        return MockProblem()


def test_generator_calls_are_different():
    """
    Each call to the generator should return a fresh set of values,
    aka the seed should not be reset
    """

    generate_fake_problem = FakeGenerator(n=5, random_seed=42)

    instances1 = generate_fake_problem(n_instances=10, n_jobs=1, save=False)
    instances2 = generate_fake_problem(n_instances=10, n_jobs=1, save=False)

    assert any(
        np.any(i1.obj_coeffs != i2.obj_coeffs) for i1, i2 in zip(instances1, instances2)
    )
    assert any(
        np.any(i1.constr_coeffs != i2.obj_coeffs)
        for i1, i2 in zip(instances1, instances2)
    )


def test_generator_reproducible_single_process():
    """
    After resetting the generator seed, generated instances should be
    identical.

    Since we have no good way of testing if Pyomo models are identical,
    we check the attributes set on FakeProblem
    """

    generate_fake_problem = FakeGenerator(n=5, random_seed=42)

    instances1 = generate_fake_problem(n_instances=10, n_jobs=1, save=False)
    generate_fake_problem.set_seed(42)
    instances2 = generate_fake_problem(n_instances=10, n_jobs=1, save=False)

    assert all(
        np.all(i1.obj_coeffs == i2.obj_coeffs) for i1, i2 in zip(instances1, instances2)
    )
    assert all(
        np.all(i1.constr_coeffs == i2.constr_coeffs)
        for i1, i2 in zip(instances1, instances2)
    )


def test_generator_reproducible_multiple_processes():
    """
    Originally, using the multiprocessing joblib backend, forked processes
    will share random states and create identical instances. This was resolved
    by pre-setting the random state for each generate call.

    This behaviour is verified using 2 processes, but it will generalize
    to any number.
    """

    generate_fake_problem = FakeGenerator(n=5, random_seed=42)

    instances = generate_fake_problem(n_instances=8, n_jobs=2, save=False)

    # If the seeds weren't pre-set, some of these would be duplicated
    obj_coeffs = [tuple(i.obj_coeffs.tolist()) for i in instances]
    assert len(instances) == len(set(obj_coeffs))


def test_generator_calls_save_if_told_to(tmp_path):
    generate_mock_problem = MockGenerator(path_prefix=tmp_path)
    instances = generate_mock_problem(n_instances=1, n_jobs=1, save=True)
    instances[0].model.write.assert_called_with(str(tmp_path / "mock_problem.mps"))
