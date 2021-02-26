from abc import abstractmethod
from typing import Generic, List, TypeVar, Union
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np

from discretenet.problem import Problem

T = TypeVar("T", bound=Problem)


class Generator(Generic[T]):
    """
    Abstract Base Class for generator for :class:`discretenet.problem.Problem` instances.

    Must implement the abstract ``__init__`` and ``__generate__`` methods.

    Takes a generic type parameter of a particular ``Problem`` subclass to be generated,
    for type hints. Usage: ::

        class MyGenerator(Generator[MyProblem]):
            ...
    """

    @abstractmethod
    def __init__(self, random_seed: int = 42, path_prefix: Union[str, Path] = None):
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
        :param path_prefix: Path prefix to pass to instance ``save()`` methods
            during batch generation. Must be set as a public instance attribute,
            to be changed by the user after generator instantiation if desired.
        """

        self.random_seed = random_seed
        self.path_prefix = path_prefix

        self.set_seed(random_seed)

    def set_seed(self, seed=42) -> None:
        """
        Set the random seed for any relevant libraries (default numpy) to ensure
        reproducibility.

        :param seed: The random seed to use
        """
        self.random_seed = seed

        np.random.seed(self.random_seed)

    @abstractmethod
    def generate(self) -> T:
        """
        Generate and return a single :class:`discretenet.problem.Problem` instance

        Should sample for parameters from the defined parameter generator callables
        (from the ``__init__()`` method), instantiate an instance with those parameters,
        and return it.

        :return: An initialized concrete ``Problem`` instance
        """

        pass

    def _generate(self, random_seed) -> T:
        self.set_seed(random_seed)
        return self.generate()

    def _generate_and_save(self, random_seed) -> T:
        self.set_seed(random_seed)
        instance = self.generate()
        instance.save()
        return instance

    def __call__(self, n_instances, n_jobs=-1, save=True) -> List[T]:
        """
        Generate and return ``n_instances`` problem instances by calling ``generate()``

        Generation is performed in parallel using ``n_jobs`` Joblib jobs with the default
        backend. If ``n_jobs`` is 1, Joblib is not used. This may be beneficial as it
        avoids having to pickle complex models to send them between processes, but with
        the drawback of not being able to exploit parallelism.

        Because forked processes have identical memory initially, all processes will
        share the same random seed. By default, this means that each set of processes
        would generate the same set of instances. To fix this, we pre-generate
        random seeds for every instance, and call ``self.set_seed()`` prior to each
        instance creation. The main process retains its random state, so subsequent
        calls to the generator will generate different instances.

        :param n_instances: Number of instances to create
        :param n_jobs: Number of joblib jobs to use
        :param save: Whether to immediately save the generated instances. Folder is
            set by ``self.path_prefix``.
        :return: A list of generated concrete ``Problem`` instances
        """

        func = self._generate_and_save if save else self._generate
        random_states = np.random.randint(np.iinfo(np.int32).max, size=n_instances)

        if n_jobs == 1:
            # When using only a single job, bypass the joblib backend entirely.
            # This saves having to use pickle, which may be slow for very large
            # models.
            instances = [func(random_seed) for random_seed in random_states]

        else:
            with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
                instances = parallel(
                    delayed(func)(random_seed) for random_seed in random_states
                )

        return instances
