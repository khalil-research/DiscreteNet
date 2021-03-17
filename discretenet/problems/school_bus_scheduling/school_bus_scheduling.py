from pathlib import Path
from typing import Union

import numpy as np
import pyomo.environ as pyo

from discretenet.problem import Problem
from discretenet.generator import Generator


class SchoolBusSchedulingProblem(Problem):
    is_linear = True

    def __init__(
        self, all_time: list, schools: list, routes: list, time_window: int, name: str
    ):
        """
        Construct a concrete Pyomo model for a school bus scheduling problem instance
        :param all_time: list of all possible time slots
            (ex: 1,2,...,120 for 2 hours where each time slot is 1 minute)
        :param schools: list of school ids
        :param routes: a matrix of route lengths, each row represents a school,
            each column represents a bus route for that school, with the value
            representing the route length in time slots (ex 30 represents a
            bus route that takes 30 minutes to complete)
        :param time_window: school start time window
        :param name: name of the instance
        """
        super().__init__()
        self.all_time = all_time
        self.schools = schools
        self.routes = routes
        self.time_window = time_window
        self.name = name

        model = pyo.ConcreteModel()
        model.T = pyo.Set(initialize=all_time)
        model.S = pyo.Set(initialize=schools)
        model.R = pyo.Set(initialize=routes)

        def Rs_init(m):
            return (
                (s, i) for s in m.S for i, route in enumerate(m.R[s])
            )  # i is the index of a route not the route values

        model.Rs = pyo.Set(dimen=2, initialize=Rs_init)

        model.x = pyo.Var(model.T, model.Rs, domain=pyo.Binary)
        model.y = pyo.Var(model.T, model.S, domain=pyo.Binary)
        model.z = pyo.Var(domain=pyo.NonNegativeIntegers)

        model.objective = pyo.Objective(expr=model.z, sense=pyo.minimize)

        def c1_rule(m, s, r):
            return pyo.quicksum(m.x[t, s, r] for t in m.T) == 1

        model.c1 = pyo.Constraint(model.Rs, rule=c1_rule)

        def c2_rule(m, s):
            return pyo.quicksum(m.y[t, s] for t in m.T) == 1

        model.c2 = pyo.Constraint(model.S, rule=c2_rule)

        def c3_rule(m, t, s, r):
            return pyo.quicksum(
                m.x[t_prime, s, r] for t_prime in range(1, t + 1)
            ) <= pyo.quicksum(
                m.y[t_prime, s] for t_prime in range(1, min(t + time_window + 1, 120))
            )

        model.c3 = pyo.Constraint(model.T, model.Rs, rule=c3_rule)

        def c4_rule(m, t, s, r):
            return pyo.quicksum(
                m.y[t_prime, s] for t_prime in range(1, t + 1)
            ) <= pyo.quicksum(m.x[t_prime, s, r] for t_prime in range(1, t + 1))

        model.c4 = pyo.Constraint(model.T, model.Rs, rule=c4_rule)

        def c5_rule(m, t):
            return (
                pyo.quicksum(
                    pyo.quicksum(
                        m.x[t_prime, s, r]
                        for t_prime in range(t, min(t + m.R[s][r], all_time[-1] + 1))
                    )
                    for s, r in m.Rs
                )
                <= model.z
            )

        model.c5 = pyo.Constraint(model.T, rule=c5_rule)
        self.model = model

    def get_name(self):
        return self.name

    def get_parameters(self):
        # to be implemented
        return None


class SchoolBusSchedulingGenerator(Generator[SchoolBusSchedulingProblem]):
    def __init__(
        self,
        random_seed: int = 42,
        path_prefix: Union[str, Path] = None,
        num_routes=6,
        max_time=120,
        num_schools=5,
        time_window=20,
        route_length_avg=30,
        route_length_std=10,
    ):
        """
        Initialize the school bus scheduling problem generator instance
        following https://arxiv.org/abs/1803.09040v2
        :param random_seed: The random seed to use
        :param path_prefix: Path prefix to pass to instance ``save()`` methods
            during batch generation. Must be set as a public instance attribute,
            to be changed by the user after generator instantiation if desired.
        :param num_routes: number of bus routes per school
        :param max_time: total number of time slots ex: 120 for 2 hours
        :param num_schools: number of schools
        :param time_window: time window of acceptable bus arrival time
        :param route_length_avg: average route length (time)
        :param route_length_std: route length standard deviation
        """
        super().__init__(random_seed, path_prefix)
        self.num_routes = num_routes
        self.all_time = list(range(1, max_time))
        self.num_schools = num_schools
        self.route_length_avg = route_length_avg
        self.route_length_std = route_length_std
        self.time_window = time_window

        self.name = "S{}_R{}_T{}_tw{}_ravg{}_rstd{}".format(
            num_schools,
            num_routes,
            max_time,
            time_window,
            route_length_avg,
            route_length_std,
        )

    def generate(self):
        schools = self.generate_schools()
        routes = self.generate_routes(schools)
        problem = SchoolBusSchedulingProblem(
            self.all_time,
            schools,
            routes,
            self.time_window,
            self.name + "_%d" % self.random_seed,
        )
        return problem

    def generate_schools(self):
        """
        generate schools using num_schools. Keeping it simple for now,
        in the future we can utilize networkx and maybe osm
        """
        return list(range(1, self.num_schools + 1))

    def generate_routes(self, schools):
        routes = []
        for i, school in enumerate(schools):
            routes.append(
                list(
                    np.random.normal(
                        self.route_length_avg, self.route_length_std, self.num_routes
                    ).astype(int)
                )
            )
        return routes


if __name__ == "__main__":
    generator = SchoolBusSchedulingGenerator(
        random_seed=1,
        path_prefix="easy",
        num_routes=3,
        max_time=120,
        num_schools=10,
        time_window=20,
        route_length_avg=30,
        route_length_std=10,
    )
    generator(1, 1)
