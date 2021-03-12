import numpy.random as random
from pathlib import Path
from typing import Union
import pyomo.environ as pyo

from discretenet.problem import Problem
from discretenet.generator import Generator


class SchoolBusSchedulingProblem(Problem):
    def __init__(self, all_time, schools, routes, time_window, name):
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

    def is_linear(self):
        return True

    def get_name(self):
        return self.name


class SchoolBusSchedulingGenerator(Generator):
    def __init__(
        self,
        random_seed: int = 42,
        path_prefix: Union[str, Path] = None,
        num_routes=6,
        max_time=120,
        # school_start_interval=5,
        num_schools=5,
        time_window=20,
        route_length_avg=30,
        route_length_std=10,
    ):
        """
        Initialize the school bus scheduling problem generator instance following https://arxiv.org/abs/1803.09040v2
        :param random_seed: The random seed to use
        :param path_prefix: Path prefix to pass to instance ``save()`` methods
            during batch generation. Must be set as a public instance attribute,
            to be changed by the user after generator instantiation if desired.
        :param num_routes: number of bus routes per school
        :param max_time: total number of time slots ex: 120 for 2 hours
        :param school_start_interval: school start time intervals ex: 10 for 10, 20, 30 ... (Not implemented)
        :param num_schools: number of schools
        :param time_window: time window of acceptable bus arrival time
        :param route_length_avg: average route length (time)
        :param route_length_std: route length standard deviation
        """
        super().__init__(random_seed, path_prefix)
        self.num_routes = num_routes
        self.all_time = range(1, max_time)
        # self.school_start_times = range(
        #     school_start_interval,
        #     max_time + school_start_interval,
        #     school_start_interval,
        # )
        self.num_schools = num_schools
        self.route_length_avg = route_length_avg
        self.route_length_std = route_length_std
        self.time_window = time_window
        self.schools = None
        self.routes = None
        self.route_ids = None

        self.name = "S{}_R{}_T{}_tw{}_ravg{}_rstd{}".format(
            num_schools,
            num_routes,
            max_time,
            time_window,
            route_length_avg,
            route_length_std,
        )

    def generate(self):
        self.generate_schools()
        self.generate_routes()
        problem = SchoolBusSchedulingProblem(
            self.all_time,
            self.schools,
            self.routes,
            self.time_window,
            self.name + "_%d" % self.random_seed,
        )
        return problem

    def generate_schools(self):
        """
        generate schools using num_schools. Keeping it simple for now,
        later on we should utilize networkx and maybe osm
        """
        self.schools = range(1, self.num_schools + 1)

    def generate_routes(self):
        routes = []
        route_ids = []
        for i, school in enumerate(self.schools):
            routes.append(
                list(
                    random.normal(
                        self.route_length_avg, self.route_length_std, self.num_routes
                    ).astype(int)
                )
            )
            route_ids.append(list(range(1, len(routes[i]) + 1)))
        self.routes = routes
        self.route_ids = route_ids


if __name__ == "__main__":
    generator = SchoolBusSchedulingGenerator(
        random_seed=1,
        path_prefix="easy",
        num_routes=3,
        max_time=120,
        # school_start_interval=5,
        num_schools=5,
        time_window=20,
        route_length_avg=30,
        route_length_std=10,
    )
    generator(1, 1)
