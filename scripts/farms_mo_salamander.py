"""Farms multi-objective optimisation for salamander experiment"""

import numpy as np
import platypus as pla
# from platypus import (
#     NSGAII,
#     NSGAIII,
#     Problem,
#     Real,
#     Hypervolume,
#     experiment,
#     calculate,
#     display,
#     nondominated
# )
import matplotlib.pyplot as plt


class ProblemLogger:
    """Problem population logger"""

    def __init__(self, n_evaluations, n_vars, n_objs):
        super(ProblemLogger, self).__init__()
        self.n_evaluations = n_evaluations
        self.variables = np.zeros([n_evaluations, n_vars])
        self.objectives = np.zeros([n_evaluations, n_objs])
        self.iteration = 0

    def log(self, variables, objectives):
        """Log individual"""
        if self.iteration < self.n_evaluations:
            self.variables[self.iteration] = variables
            self.objectives[self.iteration] = objectives
        self.iteration += 1

    def plot(self, algorithm):
        """Plot variables"""
        nondominated_solutions = pla.nondominated(algorithm.result)

        for solution in nondominated_solutions:
            print("Decision vector: {} Fitness: {}".format(
                solution.variables,
                list(solution.objectives)
            ))

        # Variables
        plt.figure("Varable space")
        plt.plot(
            self.variables[:, 0],
            self.variables[:, 1],
            "bo"
        )
        plt.plot(
            [s.variables[0] for s in nondominated_solutions],
            [s.variables[1] for s in nondominated_solutions],
            "ro"
        )
        plt.grid(True)

        # Fitness
        plt.figure("Fitness space")
        plt.plot(
            self.objectives[:, 0],
            self.objectives[:, 1],
            "bo"
        )
        plt.plot(
            [s.objectives[0] for s in nondominated_solutions],
            [s.objectives[1] for s in nondominated_solutions],
            "ro"
        )
        plt.grid(True)
        # for solution in algorithm.result:
        #     plt.plot(solution.variables[0], solution.variables[1], "bx")
        print("Pareto front size: {}/{}".format(
            len(nondominated_solutions),
            self.iteration
        ))


class Schaffer(pla.Problem):

    def __init__(self, n_evaluations):
        n_vars, n_objs = 2, 2
        super(Schaffer, self).__init__(nvars=n_vars, nobjs=n_objs)
        self.types[0] = pla.Real(-10, 10)
        self.types[1] = pla.Real(-10, 10)
        self.logger = ProblemLogger(n_evaluations, n_vars, n_objs)

    def evaluate(self, solution):
        solution.objectives[0] = (
            + (solution.variables[0]-4)**2
            + (solution.variables[0]-2)**3
            + (solution.variables[1]-7)**4
        )
        solution.objectives[1] = (
            + (solution.variables[0]-7)**2
            + (solution.variables[1]-0)**2
        )
        self.logger.log(solution.variables, solution.objectives)


def main():
    """Main"""

    n_evaluations = int(1e5)

    problem = Schaffer(n_evaluations)
    # algorithm = pla.NSGAII(problem)
    # algorithm = pla.NSGAIII(problem, divisions_outer=10)
    # algorithm = pla.CMAES(problem)
    algorithm = pla.MOEAD(problem)
    # algorithms = [
    #     NSGAII,
    #     (NSGAIII, {"divisions_outer":12}),
    #     (CMAES, {"epsilons":[0.05]}),
    #     GDE3,
    #     IBEA,
    #     (MOEAD, {
    #         "weight_generator":normal_boundary_weights,
    #         "divisions_outer":12
    #     }),
    #     (OMOPSO, {"epsilons":[0.05]}),
    #     SMPSO,
    #     SPEA2,
    #     (EpsMOEA, {"epsilons":[0.05]})
    # ]
    algorithm.run(n_evaluations)

    problem.logger.plot(algorithm)

    plt.show()

    # problems = [problem]

    # algorithms = [NSGAII, (NSGAIII, {"divisions_outer":12})]

    # # run the experiment
    # results = experiment(algorithms, problems, nfe=1000, seeds=10)

    # # calculate the hypervolume indicator
    # hyp = Hypervolume(minimum=[0, 0, 0], maximum=[1, 1, 1])
    # hyp_result = calculate(results, hyp)
    # display(hyp_result, ndigits=3)


if __name__ == "__main__":
    main()
