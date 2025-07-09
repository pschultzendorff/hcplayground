import logging
from typing import Callable, Optional

import jax.numpy as jnp
from jax import grad, jit, vmap
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Newton:
    """
    A class to perform Newton's method for finding roots of a function.

    Attributes:
        f: The function for which to find the roots.
        gradf: The gradient of the function.
        epsilon: The tolerance for the solution.
        x: The current solution estimate.

    """

    def __init__(self, f: Callable, x: Optional[float]=None, epsilon: float=1e-3, max_steps: int = 30) -> None:
        """Initialize the Newton method with a function, tolerance, and optional initial
        guess.

        Args:
            f: The function for which to find the roots.
            epsilon: The tolerance for the solution.
            x: The initial guess for the solution. Defaults to None.

        """
        self.f: Callable = f
        self.gradf: Callable = grad(f)
    
        if x is not None:
            self.x: float = x
        else:
            self.x = 0.0
        self.fx: float = self.f(self.x)

        self.xx: list[float] = [self.x]
        self.fxx: list[float] = [self.fx]

        self.epsilon: float = epsilon
        self.max_steps: int = max_steps
        logger.info(f"Initialized Newton method with epsilon={self.epsilon}, max_steps={self.max_steps} and initial x={self.x}")

    def solve(self) -> bool:
        """Solve for the root of the function using Newton's method.

        Returns:
            True if the solution converged, False otherwise.

        """
        self.iter: int = 0
        with logging_redirect_tqdm([logging.root]):
            with tqdm(total=100, desc="Newton method", unit="step", position=1, leave=False) as pbar:
                while jnp.abs(self.fx) > self.epsilon and self.iter < self.max_steps:
                    self.step()
                    pbar.set_postfix({"x": self.x, "f(x)": self.fx, "Delta x": jnp.abs(self.x - self.xx[-2])})
                    pbar.update(1)
        if jnp.abs(self.fx) <= self.epsilon:
            logger.info(f"Newton converged to solution x={self.x} in {self.iter} steps")
            return True
        else:
            logger.info(f"Newton did not converge to solution after {self.max_steps} steps. Last residual is {self.fx}")
            return False

    def step(self) -> float:
        """Perform a single step of Newton's method.

        Returns:
            The updated solution estimate.

        """
        self.x = self.x - self.fx / self.gradf(self.x)
        self.fx = self.f(self.x)

        self.xx.append(self.x)
        self.fxx.append(self.fx)
        self.iter += 1
        return self.x


class HC:

    def __init__(self,f: Callable, g: Callable, x: Optional[float] = None, decay: float = 0.9, epsilon: float = 1e-3, max_steps: int = 30) -> None:
        self.f: Callable = f
        self.g: Callable = g
        self.decay: float = decay
        if x is not None:
            self.x: float = x
        else:
            self.x = 0.0


        self.epsilon: float = epsilon
        self.max_steps: int = max_steps
        logger.info(f"Initialized HC method with {self.epsilon=}, {self.max_steps=} and initial x={self.x}")

    def solve(self) -> bool:
        """Solve for the root of ``f`` using the homotopy continuation method.

        Returns:
            True if converged, else False.

        """
        self.iter: int = 0
        self.lambda_: float = 1.0

        self.lambdas: list[float] = [self.lambda_]
        self.xx: list[list[float]] = []
        self.hxx: list[list[float]] = []

        self.converged: bool = False
        self.diverged: bool = False

        with logging_redirect_tqdm([logging.root]):
            with tqdm(total=self.lambda_, desc="Homotopy continuation", unit="step", position=0) as pbar:
                while self.lambda_ > self.epsilon and self.iter < self.max_steps:
                    self.step()
                    if self.diverged:
                        logger.info(f"Stopping homotopy continuation due to non-convergence.")
                        break

                    pbar.set_postfix({"x": self.x, "lambda": self.lambda_})
                    pbar.update(self.lambdas[-2] - self.lambda_)
        if not self.diverged:
            self.converged = True
            logger.info(f"Homotopy continuation converged to solution x={self.x}")
        return self.converged

    def step(self) -> float:
        """Perform a single step of the homotopy continuation method.

        Returns:
            True if Newton converged, False otherwise.

        """
        h = lambda x: self.lambda_*self.g(x) + (1 - self.lambda_) * self.f(x)
        self.Newton: Newton = Newton(h, self.x)
        if self.Newton.solve():
            self.x = self.Newton.x
            self.lambda_ *= self.decay

            self.lambdas.append(self.lambda_)
            self.xx.append(self.Newton.xx)
            self.hxx.append(self.Newton.fxx)
            self.iter += 1
        else:
            self.diverged= True
        return self.x
