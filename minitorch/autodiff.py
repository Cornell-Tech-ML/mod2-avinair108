from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_list = list(vals)

    vals_list[arg] += epsilon
    f_plus = f(*vals_list)

    # subtract twice because added before
    vals_list[arg] -= 2 * epsilon
    f_minus = f(*vals_list)

    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the variable."""

    ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Checks if the variable is a leaf in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Checks if the variable is constant (i.e., has no gradient)."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients.

        Args:
        ----
            d_output (Any): The derivative of the output with respect to this variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: Gradients for each parent variable.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    result: list[Variable] = []

    def dfs(v: Variable) -> None:
        """Function to do depth-first search."""
        if v.unique_id in visited or v.is_constant():
            return

        visited.add(v.unique_id)

        for parent in v.parents:
            dfs(parent)

        result.append(v)

    dfs(variable)

    return reversed(result)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation to compute derivatives for leaf nodes.

    Args:
    ----
        variable (Variable): The variable to start backpropagation from.
        deriv (Any): The derivative to propagate backward through the computation graph.

    """
    derivatives = {}

    topo_order = list(topological_sort(variable))

    derivatives[variable.unique_id] = deriv

    for var in topo_order:
        d_var = derivatives.get(var.unique_id, 0.0)

        if var.is_leaf():
            var.accumulate_derivative(d_var)
        else:
            for parent, local_deriv in var.chain_rule(d_var):
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = 0.0
                derivatives[parent.unique_id] += local_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Get saved values."""
        return self.saved_values
