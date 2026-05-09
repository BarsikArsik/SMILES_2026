"""
zo_optimizer.py — Zero-order optimizer skeleton (student-implemented).

Students: Implement your gradient-free optimization logic inside
``ZeroOrderOptimizer``. The skeleton uses a 2-point central-difference
estimator as a starting point — you are expected to replace or extend it.

Key design points
-----------------
* **Layer selection** is entirely your responsibility. Set ``self.layer_names``
  to the list of parameter names you want to optimize. You can change this list
  at any time — even between ``.step()`` calls — to implement curriculum or
  progressive-layer strategies.
* **Compute budget** is enforced by ``validate.py``: ``.step()`` is called
  exactly ``n_batches`` times. Each call may invoke the model as many times as
  your estimator requires, but be mindful that more evaluations per step leave
  fewer steps in the total budget.
* **No gradients** are computed anywhere in this file. All updates must be
  derived from scalar loss values obtained by calling ``loss_fn()``.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """Gradient-free optimizer for fine-tuning a subset of model parameters.

    The optimizer maintains a list of *active* parameter names
    (``self.layer_names``). On each ``.step()`` call it perturbs only those
    parameters, estimates a pseudo-gradient from forward-pass loss values, and
    applies an update. All other parameters remain strictly frozen.

    Args:
        model:            The ``nn.Module`` to optimize.
        lr:               Step size / learning rate.
        eps:              Perturbation magnitude for the finite-difference
                          estimator.
        perturbation_mode: Distribution used to sample the perturbation
                          direction. ``"gaussian"`` draws from N(0, I);
                          ``"uniform"`` draws from U(-1, 1) and normalises.

    Student task:
        1. Set ``self.layer_names`` to the parameter names you want to tune.
           Inspect available names with ``[n for n, _ in model.named_parameters()]``.
        2. Replace or extend ``_estimate_grad`` with a better estimator.
        3. Replace or extend ``_update_params`` with a better update rule.
        4. Optionally change ``self.layer_names`` inside ``.step()`` to
           implement dynamic layer selection strategies.

    Example — tune only the final linear layer::

        optimizer = ZeroOrderOptimizer(model)
        optimizer.layer_names = ["fc.weight", "fc.bias"]
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 2e-3,
        eps: float = 5e-4,
        perturbation_mode: str = "gaussian",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps

        if perturbation_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode

        # ------------------------------------------------------------------
        # STUDENT: Set self.layer_names to the parameters you want to tune.
        #
        # The default below selects only the final classification head.
        # You may replace this with any subset of named parameters, e.g.:
        #   self.layer_names = ["layer4.1.conv2.weight", "fc.weight", "fc.bias"]
        #
        # You can also update self.layer_names inside .step() to implement
        # a dynamic schedule (e.g. gradually unfreeze deeper layers).
        # ------------------------------------------------------------------
        self.layer_names: list[str] = ["fc.weight", "fc.bias"]
        # ------------------------------------------------------------------

        # Added for a more stable zero-order update.
        self.momentum = 0.8
        self.grad_clip = 1.0
        self.normalize_grad = True
        self.velocity: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Internal helpers — students may modify these.
    # ------------------------------------------------------------------

    def _active_params(self) -> dict[str, nn.Parameter]:
        """Return a mapping from name → parameter for all active layer names.

        Only parameters whose names appear in ``self.layer_names`` are
        returned. Parameters not in this mapping are never modified.

        Returns:
            Dict mapping parameter name to its ``nn.Parameter`` tensor.

        Raises:
            KeyError: If a name in ``self.layer_names`` does not exist in the
                      model.
        """
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        """Sample a random perturbation vector of the same shape as ``param``.

        For SPSA we use Rademacher noise: each element is independently sampled
        from {-1, +1}. This allows simultaneous perturbation of all active
        parameters using only two loss evaluations.
        """
        return torch.empty_like(param).bernoulli_(0.5).mul_(2.0).sub_(1.0)

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """Estimate a pseudo-gradient for each active parameter.

        Replaced skeleton with an SPSA-style estimator:
        all active parameters are perturbed simultaneously, requiring only
        two loss evaluations regardless of model size.

        Args:
            loss_fn: Callable that evaluates the objective on the current batch
                     and returns a scalar ``float``. May be called multiple
                     times; each call must use the *same* batch.
            params:  Dict of active parameter name → tensor (from
                     ``_active_params``).

        Returns:
            Dict mapping each parameter name to its estimated pseudo-gradient
            tensor (same shape as the parameter).
        """
        grads: dict[str, torch.Tensor] = {}
        deltas = {name: self._sample_direction(param) for name, param in params.items()}

        with torch.no_grad():
            # f(theta + eps * delta)
            for name, param in params.items():
                param.data.add_(self.eps * deltas[name])
            f_plus = float(loss_fn())

            # f(theta - eps * delta)
            for name, param in params.items():
                param.data.sub_(2.0 * self.eps * deltas[name])
            f_minus = float(loss_fn())

            # restore original parameters
            for name, param in params.items():
                param.data.add_(self.eps * deltas[name])

        scale = (f_plus - f_minus) / (2.0 * self.eps)

        for name, param in params.items():
            grad = scale * deltas[name]

            if self.normalize_grad:
                grad = grad / (param.numel() ** 0.5)

            grads[name] = grad

        return grads

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Apply the estimated pseudo-gradients to the active parameters.

        Replaced vanilla update with clipped momentum descent for a more stable
        zero-order optimisation step.
        """
        with torch.no_grad():
            for name, param in params.items():
                grad = grads[name]
                grad = torch.clamp(grad, min=-self.grad_clip, max=self.grad_clip)

                if name not in self.velocity:
                    self.velocity[name] = torch.zeros_like(param.data)

                self.velocity[name].mul_(self.momentum).add_(grad, alpha=1.0 - self.momentum)
                param.data.sub_(self.lr * self.velocity[name])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Perform one zero-order optimisation step.

        Calls ``loss_fn`` one or more times to estimate pseudo-gradients for
        the currently active parameters (``self.layer_names``), then applies
        an update. Parameters *not* in ``self.layer_names`` are never touched.

        Args:
            loss_fn: A callable that takes no arguments and returns a scalar
                     ``float`` representing the loss on the current mini-batch.
                     ``validate.py`` guarantees that every call to ``loss_fn``
                     within a single ``.step()`` invocation uses the *same*
                     fixed batch of data.

        Returns:
            The loss value at the *start* of the step (before any update),
            obtained from the first call to ``loss_fn()``.

        Note:
            ``validate.py`` calls ``.step()`` exactly ``n_batches`` times.
            Each forward pass inside ``loss_fn`` counts toward your compute
            budget, so prefer estimators that minimise the number of calls.
        """
        params = self._active_params()

        # Record the loss before any perturbation.
        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        return float(loss_before)