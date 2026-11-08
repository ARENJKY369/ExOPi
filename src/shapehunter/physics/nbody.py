"""Newtonian N-body dynamics using a fixed-step velocity-Verlet integrator."""

from __future__ import annotations

import numpy as np


def acceleration(positions: np.ndarray, masses: np.ndarray, gravitational_constant: float = 1.0) -> np.ndarray:
    """Return pairwise Newtonian acceleration; collisions are rejected."""
    x = np.asarray(positions, dtype=float)
    m = np.asarray(masses, dtype=float)
    if x.ndim != 2 or m.shape != (x.shape[0],) or np.any(m <= 0):
        raise ValueError("positions and positive masses have incompatible shapes")
    displacement = x[None, :, :] - x[:, None, :]
    distance2 = np.sum(displacement**2, axis=-1)
    np.fill_diagonal(distance2, np.inf)
    if np.any(distance2 == 0):
        raise ValueError("point-mass collision")
    return gravitational_constant * np.sum(displacement * (m[None, :] / distance2**1.5)[:, :, None], axis=1)


def total_energy(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, gravitational_constant: float = 1.0) -> float:
    """Return kinetic plus Newtonian potential energy."""
    x, v, m = np.asarray(positions, float), np.asarray(velocities, float), np.asarray(masses, float)
    acceleration(x, m, gravitational_constant)
    if v.shape != x.shape:
        raise ValueError("velocities must match positions")
    i, j = np.triu_indices(len(m), 1)
    distances = np.linalg.norm(x[i] - x[j], axis=1)
    return float(0.5 * np.sum(m[:, None] * v**2) - gravitational_constant * np.sum(m[i] * m[j] / distances))


def velocity_verlet(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, dt: float, steps: int, gravitational_constant: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Integrate and return position and velocity histories, including the initial state."""
    x, v, m = np.asarray(positions, float).copy(), np.asarray(velocities, float).copy(), np.asarray(masses, float)
    if dt <= 0 or steps < 0 or int(steps) != steps or v.shape != x.shape:
        raise ValueError("invalid timestep, step count, or velocity shape")
    a = acceleration(x, m, gravitational_constant)
    xs = np.empty((steps + 1,) + x.shape)
    vs = np.empty_like(xs)
    xs[0], vs[0] = x, v
    for i in range(steps):
        x = x + v * dt + 0.5 * a * dt**2
        next_a = acceleration(x, m, gravitational_constant)
        v = v + 0.5 * (a + next_a) * dt
        a = next_a
        xs[i + 1], vs[i + 1] = x, v
    return xs, vs


def adaptive_integrate(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    duration: float,
    gravitational_constant: float = 1.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate point masses with SciPy's adaptive solver instead of a fixed timestep."""
    from scipy.integrate import solve_ivp

    x = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    m = np.asarray(masses, dtype=float)
    if duration <= 0 or not 0 < rtol < 1 or not 0 < atol < 1:
        raise ValueError("duration and integration tolerances must be positive and finite")
    if max_step is not None and max_step <= 0:
        raise ValueError("max_step must be positive")
    if x.ndim != 2 or v.shape != x.shape or m.shape != (x.shape[0],):
        raise ValueError("positions, velocities, and masses have incompatible shapes")
    a0 = acceleration(x, m, gravitational_constant)

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        n_bodies = len(m)
        body_x = state[: n_bodies * x.shape[1]].reshape(x.shape)
        body_v = state[n_bodies * x.shape[1] :].reshape(v.shape)
        body_a = acceleration(body_x, m, gravitational_constant)
        return np.concatenate((body_v.ravel(), body_a.ravel()))

    initial = np.concatenate((x.ravel(), v.ravel()))
    solution = solve_ivp(
        rhs,
        (0.0, float(duration)),
        initial,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=float(duration) if max_step is None else float(max_step),
    )
    if not solution.success:
        raise RuntimeError(f"adaptive integration failed: {solution.message}")
    times = np.asarray(solution.t, dtype=float)
    states = np.asarray(solution.y, dtype=float)
    stride = len(m) * x.shape[1]
    xs = states[:stride].T.reshape((len(times),) + x.shape)
    vs = states[stride:].T.reshape((len(times),) + v.shape)
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(vs)):
        raise RuntimeError("adaptive integration produced non-finite state")
    _ = a0
    return times, xs, vs
