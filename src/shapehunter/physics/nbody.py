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
