"""
UAF Core — Unified Adaptive Framework v4.0
==========================================
Central coherence engine shared by all applications.

THEORY
------
Every adaptive system at level l evolves via:

    dΓ_l/dτ = η · (Σ_{k<l} Φ_{k→l} · E_{k,l}) · (1 - α_l)²
             - λ · ΔS_env · α_l

where:
  Γ_l   — future-coherence at level l  (how much open possibility remains)
  τ     — internal relational time (not external clock)
  Φ_{k→l} — correlation flux from level k to l
  E_{k,l} — entanglement strength between levels
  α_l   — integration degree (0 = fully open, 1 = fully crystallised)
  η     — amplification (technology, trust, energy input)
  λ     — decoherence rate (environmental noise)
  ΔS_env — rate of entropy growth in surroundings

FIXES vs prior versions
-----------------------
1. Euler integration was used for the ODE — replaced with RK4.
2. Γ was allowed to go negative — now clamped to [0, 1].
3. α dynamics had no coupling back to Γ — corrected below.
4. demo.py (UAF-3.1) used numba @jit on code with Python objects —
   removed; pure numpy is fast enough and portable.
5. UAF-Quantum/main.py was identical to XGrav/main.py (wrong copy) —
   fixed; Quantum module now runs the ODE hierarchy.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


EPS = 1e-12


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class UAFParams:
    """Hyper-parameters for a single hierarchy level."""
    eta: float = 1.0          # amplification coefficient
    lam: float = 0.5          # decoherence rate
    kappa: float = 0.3        # α-growth rate  (KAPPA fix: was unconstrained)
    delta: float = 0.1        # α-decay coupling to Γ
    mu: float = 0.2           # cross-level alignment strength
    beta: float = 1.0         # correlation flux scale
    nu: float = 0.5           # flux damping for large Γ differences
    c: float = 0.0            # energy cost threshold for α growth
    n_levels: int = 3         # number of hierarchy levels


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class UAFState:
    """Full state of the UAF hierarchy."""
    gamma: np.ndarray   # shape (n_levels,)  coherence of future
    alpha: np.ndarray   # shape (n_levels,)  integration degree
    tau: float = 0.0    # internal time

    @classmethod
    def initial(cls, n_levels: int, gamma0: float = 0.8, alpha0: float = 0.1):
        return cls(
            gamma=np.full(n_levels, gamma0),
            alpha=np.full(n_levels, alpha0),
        )

    def copy(self) -> "UAFState":
        return UAFState(
            gamma=self.gamma.copy(),
            alpha=self.alpha.copy(),
            tau=self.tau,
        )


# ---------------------------------------------------------------------------
# Core equations (pure functions, easy to test)
# ---------------------------------------------------------------------------

def correlation_flux(gamma: np.ndarray, alpha: np.ndarray,
                     k: int, l: int, p: UAFParams) -> float:
    """
    Φ_{k→l} = β · (Γ_k · α_l) / (1 + ν·|Γ_k - Γ_l|)
    """
    num = p.beta * gamma[k] * alpha[l]
    den = 1.0 + p.nu * abs(gamma[k] - gamma[l])
    return num / (den + EPS)


def level_efficiency(gamma: np.ndarray, alpha: np.ndarray,
                     l: int, p: UAFParams) -> float:
    """
    E_l = (prod_{i in l} α_i)^{1/N_l} · weighted_flux_ratio
    For a single level l this simplifies to α_l itself times
    the ratio of incoming flux to total flux.
    """
    if l == 0:
        return alpha[0]
    fluxes = [correlation_flux(gamma, alpha, k, l, p) for k in range(l)]
    total_flux = sum(fluxes) + EPS
    weighted_gamma = sum(f * gamma[k] for k, f in enumerate(fluxes))
    ratio = weighted_gamma / total_flux
    return alpha[l] * ratio


def _derivatives(state: UAFState, p: UAFParams,
                 delta_S_env: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute dΓ/dτ and dα/dτ for all levels.
    Returns (dgamma, dalpha) both shape (n_levels,).
    """
    L = len(state.gamma)
    dgamma = np.zeros(L)
    dalpha = np.zeros(L)

    for l in range(L):
        # ── dΓ_l/dτ ──────────────────────────────────────────────────────
        if l == 0:
            # bottom level: no incoming flux from below
            dgamma[l] = -p.lam * delta_S_env * state.alpha[l] * state.gamma[l]
        else:
            growth = 0.0
            for k in range(l):
                phi = correlation_flux(state.gamma, state.alpha, k, l, p)
                E = level_efficiency(state.gamma, state.alpha, k, p)
                growth += phi * E
            decay = p.lam * delta_S_env * state.alpha[l]
            dgamma[l] = (p.eta * growth * (1 - state.alpha[l]) ** 2
                         - decay * state.gamma[l])

        # ── dα_l/dτ ──────────────────────────────────────────────────────
        E_l = level_efficiency(state.gamma, state.alpha, l, p)
        # growth: logistic × (energy - cost)
        grow = p.kappa * state.alpha[l] * (1 - state.alpha[l]) * (E_l - p.c)
        # decay: loss of integration when Γ is high (system stays open)
        decay_a = p.delta * (1 - state.alpha[l]) * state.gamma[l]
        # alignment: pull toward neighbours
        align = 0.0
        if l > 0:
            for k in range(l):
                phi = correlation_flux(state.gamma, state.alpha, k, l, p)
                align += phi * (state.gamma[k] - state.gamma[l])
            align *= p.mu
        dalpha[l] = grow - decay_a + align

    return dgamma, dalpha


# ---------------------------------------------------------------------------
# RK4 integrator  (FIX: replaces Euler)
# ---------------------------------------------------------------------------

def _rk4_step(state: UAFState, p: UAFParams,
              dtau: float, delta_S_env: float = 0.0) -> UAFState:
    def f(s):
        return _derivatives(s, p, delta_S_env)

    g0, a0 = state.gamma, state.alpha

    dg1, da1 = f(state)
    s2 = UAFState(np.clip(g0 + 0.5*dtau*dg1, 0, 1),
                  np.clip(a0 + 0.5*dtau*da1, 0, 1), state.tau)
    dg2, da2 = f(s2)
    s3 = UAFState(np.clip(g0 + 0.5*dtau*dg2, 0, 1),
                  np.clip(a0 + 0.5*dtau*da2, 0, 1), state.tau)
    dg3, da3 = f(s3)
    s4 = UAFState(np.clip(g0 + dtau*dg3, 0, 1),
                  np.clip(a0 + dtau*da3, 0, 1), state.tau)
    dg4, da4 = f(s4)

    new_gamma = g0 + (dtau/6) * (dg1 + 2*dg2 + 2*dg3 + dg4)
    new_alpha = a0 + (dtau/6) * (da1 + 2*da2 + 2*da3 + da4)

    return UAFState(
        gamma=np.clip(new_gamma, 0.0, 1.0),
        alpha=np.clip(new_alpha, 0.0, 1.0),
        tau=state.tau + dtau,
    )


# ---------------------------------------------------------------------------
# UAFSystem: high-level API
# ---------------------------------------------------------------------------

class UAFSystem:
    """
    Simulates a UAF hierarchy over internal time τ.

    Usage
    -----
    sys = UAFSystem(UAFParams(n_levels=3))
    trajectory = sys.run(steps=500, dtau=0.01)
    """

    def __init__(self, params: Optional[UAFParams] = None):
        self.p = params or UAFParams()
        self.state = UAFState.initial(self.p.n_levels)
        self.history: List[UAFState] = [self.state.copy()]

    def step(self, dtau: float = 0.01, delta_S_env: float = 0.0):
        self.state = _rk4_step(self.state, self.p, dtau, delta_S_env)
        self.history.append(self.state.copy())

    def run(self, steps: int = 500, dtau: float = 0.01,
            delta_S_env: float = 0.0) -> List[UAFState]:
        for _ in range(steps):
            self.step(dtau, delta_S_env)
        return self.history

    # ── convenience getters ──────────────────────────────────────────────

    def gamma_trajectory(self) -> np.ndarray:
        """shape (T, n_levels)"""
        return np.array([s.gamma for s in self.history])

    def alpha_trajectory(self) -> np.ndarray:
        """shape (T, n_levels)"""
        return np.array([s.alpha for s in self.history])

    def coherence(self) -> float:
        """Current mean coherence across levels."""
        return float(np.mean(self.state.gamma))

    def time_arrow_strength(self) -> float:
        """
        Arrow of time: how strongly the decay term dominates growth.
        > 0  → future crystallising (normal arrow)
        < 0  → future expanding (time-reversal-like regime)
        """
        L = self.p.n_levels
        if L < 2:
            return 0.0
        g, a = self.state.gamma, self.state.alpha
        # top level only
        l = L - 1
        decay = self.p.lam * a[l] * g[l]
        growth = sum(
            correlation_flux(g, a, k, l, self.p)
            * level_efficiency(g, a, k, self.p)
            for k in range(l)
        ) * self.p.eta * (1 - a[l]) ** 2
        return float(decay - growth)
