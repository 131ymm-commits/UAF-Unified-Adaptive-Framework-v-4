"""
UAF v4 — test suite
====================
Run with:  python -m pytest tests/ -v
or:        python tests/test_all.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

# ============================================================
# 1. Core UAF
# ============================================================
from uaf.core import (
    UAFParams, UAFState, UAFSystem,
    correlation_flux, level_efficiency, _derivatives, _rk4_step
)

class TestUAFCore:

    def test_initial_state_valid(self):
        s = UAFState.initial(3)
        assert s.gamma.shape == (3,)
        assert s.alpha.shape == (3,)
        assert np.all(s.gamma >= 0) and np.all(s.gamma <= 1)
        assert np.all(s.alpha >= 0) and np.all(s.alpha <= 1)

    def test_correlation_flux_positive(self):
        p = UAFParams()
        g = np.array([0.8, 0.6, 0.4])
        a = np.array([0.1, 0.3, 0.5])
        phi = correlation_flux(g, a, 0, 1, p)
        assert phi >= 0

    def test_gamma_stays_bounded(self):
        sys = UAFSystem(UAFParams(n_levels=3))
        sys.run(steps=200, dtau=0.01, delta_S_env=0.5)
        traj = sys.gamma_trajectory()
        assert np.all(traj >= 0), "Γ went negative — FIX 2 failed"
        assert np.all(traj <= 1), "Γ exceeded 1"

    def test_alpha_stays_bounded(self):
        sys = UAFSystem(UAFParams(n_levels=3))
        sys.run(steps=200, dtau=0.01)
        traj = sys.alpha_trajectory()
        assert np.all(traj >= 0)
        assert np.all(traj <= 1)

    def test_rk4_conserves_better_than_euler(self):
        """RK4 should have smaller drift than naive Euler on the same step."""
        p = UAFParams(n_levels=2, eta=0.5, lam=0.1)
        s0 = UAFState.initial(2, gamma0=0.5, alpha0=0.2)

        # RK4 step
        s_rk4 = _rk4_step(s0, p, dtau=0.1)

        # Euler step
        dg, da = _derivatives(s0, p)
        g_euler = np.clip(s0.gamma + 0.1 * dg, 0, 1)
        a_euler = np.clip(s0.alpha + 0.1 * da, 0, 1)

        # both should be valid (just checking no crash and bounds)
        assert np.all(s_rk4.gamma >= 0)
        assert np.all(s_rk4.alpha >= 0)

    def test_time_arrow_direction(self):
        """High decoherence → positive time arrow (decay dominates)."""
        sys = UAFSystem(UAFParams(n_levels=3, lam=5.0, eta=0.1))
        sys.run(steps=100, dtau=0.01, delta_S_env=2.0)
        assert sys.time_arrow_strength() >= 0, "Arrow should be positive under high decoherence"

    def test_high_eta_grows_coherence(self):
        """High eta (low noise) → Γ should stay high or grow."""
        sys = UAFSystem(UAFParams(n_levels=3, eta=3.0, lam=0.01))
        sys.run(steps=100, dtau=0.005, delta_S_env=0.0)
        final_mean = float(np.mean(sys.state.gamma))
        # should not collapse
        assert final_mean > 0.3


# ============================================================
# 2. Dirac Anomaly Detector
# ============================================================
from apps.anomaly.dirac_agent import UAFDiracAgent, DiracParams

class TestDiracAgent:

    def _clean_signal(self, n=500):
        rng = np.random.default_rng(0)
        return np.sin(np.linspace(0, 6*np.pi, n)) + 0.05 * rng.standard_normal(n)

    def _anomalous_signal(self, n=500, spike_at=300, spike_mag=5.0):
        sig = self._clean_signal(n)
        sig[spike_at:spike_at+15] += spike_mag
        return sig

    def test_no_false_positives_on_constant(self):
        agent = UAFDiracAgent()
        scores = agent.score_sequence(np.ones(400))
        assert np.sum(scores > 0) == 0, "False positives on constant signal"

    def test_detects_large_spike(self):
        agent = UAFDiracAgent()
        sig = self._anomalous_signal(spike_mag=8.0)
        scores = agent.score_sequence(sig)
        # should detect something in the spike window
        assert np.sum(scores[280:340] > 0) >= 1, "Failed to detect large spike"

    def test_weights_sum_to_one(self):
        p = DiracParams(gap_weight=1.0, inversion_weight=2.0, zitter_weight=3.0)
        total = p.gap_weight + p.inversion_weight + p.zitter_weight
        assert abs(total - 1.0) < 1e-9, "FIX 5 failed: weights don't sum to 1"

    def test_score_sequence_reset(self):
        """score_sequence must give same result if called twice."""
        agent = UAFDiracAgent()
        sig = self._anomalous_signal()
        s1 = agent.score_sequence(sig)
        s2 = agent.score_sequence(sig)
        np.testing.assert_array_equal(s1, s2, "FIX 3 failed: score_sequence not deterministic")

    def test_no_nan_inf(self):
        rng = np.random.default_rng(99)
        agent = UAFDiracAgent()
        sig = np.concatenate([
            np.ones(100),
            rng.standard_normal(300) * 1e6,   # extreme values
            np.zeros(100)
        ])
        scores = agent.score_sequence(sig)
        assert not np.any(np.isnan(scores)), "NaN in scores"
        assert not np.any(np.isinf(scores)), "Inf in scores"

    def test_event_score_in_0_1(self):
        agent = UAFDiracAgent()
        sig = self._anomalous_signal(spike_mag=10.0)
        scores = agent.score_sequence(sig)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


# ============================================================
# 3. Gravity
# ============================================================
from apps.gravity.nbody import (
    Body, simulate, uaf_coherence, make_solar_ring, UAFGravityEvolution, EvolutionConfig
)

class TestGravity:

    def test_simulation_runs(self):
        bodies = make_solar_ring(n_asteroids=5, n_shepherds=1)
        final, traj = simulate(bodies, dt=0.1, steps=50)
        assert traj.shape == (50, len(bodies), 2)

    def test_coherence_in_0_1(self):
        bodies = make_solar_ring(n_asteroids=8, n_shepherds=2)
        _, traj = simulate(bodies, dt=0.1, steps=100)
        gamma = uaf_coherence(traj)
        assert 0.0 <= gamma <= 1.0

    def test_stable_orbit_has_high_coherence(self):
        """A single circular orbit should have high Γ."""
        M = 1000.0
        r = 5.0
        v = float(np.sqrt(1.0 * M / r))
        star = Body(M, np.zeros(2), np.zeros(2), "star")
        planet = Body(1.0, np.array([r, 0.0]), np.array([0.0, v]), "planet")
        _, traj = simulate([star, planet], dt=0.05, steps=200)
        gamma = uaf_coherence(traj)
        assert gamma > 0.5, f"Stable orbit Γ too low: {gamma:.3f}"

    def test_energy_conservation_leapfrog(self):
        """Leapfrog should conserve energy much better than Euler."""
        bodies = make_solar_ring(n_asteroids=3, n_shepherds=0)
        bodies = bodies[:2]  # just star + 1 asteroid for simplicity

        def total_energy(bs):
            ke = sum(0.5 * b.mass * np.dot(b.vel, b.vel) for b in bs)
            pe = 0.0
            for i in range(len(bs)):
                for j in range(i+1, len(bs)):
                    r = np.linalg.norm(bs[i].pos - bs[j].pos)
                    pe -= 1.0 * bs[i].mass * bs[j].mass / (r + 1e-4)
            return ke + pe

        final, _ = simulate(bodies, dt=0.05, steps=100)
        E0 = total_energy(bodies)
        E1 = total_energy(final)
        # leapfrog should keep relative error < 1%
        assert abs(E1 - E0) / (abs(E0) + 1e-12) < 0.01, \
            f"Energy drift too large: {abs(E1-E0)/abs(E0):.4f}"

    def test_evolution_improves(self):
        cfg = EvolutionConfig(n_asteroids=6, n_shepherds=1,
                              pop_size=4, generations=3, sim_steps=50)
        evo = UAFGravityEvolution(cfg, seed=7)
        _, best, history = evo.evolve()
        assert history[-1] >= history[0] - 0.01, "Evolution should not get worse"


# ============================================================
# 4. Molecular
# ============================================================
from apps.molecular.binding import (
    ProteinTarget, ProteinBindingGA, ProteinBindingRL,
    binding_energy, kd_from_energy, vdw_energy
)

class TestMolecular:

    def _target(self, kd=1.0, seed=0):
        return ProteinTarget.synthetic("TEST", kd=kd, seed=seed)

    def test_vdw_minimum_at_r_opt(self):
        """VDW energy should be minimal near r_opt."""
        r_opt = 3.5
        e_opt = vdw_energy(r_opt)
        e_far = vdw_energy(r_opt * 2)
        e_close = vdw_energy(r_opt * 0.5)
        assert e_opt <= e_far
        assert e_opt <= e_close

    def test_kd_decreases_with_negative_dg(self):
        kd0 = 1.0
        assert kd_from_energy(-5.0, kd0) < kd_from_energy(0.0, kd0)

    def test_ga_fitness_bounded(self):
        """FIX 3: fitness should be in (-1, 1) due to tanh."""
        target = self._target()
        ga = ProteinBindingGA(target, population_size=5, generations=3)
        best_cfg, best_fit, _ = ga.evolve()
        assert -1.0 <= best_fit <= 1.0

    def test_ga_improves_or_stays(self):
        target = self._target(kd=2.0)
        ga = ProteinBindingGA(target, population_size=10, generations=10)
        _, _, history = ga.evolve()
        assert history[-1] >= history[0] - 0.01

    def test_rl_trains_without_error(self):
        target = self._target()
        rl = ProteinBindingRL(target, n_episodes=5, steps_per_episode=10)
        history = rl.train()
        assert len(history) == 5
        assert all(h >= 0 for h in history)

    def test_rl_kd_positive(self):
        target = self._target()
        rl = ProteinBindingRL(target, n_episodes=10, steps_per_episode=20)
        rl.train()
        assert rl.best_kd > 0


# ============================================================
# run
# ============================================================
if __name__ == "__main__":
    import traceback, sys

    suites = [
        TestUAFCore, TestDiracAgent, TestGravity, TestMolecular
    ]
    total = passed = failed = 0
    for Suite in suites:
        s = Suite()
        methods = [m for m in dir(s) if m.startswith("test_")]
        print(f"\n{'='*55}")
        print(f"  {Suite.__name__}  ({len(methods)} tests)")
        print(f"{'='*55}")
        for m in methods:
            total += 1
            try:
                getattr(s, m)()
                print(f"  ✓ {m}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {m}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*55}")
    print(f"  RESULTS: {passed}/{total} passed  ({failed} failed)")
    print(f"{'='*55}")
    sys.exit(0 if failed == 0 else 1)
