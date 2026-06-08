"""
UAF v4 — Extended Tests (GAP 1-6)
===================================
Run: python tests/test_extensions.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import traceback


# ============================================================
# 1. SDE (GAP 1)
# ============================================================
from uaf.core import UAFParams, UAFState
from uaf.sde import UAFStochasticSystem, SDEParams

class TestSDE:

    def test_state_stays_bounded(self):
        sys = UAFStochasticSystem(
            UAFParams(n_levels=3),
            SDEParams(sigma_gamma=0.2, sigma_alpha=0.1),
            seed=1
        )
        sys.run(steps=500, dtau=0.005)
        g = np.array([s.gamma for s in sys.history])
        a = np.array([s.alpha for s in sys.history])
        assert np.all(g >= 0) and np.all(g <= 1), "SDE Γ out of bounds"
        assert np.all(a >= 0) and np.all(a <= 1), "SDE α out of bounds"

    def test_noise_increases_variance_vs_deterministic(self):
        """Stochastic system should have more variance than deterministic."""
        from uaf.core import UAFSystem
        p = UAFParams(n_levels=2)

        det = UAFSystem(p)
        det.run(steps=300, dtau=0.005)
        det_g = det.gamma_trajectory()

        stoch = UAFStochasticSystem(p, SDEParams(sigma_gamma=0.1), seed=42)
        stoch.run(steps=300, dtau=0.005)
        stoch_g = np.array([s.gamma for s in stoch.history])

        assert np.var(stoch_g) > np.var(det_g), \
            "Stochastic should have higher variance"

    def test_ensemble_shape(self):
        sys = UAFStochasticSystem(UAFParams(n_levels=2), seed=0)
        ens = sys.ensemble(n_runs=5, steps=50, dtau=0.005)
        assert ens.shape == (5, 51, 2, 2), f"Wrong shape: {ens.shape}"

    def test_coherence_uncertainty_returns_dict(self):
        sys = UAFStochasticSystem(UAFParams(n_levels=2), seed=0)
        result = sys.coherence_uncertainty(n_runs=10, steps=50)
        assert "mean" in result
        assert "std" in result
        assert 0.0 <= result["mean"] <= 1.0
        assert result["std"] >= 0.0

    def test_bifurcation_scan(self):
        sys = UAFStochasticSystem(UAFParams(n_levels=2), seed=0)
        sigmas = np.linspace(0.01, 0.3, 4)
        results = sys.scan_noise_bifurcation(sigmas, n_runs=5, steps=50)
        assert len(results) == 4
        # variance (susceptibility) should be non-negative
        assert all(r["var_gamma"] >= 0 for r in results)

    def test_bimodality_high_noise(self):
        """Very high noise may produce bimodal distribution."""
        sys = UAFStochasticSystem(
            UAFParams(n_levels=2, eta=0.1, lam=2.0),
            SDEParams(sigma_gamma=0.3), seed=0
        )
        ens = sys.ensemble(n_runs=50, steps=200)
        final = ens[:, -1, 0, 0]
        # just check it doesn't crash and returns a float
        from uaf.sde import _bimodality_index
        bc = _bimodality_index(final)
        assert isinstance(bc, float)
        assert bc >= 0.0


# ============================================================
# 2. FEP (GAP 3 + 4)
# ============================================================
from uaf.fep import (
    Belief, GenerativeModel, variational_free_energy, infer,
    Policy, expected_free_energy, UAFAgent
)

class TestFEP:

    def _setup(self):
        model = GenerativeModel(n_states=3, n_obs=2, obs_sigma=0.1, seed=0)
        return model

    def test_belief_entropy_decreases_after_inference(self):
        """After seeing data, belief should become more certain (lower entropy)."""
        model = self._setup()
        obs = model.A @ np.array([0.5, -0.3, 0.8])

        prior = Belief.uniform(3)
        h_prior = prior.entropy()

        posterior, _ = infer(prior, obs, model, steps=30)
        h_post = posterior.entropy()

        assert h_post < h_prior, \
            f"Entropy didn't decrease: {h_prior:.3f} → {h_post:.3f}"

    def test_free_energy_decreases_with_inference(self):
        """VFE should decrease as belief is updated."""
        model = self._setup()
        obs = np.array([0.5, -0.3])
        belief = Belief.uniform(3)
        _, fe_history = infer(belief, obs, model, steps=20)
        # should generally trend downward
        assert fe_history[-1] < fe_history[0], \
            "Free energy didn't decrease during inference"

    def test_fe_breakdown_non_negative_complexity(self):
        model = self._setup()
        obs = np.array([0.3, 0.7])
        belief = Belief(mu=np.array([0.5, 0.5, 0.5]),
                        log_sigma=np.zeros(3))
        fe = variational_free_energy(belief, obs, model,
                                     rng=np.random.default_rng(0))
        assert fe["complexity"] >= 0.0, "KL divergence must be non-negative"

    def test_elbo_equals_neg_F(self):
        model = self._setup()
        obs = np.array([0.1, 0.2])
        belief = Belief.uniform(3)
        fe = variational_free_energy(belief, obs, model,
                                     rng=np.random.default_rng(0))
        assert abs(fe["ELBO"] + fe["F"]) < 1e-9

    def test_prediction_error_decreases_after_inference(self):
        """PE should decrease as we update beliefs toward observations."""
        model = self._setup()
        s_true = np.array([1.0, 0.0, -1.0])
        obs = model.A @ s_true

        prior = Belief.uniform(3)
        pe_prior = model.prediction_error(obs, prior.mu)

        posterior, _ = infer(prior, obs, model, steps=50)
        pe_post = model.prediction_error(obs, posterior.mu)

        assert np.linalg.norm(pe_post) < np.linalg.norm(pe_prior), \
            "PE didn't decrease after inference"

    def test_agent_step_returns_dict(self):
        model = self._setup()
        agent = UAFAgent(model, n_policies=3, seed=0)
        obs = np.array([0.5, -0.5])
        result = agent.step(obs)
        required = {"F", "complexity", "accuracy", "ELBO",
                    "best_G", "action", "belief_mu"}
        for k in required:
            assert k in result, f"Missing key: {k}"

    def test_agent_free_energy_trajectory(self):
        model = self._setup()
        agent = UAFAgent(model, seed=0)
        rng = np.random.default_rng(7)
        obs_seq = rng.standard_normal((20, 2))
        agent.run(obs_seq)
        F_traj = agent.free_energy_trajectory()
        assert len(F_traj) == 20
        assert not np.any(np.isnan(F_traj))

    def test_expected_free_energy_has_epistemic_pragmatic(self):
        model = self._setup()
        belief = Belief.uniform(3)
        policy = Policy(actions=np.zeros((3, 3)))
        result = expected_free_energy(policy, belief, model,
                                      rng=np.random.default_rng(0))
        assert "epistemic_value" in result
        assert "pragmatic_value" in result
        assert "G" in result

    def test_preferred_obs_reduces_G(self):
        """Agent with preferred obs should achieve lower avg G over sequence."""
        model = GenerativeModel(n_states=3, n_obs=2, obs_sigma=0.2, seed=0)
        preferred = np.array([1.0, 0.0])
        # observations close to preferred
        rng = np.random.default_rng(5)
        obs_seq = preferred + rng.standard_normal((15, 2)) * 0.1

        agent_pref = UAFAgent(model, preferred_obs=preferred, n_policies=8, seed=0)
        agent_free = UAFAgent(model, preferred_obs=None, n_policies=8, seed=0)

        res_pref = agent_pref.run(obs_seq)
        res_free = agent_free.run(obs_seq)

        avg_G_pref = np.mean([r["best_G"] for r in res_pref[-5:]])
        avg_G_free = np.mean([r["best_G"] for r in res_free[-5:]])
        # preferred agent converges to lower G when observations match preference
        assert avg_G_pref < avg_G_free, \
            f"Preferred agent G={avg_G_pref:.3f} should be < free G={avg_G_free:.3f}" 


# ============================================================
# 3. Information Metrics (GAP 6)
# ============================================================
from uaf.metrics import (
    surprisal, complexity, mutual_information_matrix,
    integrated_information, coherence_bandwidth, time_horizon,
    information_report
)
from uaf.core import UAFSystem

class TestMetrics:

    def _traj(self, steps=200):
        sys = UAFSystem(UAFParams(n_levels=3))
        sys.run(steps=steps, dtau=0.01)
        g = sys.gamma_trajectory()
        a = sys.alpha_trajectory()
        return g, a

    def test_surprisal_decreases_with_gamma(self):
        g_high = np.array([0.9, 0.8])
        g_low  = np.array([0.1, 0.2])
        assert np.all(surprisal(g_high, g_high) < surprisal(g_low, g_low))

    def test_complexity_zero_at_half(self):
        a = np.array([0.5, 0.5, 0.5])
        assert abs(complexity(a)) < 1e-6

    def test_complexity_high_at_extremes(self):
        a_extreme = np.array([0.01, 0.99])
        a_mid     = np.array([0.5, 0.5])
        assert complexity(a_extreme) > complexity(a_mid)

    def test_mi_matrix_symmetric(self):
        g, _ = self._traj()
        M = mutual_information_matrix(g, n_bins=10)
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_mi_matrix_diagonal_zero(self):
        g, _ = self._traj()
        M = mutual_information_matrix(g, n_bins=10)
        assert np.all(M.diagonal() == 0)

    def test_mi_non_negative(self):
        g, _ = self._traj()
        M = mutual_information_matrix(g)
        assert np.all(M >= 0)

    def test_integrated_information_non_negative(self):
        g, _ = self._traj()
        phi = integrated_information(g)
        assert phi >= 0

    def test_coherence_bandwidth_in_0_1(self):
        g, _ = self._traj()
        bw = coherence_bandwidth(g, level=0, threshold=0.5)
        assert 0.0 <= bw <= 1.0

    def test_time_horizon_positive(self):
        g, _ = self._traj(steps=300)
        tau = time_horizon(g)
        assert tau >= 0.0

    def test_information_report_structure(self):
        g, a = self._traj()
        report = information_report(g, a)
        required = {
            "total_surprisal", "complexity",
            "mutual_information_matrix",
            "integrated_information_phi",
            "time_horizon_tau",
            "interpretation"
        }
        for k in required:
            assert k in report, f"Missing key: {k}"

    def test_high_eta_lowers_surprisal(self):
        """System with high η should maintain lower surprisal."""
        def final_surprisal(eta):
            sys = UAFSystem(UAFParams(n_levels=3, eta=eta, lam=0.1))
            sys.run(steps=200, dtau=0.005)
            from uaf.metrics import total_surprisal
            return total_surprisal(sys.state.gamma, sys.state.alpha)

        assert final_surprisal(2.0) < final_surprisal(0.1), \
            "High η should reduce surprisal"


# ============================================================
# 4. Markov Blanket (GAP 2)
# ============================================================
from uaf.markov_blanket import (
    MarkovBlanketState, UAFHierarchyMB, detect_emergent_blankets
)

class TestMarkovBlanket:

    def test_blanket_state_initialises(self):
        s = MarkovBlanketState(level=0)
        assert 0.0 <= s.gamma <= 1.0
        assert 0.0 <= s.alpha <= 1.0

    def test_integrity_in_0_1(self):
        for alpha in [0.0, 0.3, 0.7, 1.0]:
            s = MarkovBlanketState(level=0, alpha=alpha)
            assert 0.0 <= s.markov_blanket_integrity() <= 1.0

    def test_hierarchy_runs(self):
        h = UAFHierarchyMB(n_levels=3)
        h.run(steps=100, dtau=0.01)
        assert len(h.history) == 101

    def test_gamma_stays_bounded_mb(self):
        h = UAFHierarchyMB(n_levels=4, coupling=0.8)
        h.run(steps=200, dtau=0.01)
        for snap in h.history:
            for s in snap:
                assert 0.0 <= s.gamma <= 1.0, f"Γ={s.gamma} out of [0,1]"
                assert 0.0 <= s.alpha <= 1.0, f"α={s.alpha} out of [0,1]"

    def test_external_signal_propagates(self):
        """Signal at level 0 should propagate upward."""
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(200) * 2.0   # strong signal

        h_signal = UAFHierarchyMB(n_levels=3)
        h_signal.run(steps=200, dtau=0.01, external_signal=signal)

        h_quiet = UAFHierarchyMB(n_levels=3)
        h_quiet.run(steps=200, dtau=0.01, external_signal=None)

        # Top level should differ
        g_signal = h_signal.history[-1][-1].gamma
        g_quiet  = h_quiet.history[-1][-1].gamma
        assert g_signal != g_quiet, "External signal had no effect on top level"

    def test_integrity_trajectory_shape(self):
        h = UAFHierarchyMB(n_levels=3)
        h.run(steps=50)
        traj = h.blanket_integrity_trajectory()
        assert traj.shape == (51, 3)

    def test_emergent_blankets_single_cluster(self):
        """Fully connected → one blanket."""
        C = np.ones((4, 4)) - np.eye(4)
        blankets = detect_emergent_blankets(C, threshold=0.5)
        assert len(blankets) == 1
        assert len(blankets[0]) == 4

    def test_emergent_blankets_two_clusters(self):
        """Block diagonal coupling → two blankets."""
        C = np.zeros((4, 4))
        C[0, 1] = C[1, 0] = 0.8
        C[2, 3] = C[3, 2] = 0.8
        blankets = detect_emergent_blankets(C, threshold=0.5)
        assert len(blankets) == 2


# ============================================================
# Runner
# ============================================================
if __name__ == "__main__":
    suites = [TestSDE, TestFEP, TestMetrics, TestMarkovBlanket]
    total = passed = failed = 0
    for Suite in suites:
        s = Suite()
        methods = [m for m in dir(s) if m.startswith("test_")]
        print(f"\n{'='*60}")
        print(f"  {Suite.__name__}  ({len(methods)} tests)")
        print(f"{'='*60}")
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
    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed  ({failed} failed)")
    print(f"{'='*60}")
    import sys; sys.exit(0 if failed == 0 else 1)
