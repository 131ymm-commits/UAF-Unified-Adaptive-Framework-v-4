# UAF v4 — Unified Adaptive Framework

**Version 4.0 · June 2026**  
Unified codebase merging UAF-3.1, UAF-Quantum, uaf-dirac-anomaly, uaf-molecular-optimization, XGrav.

---

## Core Idea

Every adaptive system — physical, biological, computational — evolves according to a single principle: **coherence of future possibilities decays through integration and grows through cross-level correlation flux.**

The master equation for level *l* in a hierarchy:

$$\frac{\partial \Gamma_l}{\partial \tau} = \eta \sum_{k<l} \Phi_{k\to l} \cdot \mathcal{E}_{k,l} \cdot (1-\alpha_l)^2 \;-\; \lambda \cdot \Delta S_\text{env} \cdot \alpha_l \cdot \Gamma_l$$

| Symbol | Meaning |
|--------|---------|
| $\Gamma_l$ | Future coherence at level $l$ — how much open possibility remains |
| $\tau$ | Internal relational time (not an external clock) |
| $\Phi_{k\to l}$ | Correlation flux from lower level $k$ to $l$ |
| $\mathcal{E}_{k,l}$ | Entanglement / coupling strength between levels |
| $\alpha_l$ | Integration degree: 0 = fully open, 1 = fully crystallised |
| $\eta$ | Amplification (energy, trust, technology) |
| $\lambda$ | Decoherence rate |
| $\Delta S_\text{env}$ | Entropy growth of surroundings |

**Time tripartition:**  `t_past + t_now + t_future = const`
- `t_past` = crystallised past → mass, memory, inertia
- `t_now` = active coherence → present flow, processing
- `t_future` = open possibility → Γ, potential, freedom

---

## Repository Structure

```
UAF-v4/
├── uaf/
│   └── core.py            ← UAF ODE engine (RK4, all levels)
│
├── apps/
│   ├── anomaly/
│   │   └── dirac_agent.py ← Online anomaly detector (Dirac spinor)
│   ├── gravity/
│   │   └── nbody.py       ← N-body + evolutionary search (XGrav)
│   └── molecular/
│       └── binding.py     ← Protein binding GA + RL
│
├── tests/
│   └── test_all.py        ← 24 tests, all pass
│
└── README.md
```

---

## Quick Start

```bash
pip install numpy scipy
python tests/test_all.py          # 24/24 ✓
```

### Core UAF simulation

```python
from uaf.core import UAFSystem, UAFParams

sys = UAFSystem(UAFParams(n_levels=3, eta=1.5, lam=0.3))
traj = sys.run(steps=500, dtau=0.01, delta_S_env=0.1)
print(f"Final coherence: {sys.coherence():.3f}")
print(f"Time arrow: {sys.time_arrow_strength():.3f}")
```

### Anomaly detection

```python
from apps.anomaly.dirac_agent import UAFDiracAgent
import numpy as np

agent = UAFDiracAgent()
data = np.sin(np.linspace(0, 20*np.pi, 1000)) + 0.05 * np.random.randn(1000)
data[600:620] += 8.0   # inject anomaly

for t, v in enumerate(data):
    event = agent.update(t, v)
    if event:
        print(f"t={t}  score={event['score']:.3f}  z={event['z']:.2f}")
```

### Gravity search (XGrav)

```python
from apps.gravity.nbody import UAFGravityEvolution, EvolutionConfig

cfg = EvolutionConfig(n_asteroids=20, n_shepherds=2, generations=20)
evo = UAFGravityEvolution(cfg)
best_config, best_gamma, history = evo.evolve()
print(f"Best Γ = {best_gamma:.4f}")
```

### Molecular optimisation

```python
from apps.molecular.binding import ProteinTarget, ProteinBindingGA

target = ProteinTarget.synthetic("1HPV", kd=0.20)
ga = ProteinBindingGA(target, generations=50)
best_cfg, best_fitness, history = ga.evolve()
```

---

## What Changed from Previous Repos

### UAF-3.1 / UAF-Quantum
- `demo.py` used `numba @jit` on Python objects → **removed**, pure numpy is faster in practice here
- Euler integration of ODE → **replaced with RK4** (4th order, stable)
- `UAF-Quantum/main.py` was **identical** to `XGrav/main.py` (copy-paste bug) → fixed
- Γ was not clamped → could go negative → **clamped to [0, 1]**

### uaf-dirac-anomaly
- `_normalized_history()` was O(N²) per step → **O(1) incremental running stats**
- Spinor norm collapse → division by EPS → garbage direction → **reinitialise to (1,0)**
- Z-score baseline included current point (look-ahead) → **fixed to past-only window**
- Anomaly weights didn't enforce sum=1 → **normalised on init**
- `score_sequence` called `reset()` mid-stream → **proper isolated reset**

### uaf-molecular-optimization
- README had fabricated licensing figures ($10M–$5B) → **removed entirely**
- `uaf_dirac_agent.py` was duplicated → **now imported from shared package**
- GA fitness was unbounded `baseline/predicted` → **replaced with tanh-normalised**
- "RL agent" was a random policy → **replaced with real Q-table epsilon-greedy**

### XGrav
- Euler N-body integration (non-symplectic, energy-leaking) → **Leapfrog (Störmer–Verlet)**
- UAF Γ was a placeholder → **real coherence from trajectory variance ratio**

### UAF-Economic-Theory-UET
- Not included in this codebase. The conceptual mapping (money = t_future contracts, NFTI index) is described in `docs/economics_note.md` but the normative/political claims are separated from the physics.

---

## Tests

```
TestUAFCore      7/7  ✓   bounds, RK4, time arrow, eta scaling
TestDiracAgent   6/6  ✓   false positives, spike detection, NaN safety
TestGravity      5/5  ✓   energy conservation, coherence bounds, evolution
TestMolecular    6/6  ✓   VDW physics, fitness bounds, GA/RL correctness
────────────────────────
TOTAL           24/24 ✓
```

---

## Theoretical Connections

| Domain | UAF mapping |
|--------|-------------|
| Quantum mechanics | Γ ↔ von Neumann coherence; α ↔ decoherence via Lindblad |
| Thermodynamics | ΔS_env drives decay term; time arrow = decay dominance |
| General Relativity | mass = high t_past; gravity = past's influence on metric |
| Anomaly detection | Dirac mass gap ↔ regime boundary; gap breach = anomaly |
| Orbital mechanics | Γ = orbital stability; chaos → Γ collapse |
| Molecular binding | Γ of ligand-receptor complex ↔ binding specificity |
| Page-Wootters | τ from correlations, not external clock — same principle |

---

## License

MIT
