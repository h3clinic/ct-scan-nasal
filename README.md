# ct-scan-nasal

> **Research only. Not medical advice.** Nothing here is validated for clinical use. The geometry is idealized and synthetic, not derived from patient scans.

A prototype for evaluating nasal airway obstruction. It builds parametric airway geometries, estimates airflow resistance by region, and (planned) wraps the tools in a small-model agent loop that explains a recommendation. Every language model stays at or under 33B parameters. Physics and decision rules are plain code, never a language model, so every number traces back to a formula or a simulation.

## Design

### Agents (small language models, planned)

| Agent | Size | Job |
| --- | --- | --- |
| Planner | ~14B | Reads the request and decides which tools to call, in what order |
| Case | 7B to 14B | Turns a description or measurements into geometry parameters |
| Critic | ~8B | Flags out-of-range inputs, disagreement between rules, and risk flags before anything is shown |
| Explainer | ~14B | Writes the plain-language recommendation from the numbers it is given; never invents numbers |

The agent count stays at four: more agents mostly add latency and failure points.

### Tools (plain code, not language models)

| Tool | Choice | Status |
| --- | --- | --- |
| Geometry generator | Python + trimesh | **Done**: `nasal.geometry` |
| Formula baseline (friction + narrowing loss) | Python | **Done**: `nasal.resistance` |
| Case generation | Python + pandas | **Done**: `nasal.cases`, `nasal-generate` |
| Meshing | gmsh | Planned |
| Airflow ground truth | OpenFOAM (via Docker) | Planned |
| Fast stand-in model | PyTorch, or NVIDIA PhysicsNeMo | Planned |
| Decision rules (standard and improved) | Plain Python functions | Planned, pending ENT input |
| Comparison and logging | pandas + JSON agent log | Planned |
| Demo | Streamlit or Gradio with PyVista or Plotly | Planned |
| Model serving | vLLM or TensorRT-LLM (OpenAI-style endpoint) | Planned |

## Workflow

1. **Generate cases**: the geometry tool makes a few hundred synthetic patients with known parameters. *(done)*
2. **Simulate**: mesh each shape and run OpenFOAM for resistance by region. This takes the longest, so it starts first.
3. **Train the fast model** on the simulation results, recording its error and speedup.
4. **Wire up the agents**: the Planner receives the case, the Case agent sets parameters, and the Planner calls the fast model and both decision rules. The Critic checks the results and the Explainer writes the recommendation.
5. **Verify with real simulation**: for a sample of cases, run virtual surgery in OpenFOAM and compare both rules' plans against it. This produces the results table.
6. **Demo**: sliders change the anatomy, the resistance map updates, and agent steps appear live with surrogate error and speed.
7. **Doctor's check**: at least one ENT review of the rules, plus the research-only label.

## Case parameters

<!-- params:start -->
| Parameter | Range | Unit | Meaning |
| --- | --- | --- | --- |
| `valve_angle_deg` | 8 to 20 | deg | Internal nasal valve angle (both sides) |
| `turbinate_scale` | 0.6 to 1.6 | - | Inferior turbinate size; 1.0 is the nominal size |
| `septum_deviation` | -1 to 1 | - | Septum bend; positive narrows the left side, negative the right |
<!-- params:end -->

## Geometry and baseline model

Each nasal side is a tube along the airway with four regions: vestibule, valve, turbinate and choana. Each region has its own cross-sectional area and elliptical aspect ratio. The turbinate region is slit-like, so it has a high aspect ratio.

- **Valve area** is a triangle with apex angle `valve_angle_deg`.
- **Turbinate open area** shrinks as `turbinate_scale` grows.
- **Septum deviation** narrows the valve and turbinate on the side it bends into, and widens them on the other side.

The baseline treats each side's pressure drop as `a*Q + b*Q^2` (Rohrer form):

- **Linear term:** laminar friction in an elliptical duct.
- **Quadratic term:** losses at the nostril entry, at each area change (contraction or sudden expansion), and at the choanal exit.

Each area-change loss counts toward the narrower of the two regions, because that constriction is what causes it. For example, the jet leaving a narrow valve counts against the valve, not the turbinate cavity it expands into. The two sides are in parallel. Resistance is reported in Pa·s/cm³ at the standard 150 Pa rhinomanometry reference pressure.

**The baseline is uncalibrated.** Its absolute values are not expected to match measured noses. It exists to rank cases and to be checked against OpenFOAM in step 5. Any calibration will be fitted from simulation, never hand-tuned.

## Usage

```bash
pip install -e ".[dev]"
nasal-generate -n 300 --seed 0 --out data          # writes data/cases.csv
nasal-generate -n 300 --seed 0 --out data --stl    # also writes one STL per side per case
pytest
```

`data/cases.csv` has one row per case: the parameters, total and per-side resistance and flow, and resistance for each side and region.

## Keeping docs in sync

Sections between `<!-- name:start -->` / `<!-- name:end -->` markers are generated from the code. After changing the parameters, run:

```bash
python -m nasal.docs
```

The test suite fails if these sections are stale. See [CLAUDE.md](CLAUDE.md) for the full consistency rules.

## License

TBD.
