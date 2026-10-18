# Nasal0

A prototype for evaluating nasal airway obstruction. It builds airway geometries (parametric tubes, and real patient airways from CT), estimates airflow resistance by region, and wraps the tools in a small-model agent loop that explains the results. Every language model stays at or under 33B parameters. Physics and decision rules are plain code, never a language model, so every number traces back to a formula or a simulation.

## At a glance

Built for the NVIDIA x Red Hat Small Models Hack; see [SUBMISSION.md](SUBMISSION.md) for the pitch and track mapping, and [PRESENTATION.md](PRESENTATION.md) for the demo script.

- **Real anatomy:** 130 NasalSeg CT airways, 126 simulated in OpenFOAM on NVIDIA Brev (median nasal resistance 0.146 Pa·s/cm³ at 150 Pa).
- **Fast surrogate:** under 0.2 ms per prediction, 16.3% mean side error (cross-validated) against 29.9% for "average patient".
- **Four agents** on NVIDIA Nemotron 3.5 Lightning 30B-A3B (3B active), served with vLLM. On 16 fixed probes (`nasal-agents --eval`), all 16 runs completed and 15 set the right case. 3 invalid tool calls out of 98 were caught by code and recovered, and every number the Explainer wrote either passed the code check or was replaced by the code template.
- **Virtual surgery, verified in OpenFOAM:** on 10 obstructed real airways, Workflow++ reached the best plan's drop for 10 of 10, the standard rule for 1 of 10.
- **Demo:** live at https://108646989dcf9d39d7.gradio.live; `nasal-demo`, a 3D resistance map of a real patient with what-if sliders and live agent steps.

```bash
pip install -e ".[dev,sim,surrogate,agents,patients,demo]"
pytest
```

## Design

### Agents (small language models)

All four agents run on one served model, [NVIDIA Nemotron 3.5 Lightning 30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) (NVFP4). It has **30B total parameters and 3B active per token**, so it meets the 33B limit. Each agent has its own instructions and may call only its own tools.

| Agent | Model | Tools it may call | Job |
| --- | --- | --- | --- |
| Case | Nemotron 3.5 Lightning 30B-A3B | `set_case`, `set_patient` | Picks a real NasalSeg patient by id, or turns a description into the three case parameters; code validates them and returns errors for the model to fix |
| Planner | Nemotron 3.5 Lightning 30B-A3B | `predict_resistance`, `formula_resistance`, `what_if`, `patient_resistance`, `patient_what_if`, `patient_surgery_options`, `finish` | Decides which physics tools answer the request; for surgery questions, scores every combination of turbinate reduction, valve repair and septoplasty on the patient's CT airway and attaches the OpenFOAM check where one was run |
| Critic | Nemotron 3.5 Lightning 30B-A3B | `review_results`, `check_parameters`, `report_flags` | Runs the code checks and picks which flags to show; it can cite only flag ids the checks produced |
| Explainer | Nemotron 3.5 Lightning 30B-A3B | none | Writes the plain-language summary; code rejects any number not in the tool results and retries once, then falls back to a template written by code |

The agent count stays at four: more agents mostly add latency and failure points. The decision rules in `nasal.decision` are unreviewed placeholders and are not wired into the agents, so the agents explain resistance and what-if changes but never recommend treatment.

### Tools (plain code, not language models)

| Tool | Choice | Status |
| --- | --- | --- |
| Geometry generator | Python + trimesh | **Done**: `nasal.geometry` |
| Formula baseline (friction + narrowing loss) | Python | **Done**: `nasal.resistance` |
| Case generation | Python + pandas | **Done**: `nasal.cases`, `nasal-generate` |
| Meshing | gmsh (Python API) | **Done**: `nasal.simulate.mesh_side` |
| Airflow ground truth | OpenFOAM v2606 (`opencfd/openfoam-run:2606` in Docker), simpleFoam | **Done**: `nasal.simulate`, `nasal-simulate` |
| Real patient airways | NasalSeg CT labels, voxel hex mesh written as an OpenFOAM polyMesh | **Done**: `nasal.anatomy`, `nasal-patients` |
| Fast stand-in model | PyTorch MLP (tubes), power law on measured geometry (patients) | **Done**: `nasal.surrogate`, `nasal-surrogate` |
| Decision rules (standard and improved) | Plain Python functions | **Prototype**: `nasal.decision` (tubes) and `nasal.decision.patient` (real airways, verified in OpenFOAM); thresholds are placeholders pending ENT review |
| Comparison and logging | pandas + JSON agent log | **Done**: `nasal-agents --log` |
| Demo | Gradio + Plotly | **Done**: `nasal.demo`, `nasal-demo` |
| Model serving | vLLM `v0.27.1` on NVIDIA Brev (1x H100), OpenAI-style endpoint | **Done**: see Agents usage |

## Workflow

1. **Generate cases**: the geometry tool makes a few hundred synthetic patients with known parameters. *(done)* Real airways from 130 NasalSeg CT scans are also built and measured. *(done; see Real patients)*
2. **Simulate**: mesh each shape and run OpenFOAM for resistance by region. This takes the longest, so it starts first. *(done: 106 idealized cases and 126 real patients)*
3. **Train the fast model** on the simulation results, recording its error and speedup. *(done; see Fast surrogate)*
4. **Wire up the agents**: the Case agent sets parameters, the Planner calls the fast model, the formula and what-if tools, the Critic checks the results and the Explainer writes the summary. *(done; the placeholder decision rules are not wired in yet)*
5. **Verify with real simulation**: for a sample of cases, run virtual surgery in OpenFOAM and compare both rules' plans against it. This produces the results table. *(done on 10 real airways; see Step 5)*
6. **Demo**: sliders change the anatomy, the resistance map updates, and agent steps appear live with surrogate error and speed. *(done for real patients; see Demo)*
7. **Doctor's check**: at least one ENT review of the rules.

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

**The baseline is uncalibrated.** Its absolute values are not expected to match measured noses. It exists to rank cases and to be checked against OpenFOAM (first comparison below). Any calibration will be fitted from simulation, never hand-tuned.

## Airflow ground truth (OpenFOAM)

`nasal.simulate` runs each side as its own domain:

- **Mesh:** gmsh lofts the same elliptical cross-sections as `nasal.geometry` and fills the tube with tetrahedra sized to the local tube width (6 across the narrow axis by default, about 60k to 120k cells per side). Patches are `inlet` (nostril), `outlet` (choana) and `wall`.
- **Solver:** OpenFOAM v2606 `simpleFoam` in the `opencfd/openfoam-run:2606` Docker image. The inlet total pressure ramps from 0 to 150 Pa over the first 300 iterations, and the outlet static pressure is 0. At 150 Pa the valve Reynolds number is around 10⁴, so the default model is k-omega SST with wall functions; `--turbulence laminar` is also available.
- **Region drops:** flux-weighted total pressure on planes at the region boundaries, whose x positions come from `REGION_LENGTH`. Region resistances are losses and sum to the side resistance. The choana value includes the kinetic energy leaving the outlet.
- **Output:** the same column names and units as `nasal.resistance.evaluate`, plus `{side}_cells`, `{side}_iterations`, `{side}_converged`, `{side}_unsteadiness` and `{side}_seconds` (wall-clock time to mesh and solve that side).

Residuals plateau instead of converging, because the valve jet is mildly unsteady. When the solver reaches 3000 iterations without meeting its residual targets (`converged` is false), the flow and plane fields are averaged over the last 300 iterations. `unsteadiness` is the flow's standard deviation over that window divided by its mean. It was at most 2% in the runs below.

### First comparison with the baseline

`nasal-simulate --nominal --extremes` runs the nominal case and the six cases in `cases.csv` (seed 0) with the lowest and highest value of each parameter. It printed these ratios (simulated / formula):

| Case | Valve ° | Turbinate | Septum | Total R sim | Total R formula | Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| nominal | 15.0 | 1.00 | +0.00 | 0.058 | 0.076 | 0.77 |
| case_0026 | 20.0 | 0.78 | +0.38 | 0.052 | 0.062 | 0.84 |
| case_0103 | 19.1 | 0.60 | −0.20 | 0.052 | 0.064 | 0.81 |
| case_0199 | 15.1 | 1.44 | +0.99 | 0.085 | 0.090 | 0.94 |
| case_0230 | 16.5 | 1.60 | +0.64 | 0.103 | 0.092 | 1.12 |
| case_0255 | 13.1 | 0.86 | −1.00 | 0.075 | 0.109 | 0.69 |
| case_0269 | 8.0 | 0.62 | +0.60 | 0.126 | 0.181 | 0.70 |

Resistance is in Pa·s/cm³. What this shows:

- **Ranking:** the formula ranks cases fairly well. The Spearman rank correlation is 0.89 on total resistance and 0.84 across the 14 sides.
- **Scale:** per-side ratios run from 0.65 to 1.28, so one constant factor would not calibrate the formula. It overestimates most sides and underestimates sides where the septum narrows a large turbinate.
- **Regions:** the formula charges the jet leaving the valve to the valve. In the simulation that loss happens downstream, so valve resistance is 0.15 to 0.6 times the formula's. Choana resistance is higher on every side, and turbinate resistance on all but one.

No constants have been fitted yet.

**Mesh sensitivity:** for the nominal left side, refining from 69k to 216k cells (`--cells-across 9`) changed the side resistance by 5% (0.116 to 0.122). The region split moved more: valve resistance fell by a third. Treat region values as approximate. For the narrowest side tested (case_0199 left), k-omega SST and laminar runs agreed within 1%.

**Run-to-run scatter:** gmsh's 3D mesher is not deterministic, so rerunning a case gives a slightly different mesh. Rerunning case_0269's left side (narrow valve, the least steady flow) changed its flow by 11% (407 to 453 cm³/s). Expect several percent of noise in single cases, more on narrow sides.

## Fast surrogate

`nasal.surrogate` is a small PyTorch network: 3 inputs, two hidden layers of 64, and 4 outputs, one log resistance per region. One network serves both sides, because the sides are mirror images; a side's input is how far the septum bends toward it. It was trained on the 90 cases simulated so far: the nominal case, five extremes outside the first 100 and 84 of the first 100 cases in `cases.csv` (seed 0). Every number below is cross-validated (5-fold): each case is predicted by a network that never saw it.

| Mean absolute error vs OpenFOAM | Surrogate | Formula baseline |
| --- | --- | --- |
| Total resistance | 7.8% | 23.1% |
| Side resistance | 9.2% | 23.4% |
| Region resistance | 27.2% | 209% |

- **Speed:** a prediction takes 0.15 ms on a laptop CPU. The median OpenFOAM case took 15 minutes of wall-clock time (both sides, 4 cores each), so the surrogate is about 6 million times faster.
- **Weak spots:** the worst side is off by 74% and region values remain rough (see the mesh notes above). The agents' Critic flags large disagreements between the surrogate and the formula.

## Real patients (NasalSeg)

`nasal.anatomy` builds real airways from [NasalSeg](https://zenodo.org/records/13893419) (130 CT scans with voxel labels, CC BY 4.0, DOI 10.1038/s41597-024-04176-1). These are real anatomy, but their flow is still a model: rigid walls, no mucosa, a single steady breath.

- **Labels:** by position in LPS coordinates, 1 and 2 are the right and left maxillary sinuses, 3 and 4 the right and left nasal cavities, and 5 the nasopharynx. `check_labels` verifies this for each scan. 3 scans (P047 to P049) fail the check; they are kept but marked `labels_ok = False`. Scans stored with a reversed axis are flipped on load.
- **Domain:** both nasal cavities plus the nasopharynx; the sinuses are left out. Each nostril inlet is a flat cut where that cavity's cross-section first reaches 30% of its median. The outlet is a flat cut where the nasopharynx is wide (half its widest axial section). The narrow lower tail of the label is cut off because it otherwise took most of the pressure drop, and rhinomanometry measures pressure in the nasopharynx.
- **Mesh:** every airway voxel (0.59 × 0.59 × 1.5 mm) becomes one hexahedral cell, written directly as an OpenFOAM polyMesh (median 65k cells). The mesh always passes `checkMesh` with zero non-orthogonality. The cost is stair-stepped walls.
- **Solver:** the same simpleFoam k-omega SST setup as the tubes, with both nostrils at 150 Pa total pressure and the outlet at 0. One run gives the left and right flows. Regions are positions along each cavity (0–20% vestibule, 20–35% valve, 35–85% turbinate, the rest plus the nasopharynx choana), not anatomical landmarks.

**Results:** 126 of 130 scans ran. 3 have a cavity or nasopharynx label that doesn't connect to the rest (P086, P098, P101), and P102 diverged twice. One more (P066) has a region boundary with no net flow and is left out of training, which leaves 125.

- **Resistance:** median total nasal resistance is 0.146 Pa·s/cm³, ranging from 0.078 to 1.28, at a median 1031 cm³/s.
- **Asymmetry:** 29 patients have one side over 1.5 times the other.
- **Steadiness:** flow varied by at most 0.8% over the averaging window.

**Patient surrogate:** each side is described by six measured numbers: volume, length, nostril area, narrowest and mean cross-section, and where the narrowest point is. A PyTorch MLP like the tube surrogate overfit these ~250 sides (27% mean and 1175% worst side error, cross-validated). A power law per region does better: ridge regression of log region resistance on log geometry (`PatientSurrogate`).

| Mean absolute error vs OpenFOAM (5-fold, 125 patients) | Power law | Average patient |
| --- | --- | --- |
| Total resistance | 14.4% | 24.5% |
| Side resistance | 16.3% | 29.9% |
| Region resistance | 35.0% | 41.3% |

It underpredicts the most obstructed sides (about 2.6 Pa·s/cm³ in OpenFOAM, predicted about 0.34), so the agents' Critic flags any patient where the surrogate and that patient's OpenFOAM result differ by more than 25%.

## Usage

```bash
pip install -e ".[dev]"
nasal-generate -n 300 --seed 0 --out data          # writes data/cases.csv
nasal-generate -n 300 --seed 0 --out data --stl    # also writes one STL per side per case
pytest
```

`data/cases.csv` has one row per case: the parameters, total and per-side resistance and flow, and resistance for each side and region.

Simulation needs Docker (running) and gmsh:

```bash
pip install -e ".[dev,sim]"
nasal-simulate --nominal --extremes             # nominal + min/max of each parameter -> data/simulated.csv
nasal-simulate --first 20 --procs 8             # first 20 cases; cases already in the output are skipped
nasal-simulate --ids case_0003 --cells-across 9 --turbulence laminar
```

`data/simulated.csv` has one row per simulated case with the same columns as `cases.csv`, plus `turbulence` and the solver checks. OpenFOAM case folders and logs go to `data/sim/<case_id>/<side>/`. Each side takes about 3 to 5 minutes on 8 cores. The command ends by printing simulated / formula ratios for every column. Tests that need OpenFOAM are skipped when Docker is not available, and the mesh tests are skipped without gmsh.

To use more cores, run several batches at once with separate `--out` files and train on all of them. Keep the total of `--procs` across batches at or below the number of physical cores. The runs call `mpirun --bind-to none`: by default each container pins its processes to the same first cores, which made parallel batches about ten times slower.

The surrogate needs PyTorch:

```bash
pip install -e ".[dev,surrogate]"
nasal-surrogate --sim data/simulated.csv data/simulated_A.csv data/simulated_B.csv
```

It merges the files by `case_id`, prints the cross-validated error and speedup, and writes `data/surrogate.pt` and `data/surrogate_report.json`. Its tests are skipped without PyTorch.

### Real patients

```bash
pip install -e ".[sim,patients]"
# download NasalSeg.zip from https://zenodo.org/records/13893419 and unzip it to nasalseg/
nasal-patients measure --labels nasalseg/labels                          # geometry of every patient -> data/patients.csv
nasal-patients simulate --labels nasalseg/labels --ids P001 --procs 8    # OpenFOAM -> data/patient_sim.csv
nasal-surrogate --patients --sim data/patient_sim.csv                     # -> data/patient_surrogate.json
```

`simulate` skips patients already in its output, so it can resume, and logs a failed patient and moves on. Each patient takes several minutes on 4 cores (we did not time them individually). We ran 12 batches of `--procs 4` on a 48-core Brev instance, each with its own `--out`, and trained on all the files together. With `data/patient_sim.csv` and `data/patient_surrogate.json` present, `nasal-agents` also accepts requests about a patient by id (for example "patient P022: what if the left narrowest section were 30% wider?").

### Demo

```bash
pip install -e ".[demo]"
nasal-demo --labels nasalseg/labels    # needs data/patient_sim.csv and data/patient_surrogate.json
nasal-demo --labels nasalseg/labels --share    # also prints a public gradio.live link (valid 72 h; keep this machine and the vLLM tunnel up)
```

The demo opens at http://127.0.0.1:7860.

- **Patient:** pick any simulated NasalSeg patient (it opens on P022, whose left side has three times the right's resistance).
- **3D airway:** an interactive Plotly view colored by each region's resistance, from that patient's OpenFOAM run or from the surrogate.
- **What-if:** sliders widen or narrow each side's narrowest section and recolor instantly through the surrogate.
- **Table:** OpenFOAM against the surrogate for every region, with the surrogate's error and its measured prediction time.
- **Agents:** "Run agents" sends a question to the four agents and lists each step as it happens: tool calls, errors sent back, the code number check. Then it shows the summary and flags. The agents need the vLLM endpoint (see Agents); everything else works offline.

### Agents

Serve the model on a GPU machine. We used an NVIDIA Brev `scaleway_H100` instance with the flags from the model card's H100 recipe, minus speculative decoding:

```bash
docker run -d --name nemotron --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface vllm/vllm-openai:v0.27.1 \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --served-model-name nemotron \
  --max-model-len 65536 --max-num-seqs 64 --enable-prefix-caching --async-scheduling \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder --enable-auto-tool-choice
```

Then forward port 8000 to your machine (for example `ssh -N -L 8000:localhost:8000 <instance>`) and run:

```bash
pip install -e ".[agents,surrogate]"
nasal-agents "Narrow valve, maybe 9 degrees, septum bends left. What if the valve were widened to 15 degrees?"
```

It prints the summary and writes every model turn, tool call, tool result and check to `data/agent_log.json`.

`nasal-agents --eval` runs 16 fixed probes (`nasal.agent_eval`), scores them in code and writes `data/agent_eval.json`. The probes include patient ids, a what-if, an unknown patient, an out-of-range valve angle and plain descriptions. Our run against the Brev endpoint:

- **Runs:** 16/16 completed, with a median of 5.9 s per four-agent run.
- **Case choice:** 15/16 set the right case. The miss: "septum to the right 0.5" became +0.5, which means the left in this project, and no code check can catch a wrong sign.
- **What-ifs:** 6/6 run when the request asked for one.
- **Tool calls:** 3 invalid out of 98, all caught by code and sent back; the model then recovered to a valid case.
- **Explainer:** 14/16 passed the number check the first time; the other 2 were replaced by the code template. `NASAL_LLM_URL` and `NASAL_LLM_MODEL` point it at another OpenAI-compatible server. `--thinking` turns on the model's reasoning mode, which is off by default for speed. The predictions use `data/surrogate.pt` when it exists and the formula otherwise; the Critic flags which one was used. The agent tests use a scripted stand-in for the model and need no server.

## Surgical decision rules (Workflow++ prototype)

`nasal.decision` compares the standard visual rule with a map-driven rule, scored by virtual surgery. Thresholds are **placeholders, not clinical values**, and no ENT has reviewed them yet.

- **Virtual surgery** (`surgery.py`): turbinate reduction shrinks `turbinate_scale` by 30%, septoplasty sets `septum_deviation` to 0, and valve repair widens `valve_angle_deg` by 4°. Results are clamped to the parameter ranges.
- **Standard rule**: turbinate reduction, plus septoplasty when `|septum_deviation|` > 0.3.
- **Workflow++ rule**: run every combination of procedures through the resistance evaluator, then pick the smallest plan that reaches 90% of the best total-resistance drop.
- **Reclassification**: `python -m nasal.decision --cases data/cases.csv --out data/decisions.csv` writes one row per case with both plans, whether the plan changed, and the resistance drop for each, then prints a summary by dominant region.

- **Suggested course for one case** (`recommend.py`): `python -m nasal.decision.recommend --valve-angle-deg 9 --turbinate-scale 1.3 --septum-deviation 0.6` prints JSON with the regional map before surgery, the Workflow++ suggestion with a code-written reason for each procedure, the standard plan for comparison, every option ranked by modeled resistance drop, and what is not modeled. It uses the trained surrogate (`data/surrogate.pt`) when present, otherwise the formula. It is not wired into `nasal.agents`.

The evaluator is pluggable (`evaluator=`). It defaults to the uncalibrated formula baseline, so the reclassification rates it prints describe the baseline, not real noses. The surrogate or OpenFOAM can be passed in once they are available. The baseline has no inferior/middle/superior flow split or wall shear yet, so the Workflow++ plan's flow-balance and ENS-risk limits are not implemented.

### Step 5: virtual surgery on real airways, verified in OpenFOAM

`nasal.decision.patient` applies the procedures to a patient's voxel airway. Each procedure adds or moves one voxel layer (about 0.6 mm). These are modeling choices, not surgical prescriptions.

- **Valve repair:** widens both valve regions (20–35% of each cavity's length) sideways and vertically.
- **Turbinate reduction:** pushes both cavities' lateral wall outward in the turbinate region (35–85%).
- **Septoplasty:** removes the wider cavity's medial layer and grows the narrower one toward it.
- **Limits:** growth never touches the other cavity or crosses the nostril or outlet cuts, and the airway is kept as one connected piece.

Both rules are applied per patient:

- **Standard rule:** turbinate reduction, plus septoplasty when one side's mean cross-section is over 1.3 times the other's (a placeholder threshold).
- **Workflow++:** scores all 7 plans with the patient surrogate on the operated airway's measured geometry.

`nasal-patients surgery --labels nasalseg/labels --ids P022 --procs 4` then runs OpenFOAM on the untouched airway and on all 7 plans, and prints how each rule's choice compares.

We ran the 12 most obstructed patients whose labels pass the position check. 2 pairs turned out to be duplicate scans in NasalSeg (P009/P010 and P080/P081 have identical geometry), so the results below cover 10 distinct airways, 8 OpenFOAM runs each:

| Plan chosen by | Mean OpenFOAM resistance drop | Within 90% of the best plan's drop |
| --- | --- | --- |
| Workflow++ (surrogate) | 10.2% | 10 of 10 (the best plan itself in 8) |
| Standard rule | 7.2% | 1 of 10 |
| Best plan per patient (OpenFOAM) | 10.2% | |

- **Best plan:** turbinate reduction with valve repair was best or near-best for every patient (mean drop 10.2%, up to 26.7%). Turbinate reduction alone averaged 7.8% and valve repair alone 1.9%.
- **Septoplasty raised resistance** in this model (mean −7.9% alone). The voxel version removes more airway than it adds, so this is an artifact of the model, not a finding about septoplasty. Workflow++ never chose it; the standard rule chose it for 2 patients.
- **The surrogate ranks plans better than it sizes them:** its predicted drops correlate only 0.47 with OpenFOAM's, with a mean error of 6.1 percentage points. It still picked the right plan, because the plans' differences were large.
- **Limits:** this is a 10-patient sample of the most obstructed airways, with one-layer changes, steady flow and rigid walls. The rules' thresholds are unreviewed placeholders.

## Keeping docs in sync

Sections between `<!-- name:start -->` / `<!-- name:end -->` markers are generated from the code. After changing the parameters, run:

```bash
python -m nasal.docs
```

The test suite fails if these sections are stale. See [CLAUDE.md](CLAUDE.md) for the full consistency rules.

## License

TBD.
