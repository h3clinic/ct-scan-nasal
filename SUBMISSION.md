# Nasal0: NVIDIA x Red Hat Small Models Hack submission

**Note:** the agents now run on OpenAI's API instead of an open small model, so this version no longer meets the hackathon's small-model rules.



A blocked nose is usually judged by eye: a surgeon looks at a CT scan, sees a bent septum or big turbinates, and operates on what looks wrong. **Nasal0** asks where the air actually loses pressure.

- **Real airways:** it builds airways from 130 real CT scans (NasalSeg) and simulates the airflow in OpenFOAM.
- **Fast stand-in:** a surrogate fitted to those simulations answers in a fraction of a millisecond, compared with minutes for a simulation.
- **Agents:** four agents on **OpenAI's API** (`gpt-4.1-mini` by default) let you ask about a patient in plain language. Every number comes from code: the model chooses tools and explains, and code checks every number it writes. For real patients, those numbers come from OpenFOAM.

When we tested virtual surgery on 10 obstructed real airways, the map-driven plan (Workflow++) matched the best plan found by OpenFOAM in 10 of 10 patients. The look-at-the-scan rule did so in 1 of 10.

## What we built

| Piece | What it does | Evidence |
| --- | --- | --- |
| Real airways | 130 NasalSeg CT scans → nasal cavities + nasopharynx; every voxel becomes an OpenFOAM hex cell | 126 of 130 simulated; median nasal resistance 0.146 Pa·s/cm³ at 150 Pa |
| Airflow ground truth | OpenFOAM v2606 simpleFoam, k-omega SST, 150 Pa, on Brev CPU instances | 126 patients and 106 idealized cases, plus 80 surgery runs |
| Fast surrogate | Power law per region on six measured geometry numbers per side | 16.3% mean side error (cross-validated) against 29.9% for "average patient"; under 0.2 ms per prediction |
| Four agents | Case → Planner → Critic → Explainer, one OpenAI model, each limited to its own tools | Reliability not yet measured on OpenAI |
| Virtual surgery | Turbinate reduction, valve repair and septoplasty applied to the voxel airway, scored by OpenFOAM | Workflow++ reached the best plan's drop in 10/10; standard rule 1/10 |
| Apps | Gradio: the studio (CT to airway, OpenFOAM, virtual surgery, agents) and the demo (3D resistance map, what-if sliders, live agent steps) | `nasal-studio`, `nasal-demo` |

## Tracks

- **Track 3, tool calling (main fit).** One model makes every call through validated tools. Code checks each call's arguments against a JSON schema and the parameter ranges, and sends errors back to the model to fix. Unknown tools are rejected. The Critic can cite only flag ids that the code checks produced. The Explainer's numbers are checked against the tool results: one retry, then a summary written by code. `nasal-agents --eval` scores 16 fixed probes in code; it has not been run on OpenAI yet.
- **Track 2, multi-agent.** Four agents, each with its own role and tools, pass context in a fixed pipeline: Case sets the patient, Planner runs physics, Critic checks, Explainer writes. They are four roles on one model rather than four different models.
- **Track 1, fine-tuned and specialized.** Not attempted: no model was fine-tuned.

NVIDIA Brev ran all the OpenFOAM batches, on 48-core CPU instances.

## Demo script

See [PRESENTATION.md](PRESENTATION.md): setup checklist, the word-for-word 4-minute script with clicks, backups and likely questions.

## Honest limits

- **Model fidelity:** the voxel airways have stair-stepped walls, and the flow model is steady with rigid walls and no mucosa. Region boundaries are positions along each cavity, not anatomical landmarks.
- **Surrogate:** it ranks well, but its size estimates are rough (16% side error; it underpredicts the most obstructed sides). The agents and the studio therefore use OpenFOAM results for real patients.
- **Virtual surgery:** each procedure is a 0.6 mm voxel change. The voxel septoplasty removes more airway than it adds, so it raises resistance in this model; that is an artifact, not a finding about septoplasty.
- **Data:** NasalSeg contains duplicate scans (P002/P003, P009/P010, P080/P081). 4 scans couldn't be simulated.
- **Rules:** the decision rules use placeholder thresholds until an ENT reviews them (workflow step 7).

## Reproduce

See the README for each command: `nasal-patients simulate` (OpenFOAM in Docker), `nasal-surrogate --patients`, `nasal-patients surgery`, `nasal-agents --eval`, `nasal-studio` and `nasal-demo`.
