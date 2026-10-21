# Nasal0: NVIDIA x Red Hat Small Models Hack submission

**Live demo:** https://7a5747cfe97ac7e65b.gradio.live (runs on the Brev H100 next to the model)



A blocked nose is usually judged by eye: a surgeon looks at a CT scan, sees a bent septum or big turbinates, and operates on what looks wrong. **Nasal0** asks where the air actually loses pressure.

- **Real airways:** it builds airways from 130 real CT scans (NasalSeg) and simulates the airflow in OpenFOAM.
- **Fast stand-in:** a surrogate fitted to those simulations answers in a fraction of a millisecond, compared with minutes for a simulation.
- **Small-model agents:** four agents on **NVIDIA Nemotron 3.5 Lightning (30B total, 3B active)**, served with **vLLM on NVIDIA Brev**, let you ask about a patient in plain language. Every number comes from code: the models choose tools and explain, and code checks every number they write.

When we tested virtual surgery on 10 obstructed real airways, the map-driven plan (Workflow++) matched the best plan found by OpenFOAM in 10 of 10 patients. The look-at-the-scan rule did so in 1 of 10.

## What we built

| Piece | What it does | Evidence |
| --- | --- | --- |
| Real airways | 130 NasalSeg CT scans → nasal cavities + nasopharynx; every voxel becomes an OpenFOAM hex cell | 126 of 130 simulated; median nasal resistance 0.146 Pa·s/cm³ at 150 Pa |
| Airflow ground truth | OpenFOAM v2606 simpleFoam, k-omega SST, 150 Pa, on Brev CPU instances | 126 patients and 106 idealized cases, plus 80 surgery runs |
| Fast surrogate | Power law per region on six measured geometry numbers per side | 16.3% mean side error (cross-validated) against 29.9% for "average patient"; under 0.2 ms per prediction |
| Four agents | Case → Planner → Critic → Explainer, one Nemotron model, each limited to its own tools | Reliability check below |
| Virtual surgery | Turbinate reduction, valve repair and septoplasty applied to the voxel airway, scored by OpenFOAM | Workflow++ reached the best plan's drop in 10/10; standard rule 1/10 |
| Demo | Gradio: 3D resistance map of a real patient, what-if sliders, agents streaming each step live | `nasal-demo` |

## Tracks and bounties

- **Track 3, tool calling (main fit).** One model makes every call through validated tools. Code checks each call's arguments against a JSON schema and the parameter ranges, and sends errors back to the model to fix. Unknown tools are rejected. The Critic can cite only flag ids that the code checks produced. The Explainer's numbers are checked against the tool results: one retry, then a summary written by code. Measured on 16 fixed probes against the live endpoint (`nasal-agents --eval`), before the patient tools switched from the surrogate to OpenFOAM results; not yet repeated:
  - **Runs:** 16/16 completed, with a median of 5.9 s per full four-agent run.
  - **Case choice:** 15/16 chose the right case. The miss: "septum to the right 0.5" was set as +0.5, a sign error no code check can catch.
  - **What-ifs:** 6/6 run when the request asked for one.
  - **Tool calls:** 98 in total. The 3 invalid ones (an unknown patient id, an out-of-range valve angle twice) were caught by code and sent back, and the model recovered to a valid case.
  - **Explainer:** 14/16 passed the number check the first time. The other 2 were caught and replaced by the code template.
- **Track 2, multi-agent.** Four agents, each with its own role and tools, pass context in a fixed pipeline: Case sets the patient, Planner runs physics, Critic checks, Explainer writes. They are four roles on one model rather than four different models.
- **Track 1, fine-tuned and specialized.** Not attempted: no model was fine-tuned.
- **Bounty: NVIDIA Nemotron 3.5 Lightning.** All four agents run on `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, with its reasoning mode off for speed and its native tool-call parser.
- **Bounty: vLLM.** The model is served by `vllm/vllm-openai:v0.27.1` on one NVIDIA Brev H100, through an OpenAI-compatible endpoint. NVIDIA Brev also ran all the OpenFOAM batches, on 48-core CPU instances.

Every language model is 30B parameters or fewer, under the 33B limit.

## Demo script

See [PRESENTATION.md](PRESENTATION.md): setup checklist, the word-for-word 4-minute script with clicks, backups and likely questions.

## Honest limits

- **Model fidelity:** the voxel airways have stair-stepped walls, and the flow model is steady with rigid walls and no mucosa. Region boundaries are positions along each cavity, not anatomical landmarks.
- **Surrogate:** it ranks well, but its size estimates are rough (16% side error; it underpredicts the most obstructed sides). The Critic flags large disagreements.
- **Virtual surgery:** each procedure is a 0.6 mm voxel change. The voxel septoplasty removes more airway than it adds, so it raises resistance in this model; that is an artifact, not a finding about septoplasty.
- **Data:** NasalSeg contains duplicate scans (P002/P003, P009/P010, P080/P081). 4 scans couldn't be simulated.
- **Rules:** the decision rules use placeholder thresholds until an ENT reviews them (workflow step 7).

## Reproduce

See the README for each command: `nasal-patients simulate` (OpenFOAM in Docker), `nasal-surrogate --patients`, `nasal-patients surgery`, `nasal-agents --eval`, and `nasal-demo`.
