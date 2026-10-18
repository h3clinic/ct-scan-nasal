# Nasal0: presenting it

Everything for a live presentation: setup, the word-for-word script with clicks, backups, likely questions and the numbers to remember.

## 30 minutes before

1. **Network:** connect to your phone hotspot, not school Wi-Fi (it blocks NVIDIA and Brev). Check with `curl -s https://api.ngc.nvidia.com` (any reply means it isn't blocked).
2. **GPU:** `brev ls` must show `nemotron-vllm` RUNNING. If it is stopped, run `brev start nemotron-vllm` and allow 15 minutes for the model to load.
3. **Tunnel:** keep the "vLLM tunnel (auto-reconnect)" terminal tab open. `curl -s localhost:8000/v1/models` must print a model list.
4. **Demo:** run `nasal-demo --labels <labels> --sim data/patient_sim.csv --model data/patient_surrogate.json` and open http://127.0.0.1:7860. It should open on P022.
5. **Warm-up:** click **Run agents** once, then reload the page so it starts clean.
6. **Tabs:** the demo, the Nasal0 gallery (scrolled to the video, as backup) and the repo README. Close everything else, mute notifications, and set the browser zoom to 125% so the room can read it.

## Script (about 4 minutes)

**[Slide or demo title: Nasal0]**

> "A blocked nose is diagnosed by eye. A surgeon looks at a CT scan, sees a bent septum or a big turbinate, and operates on what looks wrong. But what looks wrong isn't always where the air gets stuck.
>
> Nasal0 shows where it gets stuck."

**[Demo: P022 open, resistance map visible] (0:20)**

> "This is a real nasal airway, from one of 130 public CT scans. Every cell of this airway went through a full OpenFOAM simulation at the pressure a doctor's breathing test uses. Red is where the pressure is lost."

**[Drag to rotate the airway] (0:40)**

> "This patient's left side carries three times the resistance of the right, and most of it is at the back of the left cavity, not at the front where you'd look first."

**[Point at the numbers and table] (1:00)**

> "A simulation like this takes minutes on a cluster. Our surrogate answers in a fraction of a millisecond. And we show its error: for this patient it is 45% off, which our system flags itself."

**[Move "Left narrowest" to +30] (1:20)**

> "So, what if we widen the narrowest part of the left side by 30%? The map and the totals update instantly."

**[Set it back to 0. Click "Run agents"] (1:40)**

> "Now ask in plain language. Four agents, all one small model: NVIDIA Nemotron 3.5 Lightning, 30 billion parameters with 3 billion active, served with vLLM on an NVIDIA Brev H100.
>
> Watch the steps stream in. The Case agent picks the patient. The Planner calls the physics tools. The Critic checks the results. The Explainer writes the answer.
>
> The models only choose tools. Every number comes from code, and code checks every number the Explainer writes. If one doesn't match, code writes the answer instead."

**[Answer appears, about 6 s] (2:20)**

> "Six seconds. And the Critic caught that the fast model and the simulation disagree for this patient, so you know when not to trust the shortcut."

**[Gallery: Virtual surgery section, or describe it] (2:40)**

> "Then we tested decisions. On the 10 most obstructed airways, we ran virtual surgery for every combination of turbinate reduction, valve repair and septoplasty, and checked each in OpenFOAM: 80 simulations.
>
> The plan picked from the resistance map matched the best plan for 10 out of 10 patients. The look-at-the-scan rule managed 1 in 10."

**[Gallery: Agents section] (3:20)**

> "And the tool calling holds up. On 16 test requests, including deliberately broken ones like an unknown patient or an impossible valve angle, all 16 ran. Code caught all 3 bad tool calls, and the model recovered."

**[Close] (3:40)**

> "Real anatomy, real airflow simulation, a sub-millisecond surrogate and small models that call tools reliably. That's Nasal0. Next step: a surgeon reviews the rules."

## If something breaks

- **Agents fail or hang:** keep going with the map, sliders and table; they work without the model. Then play the agent part of the video in the gallery (from 1:05).
- **Demo won't start:** play the gallery video from the start and narrate the same script over it.
- **The tunnel drops:** it reconnects by itself within a few seconds; click **Run agents** again.
- **No internet at all:** the demo still runs locally except the agents. Use the video for that part.

## Likely questions

**Is this real patient data?**
Yes, the airways come from NasalSeg, a public dataset of 130 CT scans. We simulated 126 of them; 4 had broken labels or didn't converge.

**How accurate is the simulation?**
It's a standard steady-state OpenFOAM setup (k-omega SST turbulence model). The median resistance, 0.146 Pa·s/cm³, is in the range measured on real adult noses. The walls are rigid and at voxel resolution, so the absolute numbers are approximate; comparing plans is the strong use.

**Why a surrogate instead of always simulating?**
A simulation takes minutes on many cores. The surrogate takes 0.2 ms, which is what makes sliders and agent answers instant. When it disagrees with a simulation by more than 25%, the Critic says so.

**How good is the surrogate?**
On real airways it averages a 16% error per side, against 30% for simply guessing the average patient. Its estimates of how much a surgery plan helps are rough (correlation 0.47 with OpenFOAM), which is why plans were scored in OpenFOAM.

**Why one model for four agents?**
Four roles with separate tools and checks, on one small model, keep it fast and cheap: a whole run takes about 6 seconds. More agents mostly add latency and failure points.

**What stops the model from making up numbers?**
Code computes every number. The Explainer's text is checked number by number against the tool results. If a number doesn't match, it gets one retry, and then code writes the answer instead. The Critic can only cite flags that the code checks produced.

**Did you fine-tune anything?**
No. We used Nemotron 3.5 Lightning as released, with its native tool calling.

**Septoplasty made things worse?**
In our model, yes, but that's an artifact: our voxel version of septoplasty removes more airway than it adds. It's not a finding about real septoplasty.

**Where are NVIDIA and Red Hat used?**
Nemotron 3.5 Lightning runs all the agents. vLLM serves it on a Brev H100. Brev CPU instances ran the OpenFOAM simulations. We also tried NVIDIA's NV-Generate-CT model for synthetic CT.

**What's next?**
A surgeon reviewing the decision rules, smoother airway walls, and more patients.

## Numbers to remember

| | |
| --- | --- |
| CT airways | 130 (126 simulated) |
| Median nasal resistance | 0.146 Pa·s/cm³ at 150 Pa |
| Surrogate | 16% side error vs 30% baseline; under 0.2 ms |
| Virtual surgery | map-driven plan matched best 10/10; look-at-the-scan rule 1/10; 80 simulations |
| Agents | 16/16 runs, 15/16 correct case, 3/98 bad calls caught; ~6 s per run |
| Model | Nemotron 3.5 Lightning, 30B total / 3B active, vLLM on a Brev H100 |
