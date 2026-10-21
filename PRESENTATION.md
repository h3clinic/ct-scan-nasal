# Nasal0: presenting it

Everything for a live presentation: setup, the word-for-word script with clicks, backups, likely questions and the numbers to remember.

## 30 minutes before

1. **Network:** connect to your phone hotspot if the venue's Wi-Fi blocks APIs. Check with `curl -s https://api.openai.com/v1/models` (any reply means it isn't blocked).
2. **OpenAI:** set `OPENAI_API_KEY` (or paste the key in the studio's Settings) and make sure the account has credits; with none, every agent request fails.
3. **Demo:** run `nasal-demo --labels <labels> --sim data/patient_sim.csv --model data/patient_surrogate.json` and open http://127.0.0.1:7860. It should open on P022.
4. **Warm-up:** click **Run agents** once, then reload the page so it starts clean.
5. **Tabs:** the demo, the Nasal0 gallery (scrolled to the video, as backup) and the repo README. Close everything else, mute notifications, and set the browser zoom to 125% so the room can read it.

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

> "Now ask in plain language. Four agents, all one OpenAI model, each with its own job and its own tools.
>
> Watch the steps stream in. The Case agent picks the patient. The Planner calls the physics tools. The Critic checks the results. The Explainer writes the answer.
>
> The models only choose tools. Every number comes from code, and code checks every number the Explainer writes. If one doesn't match, code writes the answer instead."

**[Answer appears] (2:20)**

> "And every resistance in that answer comes from this patient's own OpenFOAM simulation, not from the fast model."

**[Gallery: Virtual surgery section, or describe it] (2:40)**

> "Then we tested decisions. On the 10 most obstructed airways, we ran virtual surgery for every combination of turbinate reduction, valve repair and septoplasty, and checked each in OpenFOAM: 80 simulations.
>
> The plan picked from the resistance map matched the best plan for 10 out of 10 patients. The look-at-the-scan rule managed 1 in 10."

**[Gallery: Agents section] (3:20)**

> "And the tool calling is checked by code. Arguments are validated against the parameter ranges, bad calls go back to the model to fix, and an unknown patient gets an error, not a guess."

**[Close] (3:40)**

> "Real anatomy, real airflow simulation, a sub-millisecond surrogate and agents whose numbers code checks. That's Nasal0. Next step: a surgeon reviews the rules."

## If something breaks

- **Agents fail or hang:** keep going with the map, sliders and table; they work without the model. Then play the agent part of the video in the gallery (from 1:05).
- **Demo won't start:** play the gallery video from the start and narrate the same script over it.
- **OpenAI refuses (no credits, bad key):** the console says why in plain words; fix it in Settings, or carry on without the agents.
- **No internet at all:** the demo still runs locally except the agents. Use the video for that part.

## Likely questions

**Is this real patient data?**
Yes, the airways come from NasalSeg, a public dataset of 130 CT scans. We simulated 126 of them; 4 had broken labels or didn't converge.

**How accurate is the simulation?**
It's a standard steady-state OpenFOAM setup (k-omega SST turbulence model). The median resistance, 0.146 Pa·s/cm³, is in the range measured on real adult noses. The walls are rigid and at voxel resolution, so the absolute numbers are approximate; comparing plans is the strong use.

**Why a surrogate instead of always simulating?**
A simulation takes minutes on many cores. The surrogate takes well under a millisecond, which is what makes the demo's sliders instant. On some airways it is off by more than 30%, so for real patients the agents read OpenFOAM results instead, and the studio runs a missing case in a few minutes.

**How good is the surrogate?**
On real airways it averages a 16% error per side, against 30% for simply guessing the average patient. Its estimates of how much a surgery plan helps are rough (correlation 0.51 with OpenFOAM), which is why plans were scored in OpenFOAM.

**Why one model for four agents?**
Four roles with separate tools and checks, on one model, keep it simple and cheap. More agents mostly add latency and failure points.

**What stops the model from making up numbers?**
Code computes every number. The Explainer's text is checked number by number against the tool results. If a number doesn't match, it gets one retry, and then code writes the answer instead. The Critic can only cite flags that the code checks produced.

**Did you fine-tune anything?**
No. The agents use an OpenAI model as offered, with its standard tool calling.

**Septoplasty made things worse?**
In our model, yes, but that's an artifact: our voxel version of septoplasty removes more airway than it adds. It's not a finding about real septoplasty.

**Where is NVIDIA used?**
NVIDIA Brev CPU instances ran the OpenFOAM simulations. We also tried NVIDIA's NV-Generate-CT model for synthetic CT. The agents run on OpenAI's API.

**Does patient data go to OpenAI?**
Only the patient's dataset id, the question and computed numbers, never a CT. Requests use store=False; OpenAI may keep API data up to 30 days for abuse monitoring unless the organization has zero data retention.

**What's next?**
A surgeon reviewing the decision rules, smoother airway walls, and more patients.

## Numbers to remember

| | |
| --- | --- |
| CT airways | 130 (126 simulated) |
| Median nasal resistance | 0.146 Pa·s/cm³ at 150 Pa |
| Surrogate | 16% side error vs 30% baseline; under 1 ms |
| Virtual surgery | map-driven plan matched best 10/10; look-at-the-scan rule 1/10; 80 simulations |
| Agents | four roles on one OpenAI model (`gpt-4.1-mini` by default); reliability not yet measured on OpenAI |
