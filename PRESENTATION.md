# Nasal0: presenting it

Everything for a live presentation: setup, the word-for-word script with clicks, backups, likely questions and the numbers to remember.

## 30 minutes before

1. **Network:** connect to your phone hotspot if the venue's Wi-Fi blocks APIs. Check with `curl -s https://api.openai.com/v1/models` (any reply means it isn't blocked).
2. **OpenAI:** set `OPENAI_API_KEY` (or paste the key in the studio's Settings) and make sure the account has credits; with none, every agent request fails.
3. **Studio:** set `NASALSEG_ROOT` to the NasalSeg folder and start `start-studio.bat` (or `./start-studio.sh`). It opens http://127.0.0.1:7861 on P022 and shows the setup check; everything the demo uses should be ready.
4. **Warm-up:** ask the console one question, so the local model is loaded, then reload the page so it starts clean.
5. **Tabs:** the studio, the Nasal0 gallery (scrolled to the video, as backup) and the repo README. Close everything else, mute notifications, and set the browser zoom to 125% so the room can read it.

## Script (about 4 minutes)

**[Slide or demo title: Nasal0]**

> "An examination and CT scan show nasal anatomy, but they do not directly quantify where airflow loses pressure. What looks narrow isn't always where the modeled pressure loss is highest.
>
> Nasal0 shows where it gets stuck."

**[Demo: P022 open, resistance map visible] (0:20)**

> "This is a real nasal airway, from one of 130 public CT scans. Every cell of this airway went through a full OpenFOAM simulation at the pressure a doctor's breathing test uses. Red is where the pressure is lost."

**[Drag to rotate the airway] (0:40)**

> "This patient's left side carries three times the resistance of the right, and most of it is at the back of the left cavity, not at the front where you'd look first."

**[Point at the region table] (1:00)**

> "The table splits it by region. The left choana alone holds most of that side's resistance."

**[Click "Run all 7 plans"] (1:20)**

> "Every combination of turbinate reduction, valve repair and septoplasty, each checked in OpenFOAM, ranked by how much it lowers resistance. These are stored runs; a new patient's run streams live while the solver works."

**[Type in the console: "Compare the modeled procedure options for this patient."] (1:40)**

> "Now ask in plain language. A small model running on this laptop answers, with no cloud and no API key.
>
> The model only writes the words. Code picks the tools, every number comes from OpenFOAM, and code checks every number in the answer. If one doesn't match, code writes the answer instead."

**[Answer appears] (2:20)**

> "And every resistance in that answer comes from this patient's own OpenFOAM simulation, not from the fast model."

**[Gallery: Virtual surgery section, or describe it] (2:40)**

> "Then we tested decisions. On the 10 most obstructed airways, we ran virtual surgery for every combination of turbinate reduction, valve repair and septoplasty, and checked each in OpenFOAM: 80 simulations.
>
> The plan picked from the resistance map came within 90% of the best plan for 8 out of 10 patients. The look-at-the-scan rule managed 1 in 10. And the same plan for everyone did just as well, so the next step is to show when tailoring the plan pays off."

**[Gallery: Agents section] (3:20)**

> "And the tool calling is checked by code. Arguments are validated against the parameter ranges, bad calls go back to the model to fix, and an unknown patient gets an error, not a guess."

**[Close] (3:40)**

> "Real anatomy, real airflow simulation, a sub-millisecond surrogate and agents whose numbers code checks. That's Nasal0. Next step: a surgeon reviews the rules."

## If something breaks

- **The console fails or hangs:** keep going with the map, the region table and the plan ranking; they work without any model. Then play the agent part of the video in the gallery (from 1:05).
- **The studio won't start:** play the gallery video from the start and narrate the same script over it.
- **OpenAI refuses (no credits, bad key):** the console says why in plain words; fix it in Settings, or carry on without the agents.
- **No internet at all:** the studio runs fully locally, including the console's local model.

## Likely questions

**Is this real patient data?**
Yes, the airways come from NasalSeg, a public dataset of 130 CT scans. We simulated 126 of them; 4 had broken labels or didn't converge.

**How accurate is the simulation?**
It's a standard steady-state OpenFOAM setup (k-omega SST turbulence model). The median resistance, 0.146 Pa·s/cm³, is in the range measured on real adult noses. The walls are rigid and at voxel resolution, so the absolute numbers are approximate; comparing plans is the strong use.

**Why a surrogate instead of always simulating?**
A simulation takes minutes on many cores. The surrogate takes well under a millisecond, but on some airways it is off by more than 30%, so for real patients the agents and the studio read OpenFOAM results instead, and the studio runs a missing case in a few minutes.

**How good is the surrogate?**
On real airways it averages a 16% error per side, against 30% for simply guessing the average patient. Its estimates of how much a surgery plan helps are rough (correlation 0.38 with OpenFOAM), which is why plans were scored in OpenFOAM.

**Why one model for four agents?**
Four roles with separate tools and checks, on one model, keep it simple and cheap. More agents mostly add latency and failure points.

**What stops the model from making up numbers?**
Code computes every number. The Explainer's text is checked number by number against the tool results. If a number doesn't match, it gets one retry, and then code writes the answer instead. The Critic can only cite flags that the code checks produced.

**Did you fine-tune anything?**
The hosted agents use the selected provider's model as offered, with tool calling. The optional local explainer is Qwen2.5-1.5B-Instruct fine-tuned with LoRA on 555 answers that passed the project's fact checks.

**Septoplasty made things worse?**
It did in our first version, because that voxel septoplasty removed more airway than it added. Now it straightens only the parts of the septum that bulge and keeps its thickness, and on its own it is neutral (0.2% lower on average); added to turbinate reduction and valve repair it helped 5 of the 10 patients, by up to 1.6 points.

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
| Virtual surgery | map-driven plan within 90% of best for 8/10 (a fixed plan ties it); look-at-the-scan rule 1/10; 80 simulations |
| Agents | four roles on one OpenAI model (`gpt-4.1-mini` by default); reliability not yet measured on OpenAI |
