# Nasal0: 5-minute script (two speakers)

**Live demo:** https://7a5747cfe97ac7e65b.gradio.live

**YOU** drives the demo. **PANKAJ** opens, covers results and closes. Times are when each line starts.

**Before going on:** open the demo on P022, run the default question once and the surgery question once (so both are warm), then reload the page.

---

**0:00 · PANKAJ** *(title: Nasal0)*

> "A blocked nose is diagnosed by eye. A surgeon looks at a CT scan and operates on what looks wrong. But what looks wrong isn't always where the air gets stuck. Nasal0 shows where it gets stuck."

**0:20 · YOU** *(demo open on P022; rotate the airway slowly)*

> "This is a real patient's nasal airway, rebuilt from a CT scan. We took 130 public scans, turned each airway into a 3D mesh, and ran a full airflow simulation on every one in OpenFOAM, the same kind of software used for car and aircraft aerodynamics. We push air through at the pressure of a normal breath. Red is where the pressure is lost, which is where breathing gets hard."

**0:50 · YOU** *(turn the airway to show the back of the left side)*

> "Look at this patient. The left side has three times the resistance of the right. And it isn't at the front, where you'd look first. 87% of the left side's resistance is right at the back, where the nose opens into the throat."

**1:15 · YOU** *(point at the table)*

> "This table splits each side into four sections, front to back: the nostril, the valve, the turbinates and the back opening. These are the OpenFOAM numbers. Next to them is our surrogate, a small formula we fitted to all 126 simulations. A simulation takes minutes on a cluster; the surrogate takes under a millisecond. And we show its error: for this patient it's 45% off. Fast is useful, but you have to know when it's wrong."

**1:50 · YOU** *(move "Left narrowest" to +30)*

> "Because it's instant, we can ask what-ifs. Widen the narrowest part of the left side by 30%, and the map and totals update right away."

**2:05 · YOU** *(set the slider back to 0; click "Run agents" with the default question)*

> "Now the same thing in plain language. Four agents, all one small model: NVIDIA Nemotron 3.5 Lightning, 30 billion parameters with only 3 billion active, served with vLLM on an NVIDIA Brev H100. The Case agent picks the patient, the Planner picks the physics tools, the Critic checks the results, and the Explainer writes the answer. The model only chooses tools. Every number comes from code, and code checks every number it writes. If one doesn't match, code writes the answer instead."

**2:40 · YOU** *(answer appears)*

> "A few seconds. And the Critic flagged that the surrogate and the simulation disagree for this patient, so you know when not to trust the shortcut."

**3:00 · PANKAJ** *(gallery: Virtual surgery)*

> "Then we tested decisions. On the 10 most obstructed airways, we simulated every surgery plan: 80 OpenFOAM runs. The plan chosen from the resistance map matched the best plan for 10 out of 10 patients. The look-at-the-scan rule: 1 out of 10."

**3:30 · YOU** *(back to the demo; type "What surgery option should I do on this patient"; click "Run agents")*

> "And the agents use that same decision rule. Let's ask: what surgery should this patient have? The Planner runs virtual surgery on this patient's own CT airway: turbinate reduction, valve repair, septoplasty, and every combination, seven plans. Then it picks the smallest plan that gets within 90% of the best result. That's Pankaj's rule."

**3:55 · YOU** *(answer appears)*

> "Turbinate reduction plus valve repair. Predicted 6% lower resistance, and when we ran it in OpenFOAM, 14% lower: the best of all seven plans. The look-at-the-scan choice, turbinate reduction alone, gets 9%. It says why each step is there, and it warns about something a surgeon should know: most of this patient's resistance sits at the back, where none of these three procedures acts."

**4:20 · PANKAJ** *(gallery: Agents)*

> "And the tool calling holds up: 16 test requests, including broken ones, all completed. Code caught every bad tool call, and the model recovered."

**4:35 · PANKAJ** *(close)*

> "Real anatomy, real airflow, instant answers, and small models you can check. That's Nasal0. The link is live; try any patient."

---

## Hand-offs

- 0:20 Pankaj → You: "...Nasal0 shows where it gets stuck." You start on the demo.
- 3:00 You → Pankaj: "...when not to trust the shortcut." Pankaj switches to the gallery (Virtual surgery).
- 3:30 Pankaj → You: "...1 out of 10." You switch back to the demo.
- 4:20 You → Pankaj: "...where none of these three procedures acts." Pankaj switches to the gallery (Agents).

## If the agents hang

You say: "The agents are running on our H100; here's a recorded run," then play the gallery video from 1:05. Everything else works without the model. For the surgery answer, say the numbers above: they are the same every time.
