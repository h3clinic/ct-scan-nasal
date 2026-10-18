# Nasal0: 3-minute script (two speakers)

**Live demo:** https://d36fd646c8e72bb746.gradio.live

**YOU** drives the demo. **PANKAJ** opens, covers results and closes. Times are when each line starts.

---

**0:00 · PANKAJ** *(title: Nasal0)*

> "A blocked nose is diagnosed by eye. A surgeon looks at a CT scan and operates on what looks wrong. But what looks wrong isn't always where the air gets stuck. Nasal0 shows where it gets stuck."

**0:20 · YOU** *(demo open on P022; rotate the airway)*

> "This is a real patient's nasal airway from a CT scan. We built 130 of them and ran a full airflow simulation in OpenFOAM on each one. Red is where the pressure is lost. This patient's left side has three times the resistance of the right, and it's at the back, not where you'd look first."

**0:50 · YOU** *(point at the table; move "Left narrowest" to +30)*

> "A simulation takes minutes. Our surrogate answers in under a millisecond, and we show its error. So we can ask: what if we widen the narrowest part by 30%? The map updates instantly."

**1:10 · YOU** *(set slider back to 0; click "Run agents")*

> "Now in plain language. Four agents, one small model: NVIDIA Nemotron 3.5 Lightning, 3 billion active parameters, served with vLLM on a Brev H100. The models only choose tools. Every number comes from code, and code checks every number they write."

**1:40 · YOU** *(answer appears)*

> "A few seconds. And the Critic flagged that the fast model and the simulation disagree here, so you know when not to trust the shortcut."

**1:55 · PANKAJ** *(gallery: Virtual surgery)*

> "Then we tested decisions. On the 10 most obstructed airways, we simulated every surgery plan: 80 OpenFOAM runs. The plan chosen from the resistance map matched the best plan for 10 out of 10 patients. The look-at-the-scan rule: 1 out of 10."

**2:25 · PANKAJ** *(gallery: Agents)*

> "And the tool calling holds up: 16 test requests, including broken ones, all completed. Code caught every bad tool call, and the model recovered."

**2:40 · PANKAJ** *(close)*

> "Real anatomy, real airflow, instant answers, and small models you can check. That's Nasal0. The link is live; try any patient."

---

## Hand-offs

- 0:20 Pankaj → You: "...Nasal0 shows where it gets stuck." You start on the demo.
- 1:55 You → Pankaj: "...when not to trust the shortcut." Pankaj switches to the gallery.

## If the agents hang

You say: "The agents are running on our H100; here's a recorded run," then play the gallery video from 1:05. Everything else works without the model.
