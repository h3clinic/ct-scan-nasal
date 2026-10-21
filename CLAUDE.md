# Working on ct-scan-nasal

## Keep everything consistent in the same change

Every code change ships with the matching doc, test and config changes in the same commit. Never leave them for later.

- **README matches the code.** If a change alters a tool, an agent, the workflow, commands, or what's done vs planned, update the README's tables and Status in the same commit.
- **One source of truth for parameters.** `PARAM_SPECS` in `src/nasal/params.py` defines every case parameter. The generator, the Critic's range check and the README table all derive from it. After changing it, run `python -m nasal.docs`; the tests fail if the README is stale.
- **Names stay in sync.** When renaming a module, function, CLI flag, CSV column or parameter, update every reference, including docs, tests and examples. Search the repo for the old name before finishing.
- **No claims without code.** The README reports only numbers (resistance, surrogate error, speedup, rule comparisons) that the repo's code actually produces. Mark planned work as planned.

## Project rules

- **No disclaimer labels.** Don't add "research only" or "not medical advice" labels; the owner removed them on purpose. State the model's technical limits (rigid walls, steady flow, placeholder thresholds) where they matter for reading a result.
- **Physics and decisions are code.** Resistance, surrogate predictions and decision rules are plain functions. Language-model agents call them and never produce or alter numbers themselves.
- **Decision rules need clinical input.** Don't invent clinical thresholds. Rules stay placeholders until an ENT reviews them, and the README says who reviewed what.
- **Agents run on OpenAI.** The agents use OpenAI's API only (requests sent with `store=False`), with about four agents. No vLLM or Nemotron. Record the default model name in the README. Send the model only the patient id, the question and computed numbers, never a CT or other patient data. The one exception is the local distilled Qwen explainer (`nasal.distill`), which runs on this computer and only writes answers from facts computed by code.
- **Calibration comes from simulation.** Fit constants in the baseline only to OpenFOAM results, never by hand to make numbers look right.

## Before committing

1. Run `python -m nasal.docs`, then `pytest`. Fix failures rather than skipping them.
2. Check the diff: does any doc, test or example now describe something that no longer matches?
3. Search for stale references to anything renamed or removed.

## Environment

- The local virtual environment is at `C:\Users\brita\.venvs\ct-scan-nasal`, because deep paths break venvs here.
- Long paths can break git on Windows here: use `git -c core.longpaths=true ...`.
