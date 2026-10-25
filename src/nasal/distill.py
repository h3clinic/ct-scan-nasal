"""Distillation data for a small local explainer: a Qwen2.5-14B teacher answers questions about tool facts, and
a Qwen2.5-1.5B student is fine-tuned on the answers that pass the number check.

Each example is one bundle of facts computed by code (a patient's OpenFOAM resistances, every surgery plan's
OpenFOAM result, one plan's before and after) plus one question; every bundle also carries the median over all simulated patients and this patient's
percentile, so "high" and "normal" have a reference. The teacher must explain the exact numbers,
what they mean in practice, the procedure, how it is generally performed, its result in the model, possible
consequences and the alternatives. Its answer is kept only if every number in it appears in the facts
(`agents.unsupported_numbers`) and it quotes the key results (`agents.missing_numbers`); two retries, then the
example is dropped. Procedure descriptions come from the teacher's general medical knowledge, not from this
project's code, and need review by an ENT before anyone relies on them.

`nasal-distill build` writes `data/distill/examples.jsonl` with a patient-level train/test split, `student-data`
swaps in the student's short prompt, and `eval` scores a model on the held-out questions. Training and
export run in a separate CUDA environment (`scripts/distill_train.py`).
"""

import json
import random
import time
from pathlib import Path

from . import agents
from .decision import patient as ps
from .params import DUPLICATE_SCANS, REGIONS, SIDES
from .patient_flow import PatientFlow

TEACHER_MODEL = "qwen2.5:14b-instruct-q4_K_M"
STUDENT_MODEL = "nasal-explainer"  # the distilled model's name in Ollama
OLLAMA_URL = "http://127.0.0.1:11434/v1"
OUT_DIR = Path("data/distill")
TEST_SHARE = 0.12  # share of patients held out for evaluation

# General textbook summary given to the teacher so it does not invent procedure details. NOT reviewed by an ENT:
# it must be checked before anyone relies on the student's procedure descriptions. Kept free of numbers, which
# the number check would reject.
PROCEDURE_NOTES = (
    "Procedure reference (general textbook summary, not reviewed by an ENT). "
    "Turbinate reduction: shrinks the inferior turbinates to widen the main airflow channel; common techniques are "
    "submucous resection or microdebrider reduction that keep the mucosa, radiofrequency or coblation reduction, and "
    "outfracture (pushing the turbinate bone outward); usually done through the nostrils with an endoscope under "
    "general or local anaesthesia. Risks: bleeding, crusting and dryness, adhesions (synechiae), regrowth, and empty "
    "nose syndrome (ENS) when too much tissue is removed. "
    "Nasal valve repair: widens or stiffens the narrowest front part of the nose; common techniques are spreader "
    "grafts (cartilage strips between the septum and upper lateral cartilages), alar batten grafts, flaring sutures "
    "and lateral crural strut grafts, through an open or closed rhinoplasty approach, often using septal or ear "
    "cartilage. Risks: change in the nose's appearance, graft shifting or being visible, infection, and persistent "
    "obstruction. "
    "Septoplasty: straightens a deviated septum through an incision inside the nose; the surgeon lifts the lining "
    "(mucoperichondrial flaps), removes or reshapes the bent cartilage and bone while keeping an L-shaped support "
    "strut, then closes or splints the septum. Risks: bleeding, septal hematoma, septal perforation, loss of "
    "support (saddle nose), numbness of the upper teeth, persistent deviation and, rarely, a cerebrospinal fluid "
    "leak. Alternatives to surgery include nasal steroid sprays, saline rinses, treating allergies and external "
    "nasal dilators; they mainly act on mucosal swelling, which this model does not include."
)

SYSTEM = (
    "You explain nasal airflow results from a research model to a clinician. The results JSON was computed by "
    "code: OpenFOAM airflow simulations on the patient's CT airway (rigid walls, one steady breath at 150 Pa) and "
    "virtual surgery that changes the voxel airway by about 0.6 mm per procedure. Rules: use only numbers that "
    "appear in the JSON, copied exactly, and never compute new ones. Resistance is in Pa*s/cm^3; higher means "
    "harder to breathe through. The 'comparison' entry was computed by code: use it for which side and region are "
    "highest and how this patient compares with the others; do not judge these yourself. Answer only what is "
    "asked, in short paragraphs with plain headings. When you "
    "describe how a procedure is performed, give the general steps as found in standard ENT references and say "
    "that the surgeon decides the technique. Say when a result comes from the model rather than from real patients. "
    "A negative drop means resistance goes up: say 'raises resistance', not a negative drop. Surgical risks and "
    "what the model does not include (not_modeled) are different things; list them separately. When a plan "
    "combines procedures (for example turbinate reduction + valve repair), describe each procedure in it, not "
    "only the first. "
) + PROCEDURE_NOTES

# The student learns the rules and the procedure knowledge from the teacher's answers, so it runs with a short
# prompt: less to read on every question is most of the compute saving.
STUDENT_SYSTEM = (
    "You explain nasal airflow results to a clinician. Use only numbers from the Results JSON, copied exactly. "
    "Resistance is in Pa*s/cm^3; higher means harder to breathe through."
)

QUESTIONS = {
    "patient": [
        "Explain these numbers: which side and which region make breathing hardest, and by how much?",
        "What do these resistance results mean for this patient in everyday life?",
        "Walk me through every number here and what it says about the airway.",
    ],
    "surgery": [
        "Which procedure should this patient have, and why? Compare it with the alternatives.",
        "Explain each surgical option, its predicted result in the model, and the main consequences and risks.",
        "What would you tell the patient about the recommended surgery and the alternatives?",
    ],
    "plan": [
        "Explain this procedure: how it is generally performed, and what it does to the airway.",
        "What is the predicted result of this procedure in the model, and what are its consequences and alternatives?",
        "How is this surgery performed, what result does the model predict, and what could go wrong?",
    ],
}


def fact_bundles(patients, surgery, labels_dir: Path | None = None) -> list[dict]:
    """Facts computed by the agents' tools for every stored patient and surgery plan."""
    flow = PatientFlow(patients, surgery, labels_dir)
    totals = patients.total_resistance.dropna()
    bundles = []

    surgery_ids = sorted(set(surgery.patient)) if surgery is not None else []
    for pid in sorted(patients.patient):
        tools = agents.Tools(patients=patients, flow=flow)
        try:
            tools.set_patient({"patient_id": pid})
            tools.patient_resistance({"patient_id": pid})
        except agents.ToolError:
            continue  # no OpenFOAM run for this patient
        tools.review_results({})
        bundles.append({"kind": "patient", "patient": pid, "facts": facts_for(tools, pid, totals)})
        if pid not in surgery_ids:
            continue
        tools.patient_surgery_options({"patient_id": pid})
        tools.review_results({})
        bundles.append({"kind": "surgery", "patient": pid, "facts": facts_for(tools, pid, totals)})
        for plan in ps.candidate_plans():
            one = agents.Tools(patients=patients, flow=flow)
            one.set_patient({"patient_id": pid})
            one.patient_resistance({"patient_id": pid})
            one.patient_what_if({"patient_id": pid, "procedures": list(plan)})
            one.results["surgery"] = tools.results["surgery"]  # the other plans, for alternatives
            one.review_results({})
            bundles.append({"kind": "plan", "patient": pid, "plan": ps.label(plan), "facts": facts_for(one, pid, totals)})
    return bundles


def facts_for(tools: agents.Tools, pid: str, totals) -> dict:
    """The tools' results plus comparisons computed by code: how this patient's total compares with all simulated
    patients (`totals`), the harder side and by how much, and the highest-resistance region."""
    pred = tools.results["prediction"]
    hard, easy = sorted(SIDES, key=lambda s: -pred[f"{s}_resistance"])
    region = max(((s, r) for s in SIDES for r in REGIONS), key=lambda sr: pred[f"{sr[0]}_{sr[1]}_resistance"])
    context = {"simulated_patients": int(len(totals)),
               "median_total_resistance_all_patients": round(float(totals.median()), 4),
               "patient_total_resistance_percentile": round(float(100 * (totals < pred["total_resistance"]).mean())),
               "harder_side": hard,
               "harder_side_times_easier_side": round(pred[f"{hard}_resistance"] / pred[f"{easy}_resistance"], 2),
               "highest_region": f"{region[0]} {region[1]}",
               "highest_region_resistance": pred[f"{region[0]}_{region[1]}_resistance"]}
    return {"case": {"patient_id": pid}, **{k: v for k, v in tools.results.items() if k != "flags"},
            "comparison": context, "flags": [f for f in tools.results.get("flags", []) if f["severity"] != "error"]}


def user_message(facts: dict, question: str) -> str:
    return f"Results JSON:\n{json.dumps(facts, indent=1)}\n\nQuestion: {question}"


def split_patients(pids: list[str], seed: int = 0) -> set[str]:
    """Held-out patients; duplicate scans (params.DUPLICATE_SCANS) go to the same side."""
    groups = sorted({DUPLICATE_SCANS.get(p, p) for p in pids})
    rng = random.Random(seed)
    test = set(rng.sample(groups, max(1, round(TEST_SHARE * len(groups)))))
    return {p for p in pids if DUPLICATE_SCANS.get(p, p) in test}


METHOD_NUMBERS = (0.6, 90.0, 3000.0, 900.0)  # mm per procedure layer, the 90% rule, solver iterations


PROCEDURE_WORDS = {"turbinate reduction": "turbinate", "valve repair": "valve", "septoplasty": "septoplasty"}


def missing_procedures(text: str, facts: dict) -> list[str]:
    """Procedures of the what-if plans that the answer never names (a combined plan must be described whole)."""
    named = {p for w in facts.get("what_if", []) for p in w["label"].split(" + ") if p in PROCEDURE_WORDS}
    return sorted(p for p in named if PROCEDURE_WORDS[p] not in text.lower())


def check_answer(text: str, facts: dict) -> str:
    """'' when the answer passes the number checks and names every procedure of the plan, else what is wrong."""
    bad, missing = agents.unsupported_numbers(text, facts, METHOD_NUMBERS), agents.missing_numbers(text, facts)
    left_out = missing_procedures(text, facts)
    return "; ".join(filter(None, [bad and f"numbers not in the JSON: {', '.join(bad)}",
                                   missing and f"missing required results: {', '.join(missing)}",
                                   left_out and f"the plan also includes {', '.join(left_out)}; describe it too"]))


def teacher_answer(client, facts: dict, question: str, model: str = TEACHER_MODEL) -> tuple[str | None, int]:
    """(checked answer or None, attempts used)."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(facts, question)}]
    for attempt in range(3):
        text = client.chat.completions.create(model=model, messages=messages, temperature=0.3,
                                              max_tokens=900).choices[0].message.content or ""
        problem = check_answer(text, facts)
        if not problem:
            return text.strip(), attempt + 1
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"Your answer had problems: {problem}. Rewrite it, using only numbers "
                                                  f"from the JSON copied exactly."}]
    return None, 3


def build(out_dir: Path = OUT_DIR, per_bundle: int = 2, seed: int = 0, limit: int | None = None,
          base_url: str = OLLAMA_URL) -> dict:
    """Generate teacher answers for every bundle (resumable) and write examples.jsonl."""
    from openai import OpenAI

    patients, surgery = agents.load_patient_table(), agents.load_surgery()
    bundles = fact_bundles(patients, surgery)
    test = split_patients(sorted({b["patient"] for b in bundles}), seed)
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "examples.jsonl"
    done = {(e["patient"], e["kind"], e.get("plan"), e["question"]) for e in map(json.loads, path.open(encoding="utf-8"))} \
        if path.exists() else set()
    client = OpenAI(base_url=base_url, api_key="ollama")
    stats = {"bundles": len(bundles), "written": 0, "dropped": 0, "skipped": 0}
    jobs = [(b, q) for b in bundles for q in rng.sample(QUESTIONS[b["kind"]], min(per_bundle, len(QUESTIONS[b["kind"]])))]
    for b, q in jobs[:limit]:
        if (b["patient"], b["kind"], b.get("plan"), q) in done:
            stats["skipped"] += 1
            continue
        start = time.time()
        answer, attempts = teacher_answer(client, b["facts"], q)
        if answer is None:
            stats["dropped"] += 1
            print(f"{b['patient']} {b['kind']} {b.get('plan', '')}: dropped after {attempts} tries", flush=True)
            continue
        row = {"patient": b["patient"], "kind": b["kind"], "plan": b.get("plan"), "question": q,
               "split": "test" if b["patient"] in test else "train", "attempts": attempts,
               "seconds": round(time.time() - start, 1),
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user_message(b["facts"], q)},
                            {"role": "assistant", "content": answer}]}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        stats["written"] += 1
        print(f"{b['patient']} {b['kind']} {b.get('plan', '')}: ok in {row['seconds']} s ({attempts} tries)", flush=True)
    return stats


def facts_of(row: dict) -> dict:
    """The Results JSON inside an example's user message."""
    return json.loads(row["messages"][1]["content"].split("Results JSON:\n", 1)[1].split("\n\nQuestion:", 1)[0])


def student_examples(path: Path = OUT_DIR / "examples.jsonl", out: Path = OUT_DIR / "student.jsonl") -> int:
    """The teacher's examples with the student's short system prompt; returns the number written."""
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
    with Path(out).open("w", encoding="utf-8") as f:
        for r in rows:
            r = {**r, "messages": [{"role": "system", "content": STUDENT_SYSTEM}, *r["messages"][1:]]}
            f.write(json.dumps(r) + "\n")
    return len(rows)


def evaluate(path: Path, model: str, system: str = STUDENT_SYSTEM, base_url: str = OLLAMA_URL,
             split: str = "test") -> dict:
    """Ask `model` every held-out question (with `system` as its prompt); report the number-check pass rate,
    speed and length."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="ollama")
    rows = [r for r in map(json.loads, Path(path).open(encoding="utf-8")) if r["split"] == split]
    passed, no_invented, seconds, words = 0, 0, [], []
    for r in rows:
        start = time.time()
        text = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": system}, r["messages"][1]],
            temperature=0.2, max_tokens=900).choices[0].message.content or ""
        seconds.append(time.time() - start)
        words.append(len(text.split()))
        passed += not check_answer(text, facts_of(r))
        no_invented += not agents.unsupported_numbers(text, facts_of(r), METHOD_NUMBERS)
    n = len(rows)
    return {"model": model, "questions": n, "number_check_passed": passed, "no_invented_numbers": no_invented,
            "median_seconds": round(sorted(seconds)[n // 2], 2) if n else None,
            "median_words": sorted(words)[n // 2] if n else None}


def key_numbers(facts: dict) -> str:
    """The results an answer must state, written by code: the total, each what-if, the recommended plan."""
    p = facts["prediction"]
    lines = [f"Total nasal resistance is {p['total_resistance']} Pa*s/cm^3 (left {p['left_resistance']}, "
             f"right {p['right_resistance']}) in OpenFOAM."]
    for w in facts.get("what_if", []):
        lines.append(f"With {w['label']}, it goes from {w['total_resistance_before']} to {w['total_resistance_after']} "
                     f"({w['total_change_pct']}%).")
    if facts.get("surgery"):
        lines.append(f"Recommended: {facts['surgery']['recommended_result']}.")
    return " ".join(lines)


SURGERY_WORDS = ("surg", "procedure", "operat", "plan", "treat", "option", "recommend", "alternativ", "fix")


def answer_locally(question: str, pid: str, patients, flow, client=None, model: str = STUDENT_MODEL,
                   base_url: str = OLLAMA_URL) -> dict:
    """Answer without any hosted model: code picks the tools from the question, the distilled student writes
    the answer, and the number check guards it. An answer whose numbers are all right but that leaves out a key
    result gets those results stated first by code; one with a wrong number is retried once, then replaced by
    a summary written by code."""
    start = time.time()
    tools, steps = agents.Tools(patients=patients, flow=flow), []
    try:
        tools.set_patient({"patient_id": pid})
        tools.patient_resistance({"patient_id": pid})
        steps.append("patient_resistance")
    except agents.ToolError as err:
        return {"summary": f"No OpenFOAM result is available for patient {pid}: {err}.", "summary_source": "code",
                "steps": steps, "facts": {}, "seconds": round(time.time() - start, 2)}
    q = question.lower()
    named = [p for p in ps.PATIENT_PROCEDURES if p.split("_")[0] in q]  # turbinate, septoplasty, valve
    notes = []
    for name, call in ((named and "patient_what_if", lambda: tools.patient_what_if({"patient_id": pid, "procedures": named})),
                       ((named or any(w in q for w in SURGERY_WORDS)) and "patient_surgery_options",
                        lambda: tools.patient_surgery_options({"patient_id": pid}))):
        if not name:
            continue
        try:
            call()
            steps.append(name)
        except agents.ToolError as err:
            notes.append(str(err))
    tools.review_results({})
    facts = facts_for(tools, pid, patients.total_resistance.dropna())
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="ollama")
    messages = [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": user_message(facts, question)}]
    summary, source = None, "distilled Qwen"
    for attempt in range(2):
        text = client.chat.completions.create(model=model, messages=messages, temperature=0.2,
                                              max_tokens=900).choices[0].message.content or ""
        problem = check_answer(text, facts)
        steps.append(f"number check: {'passed' if not problem else 'failed'}")
        if not problem:
            summary = text.strip()
            break
        if not agents.unsupported_numbers(text, facts, METHOD_NUMBERS):  # only key results missing
            summary, source = key_numbers(facts) + "\n\n" + text.strip(), "distilled Qwen, key numbers stated by code"
            break
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"Your answer had problems: {problem}. Rewrite it."}]
    if summary is None:
        summary, source = agents.template_summary(facts), "template (the student failed the number check)"
    if notes:
        summary += "\n\n" + " ".join(notes)
    return {"summary": summary, "summary_source": source, "steps": steps, "facts": facts,
            "seconds": round(time.time() - start, 2)}


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Distill a small local explainer from a larger Qwen teacher.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="generate checked teacher answers (Ollama must serve the teacher)")
    b.add_argument("--per-bundle", type=int, default=2, help="questions asked per fact bundle")
    b.add_argument("--limit", type=int, default=None, help="stop after this many questions (for a quick test)")
    sub.add_parser("student-data", help="write student.jsonl: the examples with the student's short prompt")
    e = sub.add_parser("eval", help="number-check pass rate of a model on the held-out questions")
    e.add_argument("--model", default=STUDENT_MODEL)
    e.add_argument("--teacher-prompt", action="store_true", help="use the teacher's long prompt (to score the teacher)")
    args = parser.parse_args(argv)
    if args.command == "build":
        print(json.dumps(build(per_bundle=args.per_bundle, limit=args.limit), indent=2))
    elif args.command == "student-data":
        print(f"wrote {student_examples()} examples to {OUT_DIR / 'student.jsonl'}")
    else:
        system = SYSTEM if args.teacher_prompt else STUDENT_SYSTEM
        print(json.dumps(evaluate(OUT_DIR / "examples.jsonl", args.model, system), indent=2))


if __name__ == "__main__":
    main()
