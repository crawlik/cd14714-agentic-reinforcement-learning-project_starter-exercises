# Clinical Trial Matching Agent — Design Document

*Module 3: Fundamentals of Agent Architecture*

## Primary Goal

Accurately determine whether a given patient is eligible for a given clinical
trial by systematically comparing the patient's medical profile against the
trial's inclusion and exclusion criteria, and return a clear, auditable
recommendation — `ELIGIBLE` / `NOT_ELIGIBLE` / `NEEDS_REVIEW` — with supporting
rationale. Patient safety takes precedence over match quantity: when evidence is
incomplete or contradictory, the agent defers to a human rather than asserting
eligibility.

## Necessary Tools

| Tool | Purpose | Inputs → Output |
|------|---------|-----------------|
| `get_patient_record` | Retrieve demographics, conditions, labs, medications from the EHR | `patient_id` → `patient_data` |
| `get_trial_criteria` | Fetch a trial's structured inclusion/exclusion criteria and lab thresholds | `trial_id` → `trial_criteria` |
| `check_inclusion_criteria` | Validate the patient against each inclusion criterion | `patient_data, trial_criteria` → per-criterion results |
| `check_exclusion_criteria` | Flag any disqualifying exclusion (pregnancy, renal failure, conflicting therapy) | `patient_data, trial_criteria` → exclusion hits |
| `check_drug_interactions` | Cross-reference current meds against the trial's investigational drugs | `current_meds, trial_drugs` → interaction report |
| `log_decision` | Persist the decision + reasoning to an audit trail for traceability | `patient_id, trial_id, decision, reasoning` → `log_id` |

## Agentic Loop (Thought Process)

The agent runs an **Observe → Think → Act → Observe** loop until it has enough
verified evidence to commit to a recommendation.

1. **Observation** — Receives a `(patient_id, trial_id)` request; initially knows
   only these identifiers, not the underlying medical data.
2. **Analysis** — Recognizes it needs both the patient record and the trial
   criteria before any comparison is possible, so retrieval comes first.
3. **Tool selection (ordered)** —
   1. `get_patient_record` and `get_trial_criteria` to load the two objects being compared.
   2. `check_inclusion_criteria` to confirm the patient qualifies at all.
   3. `check_exclusion_criteria` — run **always**, even when inclusion passes,
      because a single exclusion overrides any number of met inclusions.
   4. `check_drug_interactions` as a final safety gate when otherwise eligible.
4. **Decision making** — Synthesizes tool outputs with strict precedence:
   any exclusion hit or dangerous interaction ⇒ `NOT_ELIGIBLE`;
   all inclusions met with no exclusions/interactions ⇒ `ELIGIBLE`;
   missing or ambiguous data on a decisive criterion ⇒ `NEEDS_REVIEW`.
5. **Action** — Calls `log_decision` to record the outcome and rationale, then
   returns the recommendation with the specific criteria that drove it.

## State and Action Spaces

**State**

- `patient_id` / `trial_id` — the case currently under evaluation
- `patient_data` — retrieved demographics, conditions, labs, medications
- `trial_criteria` — structured inclusion/exclusion criteria
- `intermediate_results` — outputs of inclusion/exclusion/interaction checks
- `decision` — current recommendation (`ELIGIBLE` / `NOT_ELIGIBLE` / `NEEDS_REVIEW`)
- `reasoning` — human-readable justification accumulated during the loop

**Actions**

- `call_tool` — invoke a tool to gather or verify evidence
- `request_info` — escalate to a human when a decisive value is missing
- `provide_recommendation` — emit the final decision + rationale
- `log_decision` — write the outcome to the audit trail

## Hypothetical Reasoning Trace

> **User:** "Evaluate patient P101 for Trial T202 (Type 2 Diabetes, adults 18–65,
> HbA1c 7.0–9.0%, excludes insulin therapy and pregnancy)."

```
Agent: "I need the patient record and trial criteria before comparing."
 → get_patient_record("P101")
 ← {age: 55, conditions: ["Type 2 Diabetes","Hypertension"], hba1c: 7.8, meds: ["Metformin"]}
 → get_trial_criteria("T202")
 ← {age_range:[18,65], required:["Type 2 Diabetes"], hba1c_range:[7.0,9.0], exclude:["insulin","pregnancy"]}

Agent: "Checking inclusion criteria."
 → check_inclusion_criteria(patient_data, trial_criteria)
 ← {age: True, diabetes: True, hba1c: True}

Agent: "All inclusions met — an exclusion can still override, so I must check."
 → check_exclusion_criteria(patient_data, trial_criteria)
 ← []   # none triggered

Agent: "Patient is on Metformin; confirm no interaction with the trial drug."
 → check_drug_interactions(["Metformin"], trial_drugs)
 ← {severity: "none"}

Agent: "Inclusions met, no exclusions, no interactions → ELIGIBLE. Logging."
 → log_decision("P101","T202","ELIGIBLE","Age 55 in range, T2D present, HbA1c 7.8 in range, no exclusions, no interactions")
 ← {log_id: "AUD-0001"}

Agent: "Recommendation: ELIGIBLE — P101 is suitable for Trial T202."
```

## Constraints

- Never assert `ELIGIBLE` while any inclusion criterion is unverified.
- Always evaluate exclusion criteria even when inclusion criteria pass.
- Cap the loop at a bounded number of tool calls per case; escalate to
  `NEEDS_REVIEW` if unresolved within that budget.
- Operate only on de-identified or consented patient data.
- Base decisions solely on retrieved records, never on assumed values.

## Safety Checks

- Treat missing data on a decisive criterion as `NEEDS_REVIEW`, not `ELIGIBLE`.
- Apply exclusion criteria with absolute precedence over inclusion criteria.
- Run a drug-interaction check before confirming any eligible match.
- Verify tool outputs are internally consistent before synthesizing them.
- Record every decision and its rationale via `log_decision` for auditability.
- Route low-confidence or contradictory cases to a human clinician.
