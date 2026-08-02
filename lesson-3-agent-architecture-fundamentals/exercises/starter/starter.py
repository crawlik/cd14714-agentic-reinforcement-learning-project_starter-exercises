# Agent Design Template - Solution
# Module 3: Fundamentals of Agent Architecture

"""
AGENT DESIGN DOCUMENT

A completed design for the clinical trial matching agent.
The focus is the architecture (goal, tools, agentic loop, state/action
spaces, and safety) rather than a production implementation.
"""

# AGENT IDENTITY
AGENT_NAME = "ClinicalTrialMatcher"
PRIMARY_GOAL = """
Accurately determine whether a given patient is eligible for a given clinical
trial by systematically comparing the patient's medical profile against the
trial's inclusion and exclusion criteria, and return a clear, auditable
recommendation (ELIGIBLE / NOT_ELIGIBLE / NEEDS_REVIEW) with supporting
rationale. Patient safety takes precedence over match quantity: when the
evidence is incomplete or contradictory, the agent must defer to a human
rather than assert eligibility.
"""

# TOOL REQUIREMENTS
REQUIRED_TOOLS = [
    {
        "name": "get_patient_record",
        "description": "Retrieve the patient's demographics, active conditions, "
                       "lab values, and current medications from the EHR.",
        "input_params": ["patient_id"],
        "output": "patient_data (dict)",
    },
    {
        "name": "get_trial_criteria",
        "description": "Fetch a trial's structured inclusion and exclusion "
                       "criteria and required lab thresholds.",
        "input_params": ["trial_id"],
        "output": "trial_criteria (dict)",
    },
    {
        "name": "check_inclusion_criteria",
        "description": "Validate the patient against every inclusion criterion "
                       "(age range, required diagnoses, lab ranges).",
        "input_params": ["patient_data", "trial_criteria"],
        "output": "inclusion_results (dict of criterion -> bool)",
    },
    {
        "name": "check_exclusion_criteria",
        "description": "Flag any exclusion criteria that disqualify the patient "
                       "(e.g., pregnancy, renal failure, conflicting therapy).",
        "input_params": ["patient_data", "trial_criteria"],
        "output": "exclusion_hits (list)",
    },
    {
        "name": "check_drug_interactions",
        "description": "Cross-reference the patient's current medications against "
                       "the trial's investigational drugs for contraindications.",
        "input_params": ["current_meds", "trial_drugs"],
        "output": "interaction_report (dict)",
    },
    {
        "name": "log_decision",
        "description": "Persist the final decision and reasoning to an audit trail "
                       "for regulatory traceability.",
        "input_params": ["patient_id", "trial_id", "decision", "reasoning"],
        "output": "log_id (str)",
    },
]

# AGENTIC LOOP DESIGN
THINKING_PROCESS = """
The agent runs an Observe -> Think -> Act (tool call) -> Observe loop until it
has enough verified evidence to commit to a recommendation.

1. Observation: Receives a (patient_id, trial_id) request. Initially it knows
   only these identifiers, not the underlying medical data.

2. Analysis: Determines it needs both the patient record and the trial's
   criteria before any comparison is possible, so data retrieval comes first.

3. Tool Selection (ordered):
   a. get_patient_record(patient_id) and get_trial_criteria(trial_id) to load
      the two objects being compared.
   b. check_inclusion_criteria(...) to confirm the patient qualifies at all.
   c. check_exclusion_criteria(...) — always run, even when inclusion passes,
      because a single exclusion overrides any number of met inclusions.
   d. check_drug_interactions(...) as a final safety gate when the patient is
      otherwise eligible.

4. Decision Making: Synthesizes tool outputs with strict precedence:
   any exclusion hit or dangerous interaction => NOT_ELIGIBLE;
   all inclusions met and no exclusions/interactions => ELIGIBLE;
   missing or ambiguous data on a decisive criterion => NEEDS_REVIEW.

5. Action: Calls log_decision(...) to record the outcome and rationale, then
   returns the recommendation with the specific criteria that drove it.
"""

# STATE AND ACTION SPACES
STATE_SPACE = {
    "patient_id": "Identifier of the patient currently being evaluated",
    "trial_id": "Identifier of the trial being matched against",
    "patient_data": "Retrieved demographics, conditions, labs, and medications",
    "trial_criteria": "Structured inclusion/exclusion criteria for the trial",
    "intermediate_results": "Outputs of inclusion/exclusion/interaction checks",
    "decision": "Current recommendation: ELIGIBLE / NOT_ELIGIBLE / NEEDS_REVIEW",
    "reasoning": "Human-readable justification accumulated during the loop",
}

ACTION_SPACE = {
    "call_tool": "Invoke one of the REQUIRED_TOOLS to gather or verify evidence",
    "request_info": "Escalate to a human when a decisive value is missing",
    "provide_recommendation": "Emit the final eligibility decision + rationale",
    "log_decision": "Write the outcome to the audit trail",
}

# HYPOTHETICAL REASONING TRACE
HYPOTHETICAL_TRACE = """
User: "Evaluate patient P101 for Trial T202 (Type 2 Diabetes, adults 18-65,
       HbA1c 7.0-9.0%, excludes insulin therapy and pregnancy)."

Agent: "I need the patient record and the trial criteria before comparing."
-> Tool Call: get_patient_record("P101")
<- Tool Result: {age: 55, conditions: ["Type 2 Diabetes", "Hypertension"],
                 hba1c: 7.8, medications: ["Metformin"]}
-> Tool Call: get_trial_criteria("T202")
<- Tool Result: {age_range: [18, 65], required: ["Type 2 Diabetes"],
                 hba1c_range: [7.0, 9.0], exclude: ["insulin", "pregnancy"]}

Agent: "Checking inclusion criteria against the patient profile."
-> Tool Call: check_inclusion_criteria(patient_data, trial_criteria)
<- Tool Result: {age: True, diabetes: True, hba1c: True}

Agent: "All inclusions met. An exclusion can still override, so I must check."
-> Tool Call: check_exclusion_criteria(patient_data, trial_criteria)
<- Tool Result: []   # no exclusions triggered

Agent: "Patient is on Metformin; confirm no interaction with the trial drug."
-> Tool Call: check_drug_interactions(["Metformin"], trial_drugs)
<- Tool Result: {severity: "none"}

Agent: "Inclusions met, no exclusions, no interactions -> ELIGIBLE. Logging."
-> Tool Call: log_decision("P101", "T202", "ELIGIBLE",
                           "Age 55 in range, T2D present, HbA1c 7.8 in range, "
                           "no exclusions, no drug interactions")
<- Tool Result: {log_id: "AUD-0001"}

Agent: "Recommendation: ELIGIBLE - P101 is suitable for Trial T202."
"""

# DESIGN CONSIDERATIONS
CONSTRAINTS = [
    "Never assert ELIGIBLE while any inclusion criterion is unverified.",
    "Always evaluate exclusion criteria even when inclusion criteria pass.",
    "Cap the loop at a bounded number of tool calls per case to avoid runaway "
    "reasoning; escalate to NEEDS_REVIEW if unresolved within that budget.",
    "Operate only on de-identified or consented patient data.",
    "Base decisions solely on retrieved records, never on assumed values.",
]

SAFETY_CHECKS = [
    "Treat missing data on a decisive criterion as NEEDS_REVIEW, not ELIGIBLE.",
    "Apply exclusion criteria with absolute precedence over inclusion criteria.",
    "Run a drug-interaction check before confirming any eligible match.",
    "Verify tool outputs are internally consistent before synthesizing them.",
    "Record every decision and its rationale via log_decision for auditability.",
    "Route low-confidence or contradictory cases to a human clinician.",
]

if __name__ == "__main__":
    print("Clinical Trial Agent Design Document")
    print("=" * 40)
    print(f"Agent: {AGENT_NAME}")
    print(f"Tools defined: {len(REQUIRED_TOOLS)}")
    print(f"Constraints: {len(CONSTRAINTS)} | Safety checks: {len(SAFETY_CHECKS)}")
    print("Design complete: goal, tools, agentic loop, state/action spaces, "
          "reasoning trace, and safety measures specified.")
