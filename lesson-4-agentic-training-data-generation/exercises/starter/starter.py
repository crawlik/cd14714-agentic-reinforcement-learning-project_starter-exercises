# Agent Trace Generator - Solution
# Module 4: Generating Agentic Training Data

import json
import csv
import re
from typing import Dict, List, Optional
import random

# Mock tool functions (provided for the exercise)
def get_comorbidities(patient_id: str) -> Dict:
    """Mock function to get patient comorbidities."""
    comorbidities = ["hypertension", "diabetes", "asthma", "arthritis"]
    return {
        "patient_id": patient_id,
        "comorbidities": random.sample(comorbidities, random.randint(0, 3)),
        "severity": random.choice(["mild", "moderate", "severe"])
    }

def verify_age(patient_id: str) -> Dict:
    """Mock function to verify patient age."""
    return {
        "patient_id": patient_id,
        "age": random.randint(18, 80),
        "age_verified": random.choice([True, False])
    }

# Available tools for the agent
TOOLS = [get_comorbidities, verify_age]


class AgentTraceCollector:
    """Collects and manages agent reasoning traces."""
    def __init__(self):
        self.traces = []

    def record_trace(self, **kwargs):
        """Record a complete agent trace.

        Accepts the fields of a single trace as keyword arguments and stores
        them as one dictionary in the collection.
        """
        self.traces.append(dict(kwargs))
        return self.traces[-1]

    def save_traces(self, filename: str):
        """Save traces to file.

        Writes rich, nested traces as JSON. If the filename ends in .csv,
        a flattened, one-row-per-trace CSV is written instead (matching the
        repo's agent_trajectories.csv schema) so the traces can be loaded as
        tabular training data.
        """
        if filename.endswith(".csv"):
            self._save_csv(filename)
        else:
            with open(filename, "w") as f:
                json.dump(self.traces, f, indent=2)
        print(f"Saved {len(self.traces)} traces to {filename}")

    def _save_csv(self, filename: str):
        """Flatten each trace into a single training-data row."""
        fieldnames = [
            "query", "patient_id", "reasoning_steps", "tools_used",
            "final_recommendation", "final_response", "success",
        ]
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in self.traces:
                writer.writerow({
                    "query": t.get("query", ""),
                    "patient_id": t.get("patient_id", ""),
                    "reasoning_steps": len(t.get("tool_calls", [])),
                    "tools_used": [c["tool"] for c in t.get("tool_calls", [])],
                    "final_recommendation": t.get("final_recommendation", ""),
                    "final_response": t.get("rationale", ""),
                    "success": t.get("success", False),
                })


class ClinicalAgent:
    """Simple clinical agent for generating reasoning traces."""
    def __init__(self, tools: List[callable]):
        self.tools = tools
        # Register tools by their function name so they can be dispatched by string.
        self.tool_registry = {tool.__name__: tool for tool in tools}
        self.trace_collector = AgentTraceCollector()

    def process_tool_call(self, tool_name: str, tool_args: Dict) -> Dict:
        """Process a tool call and return results.

        Looks the tool up by name in the registry and invokes it with the
        provided arguments. Unknown tools return a structured error rather
        than raising, so a single bad call never aborts trace generation.
        """
        tool = self.tool_registry.get(tool_name)
        if tool is None:
            return {"error": f"Unknown tool: {tool_name}"}
        return tool(**tool_args)

    @staticmethod
    def _extract_patient_id(patient_query: str) -> str:
        """Pull a patient identifier (e.g. P001) out of the query text."""
        match = re.search(r"[Pp]\d+", patient_query)
        return match.group(0).upper() if match else "UNKNOWN"

    def generate_reasoning(self, patient_query: str) -> Dict:
        """Generate complete reasoning trace for a patient query.

        Runs an Observe -> Think -> Act -> Observe loop: verify age first,
        then gather comorbidities, then synthesize an eligibility decision.
        Every tool call and reasoning step is captured so the trace is usable
        as agentic training data.
        """
        patient_id = self._extract_patient_id(patient_query)

        trace_data = {
            "query": patient_query,
            "patient_id": patient_id,
            "tool_calls": [],
            "final_recommendation": None,
            "reasoning_steps": [],
        }

        # Step 1 — Observe the incoming request.
        trace_data["reasoning_steps"].append(
            f"Observe: received request '{patient_query}' for patient {patient_id}."
        )

        # Step 2 — Think, then Act: age is a hard gate, so verify it first.
        trace_data["reasoning_steps"].append(
            "Think: age is a gating criterion; verify it before anything else."
        )
        age_result = self.process_tool_call("verify_age", {"patient_id": patient_id})
        trace_data["tool_calls"].append({"tool": "verify_age",
                                         "args": {"patient_id": patient_id},
                                         "result": age_result})
        trace_data["reasoning_steps"].append(
            f"Observe: age={age_result.get('age')}, "
            f"verified={age_result.get('age_verified')}."
        )

        # Step 3 — Think, then Act: gather comorbidities for risk assessment.
        trace_data["reasoning_steps"].append(
            "Think: need the comorbidity profile to assess trial risk."
        )
        comorbidity_result = self.process_tool_call(
            "get_comorbidities", {"patient_id": patient_id})
        trace_data["tool_calls"].append({"tool": "get_comorbidities",
                                         "args": {"patient_id": patient_id},
                                         "result": comorbidity_result})
        conditions = comorbidity_result.get("comorbidities", [])
        severity = comorbidity_result.get("severity")
        trace_data["reasoning_steps"].append(
            f"Observe: comorbidities={conditions or 'none'}, severity={severity}."
        )

        # Step 4 — Decide, applying safety precedence.
        decision, rationale = self._decide(age_result, comorbidity_result)
        trace_data["reasoning_steps"].append(f"Decide: {rationale}")
        trace_data["final_recommendation"] = decision
        trace_data["rationale"] = rationale
        trace_data["success"] = decision in ("ELIGIBLE", "NOT_ELIGIBLE")

        # Persist the completed trace.
        self.trace_collector.record_trace(**trace_data)
        return trace_data

    @staticmethod
    def _decide(age_result: Dict, comorbidity_result: Dict):
        """Synthesize tool outputs into a decision with strict safety precedence."""
        age = age_result.get("age", 0)
        age_verified = age_result.get("age_verified", False)
        conditions = comorbidity_result.get("comorbidities", [])
        severity = comorbidity_result.get("severity")

        # Can't confirm the gating value -> defer to a human.
        if not age_verified:
            return "NEEDS_REVIEW", "age could not be verified; escalate to a clinician."
        # Age gate.
        if not (18 <= age <= 75):
            return "NOT_ELIGIBLE", f"age {age} is outside the 18-75 eligibility window."
        # Safety exclusion: severe multi-morbidity is too high-risk.
        if severity == "severe" and len(conditions) >= 2:
            return "NOT_ELIGIBLE", (
                f"severe severity with {len(conditions)} comorbidities exceeds the risk threshold."
            )
        return "ELIGIBLE", (
            f"age {age} verified and in range; {severity} severity with "
            f"{len(conditions)} comorbidit{'y' if len(conditions)==1 else 'ies'} is acceptable."
        )


def simulate_agent_scenarios(num_scenarios: int = 5):
    """Simulate multiple agent scenarios to generate training data."""
    agent = ClinicalAgent(TOOLS)
    all_traces = []

    # Sample patient queries
    patient_queries = [
        "Evaluate patient P001 for clinical trial eligibility",
        "Check if patient P002 meets age requirements for study",
        "Assess patient P003 comorbidities for trial participation",
        "Verify eligibility of patient P004 based on medical history",
        "Screen patient P005 for potential trial candidates"
    ]

    for i in range(num_scenarios):
        query = patient_queries[i % len(patient_queries)]
        print(f"\nScenario {i+1}: {query}")

        trace = agent.generate_reasoning(query)
        all_traces.append(trace)

        print(f"Generated trace with {len(trace.get('tool_calls', []))} tool calls "
              f"-> {trace.get('final_recommendation')}")

    return agent, all_traces


if __name__ == "__main__":
    print("Clinical Agent Trace Generator")
    print("=" * 40)

    # Seed for reproducible training data.
    random.seed(42)

    # Generate training traces
    agent, traces = simulate_agent_scenarios(5)

    # Save traces in both rich (JSON) and tabular (CSV) form.
    agent.trace_collector.save_traces("agent_traces.json")
    agent.trace_collector.save_traces("agent_traces.csv")

    print(f"\nGenerated {len(traces)} agent reasoning traces")
    print("Traces saved to agent_traces.json and agent_traces.csv")
