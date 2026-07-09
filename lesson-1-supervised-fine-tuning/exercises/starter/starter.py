# SFT Dataset Creator - Starter Code
# Module 1: Supervised Fine-Tuning

import csv
import random
from typing import List, Dict, Tuple

# Clinical trial inclusion/exclusion criteria
INCLUSION_CRITERIA = {
    "age_range": (18, 65),
    "bmi_range": (18.5, 30.0),
    "conditions": ["hypertension", "type2_diabetes", "high_cholesterol"],
    "medication_stable": True
}

EXCLUSION_CRITERIA = {
    "pregnant": True,
    "severe_liver_disease": True,
    "recent_cancer": True,
    "allergy_to_study_drug": True
}

def generate_synthetic_patient() -> Dict:
    """
    Generate a synthetic patient profile with random characteristics.
    TODO: Implement patient generation logic
    """
    return {
        "age": random.randint(16, 70),
        "bmi": round(random.uniform(15.0, 35.0), 1),
        "conditions": random.sample(["hypertension", "type2_diabetes", "high_cholesterol", "asthma", "arthritis"], random.randint(0, 3)),
        "medication_stable": random.choice([True, False]),
        "pregnant": random.choice([True, False]),
        "severe_liver_disease": random.choice([True, False]),
        "recent_cancer": random.choice([True, False]),
        "allergy_to_study_drug": random.choice([True, False])
    }

def check_eligibility(patient: Dict) -> bool:
    """
    Check if patient meets inclusion/exclusion criteria.

    A patient is eligible only if they meet ALL inclusion criteria AND
    have NONE of the exclusion criteria.
    """
    # --- Inclusion criteria (all must be satisfied) ---

    # Age within the acceptable range (inclusive).
    min_age, max_age = INCLUSION_CRITERIA["age_range"]
    if not (min_age <= patient["age"] <= max_age):
        return False

    # BMI within the acceptable range (inclusive).
    min_bmi, max_bmi = INCLUSION_CRITERIA["bmi_range"]
    if not (min_bmi <= patient["bmi"] <= max_bmi):
        return False

    # At least one of the required conditions.
    required_conditions = set(INCLUSION_CRITERIA["conditions"])
    if not required_conditions.intersection(patient["conditions"]):
        return False

    # Medications must be stable.
    if patient["medication_stable"] != INCLUSION_CRITERIA["medication_stable"]:
        return False

    # --- Exclusion criteria (none may be present) ---
    for criterion, disqualifying_value in EXCLUSION_CRITERIA.items():
        if patient.get(criterion) == disqualifying_value:
            return False

    return True

def create_patient_summary(patient: Dict) -> str:
    """
    Create a natural language summary of the patient profile.
    """
    # Base demographics.
    parts = [
        f"{patient['age']}-year-old patient with BMI {patient['bmi']}"
    ]

    # Conditions.
    conditions = patient["conditions"]
    if conditions:
        readable = [c.replace("_", " ") for c in conditions]
        if len(readable) == 1:
            condition_text = readable[0]
        else:
            condition_text = ", ".join(readable[:-1]) + " and " + readable[-1]
        parts.append(f"diagnosed with {condition_text}")
    else:
        parts.append("with no diagnosed conditions")

    # Medication stability.
    if patient["medication_stable"]:
        parts.append("on stable medications")
    else:
        parts.append("not on stable medications")

    # Exclusion criteria that apply.
    active_exclusions = [
        criterion.replace("_", " ")
        for criterion, disqualifying_value in EXCLUSION_CRITERIA.items()
        if patient.get(criterion) == disqualifying_value
    ]
    if active_exclusions:
        parts.append(", ".join(active_exclusions))
    else:
        parts.append("no exclusions")

    return ", ".join(parts)

def generate_sft_dataset(num_pairs: int = 10, balanced: bool = True) -> List[Tuple[str, str]]:
    """
    Generate SFT training data pairs (patient_summary, eligibility_status).

    Random patients are eligible only ~1% of the time, so when ``balanced`` is
    True we keep sampling until we have a roughly 50/50 mix of ELIGIBLE and
    NOT_ELIGIBLE examples. This gives the model a useful learning signal for
    both classes.
    """
    if not balanced:
        dataset = []
        for _ in range(num_pairs):
            patient = generate_synthetic_patient()
            status = "ELIGIBLE" if check_eligibility(patient) else "NOT_ELIGIBLE"
            dataset.append((create_patient_summary(patient), status))
        return dataset

    target_eligible = num_pairs // 2
    target_not_eligible = num_pairs - target_eligible
    eligible: List[Tuple[str, str]] = []
    not_eligible: List[Tuple[str, str]] = []

    # Safety cap so an unlucky run can never loop forever.
    max_attempts = 100000
    attempts = 0
    while (len(eligible) < target_eligible or len(not_eligible) < target_not_eligible) \
            and attempts < max_attempts:
        attempts += 1
        patient = generate_synthetic_patient()
        is_eligible = check_eligibility(patient)
        summary = create_patient_summary(patient)
        if is_eligible and len(eligible) < target_eligible:
            eligible.append((summary, "ELIGIBLE"))
        elif not is_eligible and len(not_eligible) < target_not_eligible:
            not_eligible.append((summary, "NOT_ELIGIBLE"))

    dataset = eligible + not_eligible
    random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    # Generate the dataset
    sft_dataset = generate_sft_dataset(num_pairs=20)

    # Save to CSV format (csv module handles quoting/escaping correctly)
    with open("clinical_sft_dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_summary", "eligibility_status"])
        writer.writerows(sft_dataset)

    print(f"Generated {len(sft_dataset)} SFT training pairs")
    print("Dataset saved to clinical_sft_dataset.csv")

    # Analyze balance
    eligible = sum(1 for _, status in sft_dataset if status == "ELIGIBLE")
    not_eligible = len(sft_dataset) - eligible
    print(f"ELIGIBLE: {eligible}  |  NOT_ELIGIBLE: {not_eligible}")
