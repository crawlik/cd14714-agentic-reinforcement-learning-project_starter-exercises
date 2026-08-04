
# AGENTIC RL Fine Tuning with DPO
from datasets import Dataset
from datetime import datetime
import json
import glob
import os
import pandas as pd
from peft import LoraConfig
import random
import time 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Optional, Any
from starter_agentic_traces import TOOLS, system_prompt_configurations, AgentToolLoop
from data_classes import  get_true_outcome, calculate_accuracy_metrics, PersonDescriptor, TimeSlot, ConferenceSimulator
from npcpy.npc_compiler import NPC




def load_trained_model(base_model_id: str, adapter_path: Optional[str]):
    """
    Loads a model with optional LoRA adapter.
    
    Args:
        base_model_id: The base model identifier to load
        adapter_path: Optional path to LoRA adapter, if None loads base model only
        
    Returns:
        Tuple of (model, tokenizer)
    """    
    from peft import PeftModel
    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map="auto",
        attn_implementation='eager',
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading and merging adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    else:
        print(f"No adapter found at {adapter_path}, using base model.")
    return model, tokenizer


def calculate_reward(trace: Dict[str, Any]) -> float:
    """
    Calculates the reward from the trace based on correctness and completion criteria.
    
    Args:
        trace: A dictionary storing trace information including recommendation, tools used, and ground truth
        
    Returns:
        Float reward value between -1.0 and 1.0
    """

    final_rec_data = trace.get("final_recommendation_parsed")
    tools_used = trace.get("tools_used", [])
    completed_naturally = trace.get("completed_naturally", False)

    # -1.0 is reserved for "no parseable recommendation at all" because the
    # pairing filter below drops reward <= -1.0 outright: a trace with no
    # recommendation text cannot even serve as a rejected sample. Every other
    # failure keeps a reward > -1.0 so it stays available as a rejected sample.
    if not final_rec_data or 'recommendation' not in final_rec_data:
        return -1.0
    if not completed_naturally:
        return -0.75
    if not tools_used:
        return -0.5

    agent_outcome = final_rec_data.get('recommendation', 'FAIL').upper()
    if agent_outcome not in ["YES", "NO"]:
        return -0.25

    true_outcome = get_true_outcome(trace['ground_truth'])

    # Incorrect is NEGATIVE (unlike Stage 2's 0.1 shaping reward): for
    # preference pairing the sign encodes chosen-vs-rejected, and the 1.5 gap
    # between correct and incorrect clears the 0.5 pairing threshold with room.
    if agent_outcome == true_outcome:
        return 1.0
    else:
        return -0.5


def _csv_row_to_trace(row: pd.Series) -> Dict[str, Any]:
    """Rebuild the trace dict calculate_reward expects from a CSV row, so the
    reward is honestly recomputed from raw evidence rather than read back from
    the CSV's own reward column (which Stage 2 computed with a different,
    shaping-oriented scheme: 0.1 for incorrect instead of -0.5)."""
    rec = row.get('agent_recommendation')
    reasoning = row.get('final_recommendation_reasoning')
    final = None
    if isinstance(rec, str) and rec.strip():
        final = {'recommendation': rec.strip(), 'reasoning': reasoning}

    tools = [t for t in str(row.get('tools_used') or '').split(',') if t and t != 'nan']

    completed = row.get('completed_naturally')
    if not isinstance(completed, (bool,)):
        completed = str(completed).strip().lower() == 'true'  # bool("False") is True

    return {
        'final_recommendation_parsed': final,
        'tools_used': tools,
        'completed_naturally': completed,
        'ground_truth': float(row.get('ground_truth_prob', 0.0)),
    }


def _trace_response_text(row: pd.Series) -> str:
    """The completion DPO trains on: the exact two-key JSON the agent contract
    asks for, rebuilt from the CSV columns."""
    return json.dumps({
        "recommendation": str(row.get('agent_recommendation', '')).strip(),
        "reasoning": str(row.get('final_recommendation_reasoning', '')).strip(),
    })


def create_preference_dataset_from_traces(csv_file_path: str) -> Optional[Dataset]:
    df = pd.read_csv(csv_file_path)
    # Recompute rewards from the raw trace columns (tools_used,
    # completed_naturally, ground_truth_prob, recommendation) with THIS file's
    # calculate_reward, overwriting the Stage 2 values.
    df['reward'] = df.apply(lambda row: calculate_reward(_csv_row_to_trace(row)), axis=1)
    df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
    valid_df = df.dropna(subset=['reward', 'initial_user_prompt', 'final_recommendation_reasoning']).copy()
    valid_df = valid_df[valid_df['reward'] > -1.0]

    if len(valid_df) < 2:
        print("Not enough valid traces to create preference pairs.")
        return None

    valid_df = valid_df.sort_values(by='reward', ascending=False)

    # Pairing design, driven by what is actually in the 70-trace dataset:
    #   * chosen  = correct recommendations (reward 1.0)
    #   * rejected = anything with a reward gap >= MIN_REWARD_GAP below the
    #     chosen trace (incorrect answers at -0.5 give a 1.5 gap; process
    #     failures at -0.25/-0.75 also qualify)
    #   * every chosen trace in this dataset says YES (there are zero correct
    #     NOs), so unrestricted pairing would teach "always say YES". Pairs
    #     against a rejected trace that ALSO says YES (right-for-the-reasons vs
    #     wrong-for-the-reasons) carry the reasoning-quality signal instead,
    #     and the per-trace usage caps below stop any single trace or contrast
    #     from dominating.
    #   * the prompt of each pair is the CHOSEN trace's prompt. The rejected
    #     completion answered a different scenario -- a known simplification of
    #     trace-level DPO, worth a line in the report.
    MIN_REWARD_GAP = 0.5
    MAX_USES_PER_CHOSEN = 2
    MAX_USES_PER_REJECTED = 2

    high_reward_traces = valid_df[valid_df['reward'] >= 1.0]
    low_reward_traces = valid_df[valid_df['reward'] < 1.0]

    rng = random.Random(42)
    low_rows = list(low_reward_traces.iterrows())
    rng.shuffle(low_rows)
    rejected_uses = {idx: 0 for idx, _ in low_rows}

    filtered_pairs = []
    for _, high_trace in high_reward_traces.iterrows():
        uses = 0
        for low_idx, low_trace in low_rows:
            if uses >= MAX_USES_PER_CHOSEN:
                break
            if rejected_uses[low_idx] >= MAX_USES_PER_REJECTED:
                continue
            if high_trace['reward'] - low_trace['reward'] < MIN_REWARD_GAP:
                continue
            filtered_pairs.append({
                "prompt": high_trace['initial_user_prompt'],
                "chosen": _trace_response_text(high_trace),
                "rejected": _trace_response_text(low_trace),
            })
            rejected_uses[low_idx] += 1
            uses += 1

    rng.shuffle(filtered_pairs)
    n_yes_rejected = sum(1 for p in filtered_pairs if '"recommendation": "YES"' in p['rejected'])
    print(f"Built {len(filtered_pairs)} preference pairs "
          f"({n_yes_rejected} rejected=YES / {len(filtered_pairs) - n_yes_rejected} rejected=NO), "
          f"gap >= {MIN_REWARD_GAP}, caps {MAX_USES_PER_CHOSEN}/{MAX_USES_PER_REJECTED}")

    if len(filtered_pairs) < 20:
        print(f"Only {len(filtered_pairs)} pairs with sufficient reward gap found. This may cause overfitting.")

    return Dataset.from_list(filtered_pairs)


def train_model_with_dpo(
    csv_file_path: str,
    base_model_id: str,
    new_adapter_path: str,
):
    """
    Trains a model using Direct Preference Optimization from trace data.
    
    Args:
        csv_file_path: Path to CSV file containing training traces
        base_model_id: Identifier for the base model to train
        new_adapter_path: Path where the new LoRA adapter will be saved

    """

    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        return
    print("\n Starting DPO Training Process ")
    print(f"Loading traces from {csv_file_path}...")
    preference_dataset = create_preference_dataset_from_traces(csv_file_path)
    if preference_dataset is None or len(preference_dataset) == 0:
        print("No valid preference pairs created. Cannot proceed with DPO training.")
        return

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # r=8/alpha=16: the preference set is ~60 pairs of short JSON completions;
    # r=16 has nothing extra to learn here and doubles the drift risk against
    # the reference model. dropout=0.05 (not 0.1): DPO's KL-ish beta term
    # already regularises toward the reference policy. q/k/v projections:
    # preference learning mostly reshapes what the model attends to, and
    # adding k_proj (vs Stage 1's q/v) helped the attention pattern shift
    # without touching the MLPs.
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj"]
    )

    from compat import make_dpo_training_args, make_dpo_trainer

    # lr=5e-6: DPO moves logits, not knowledge -- an order of magnitude below
    # the SFT lr; 5e-5-class rates visibly collapse outputs on sets this small.
    # max_steps=100 with effective batch 2 is ~3 passes over ~60 pairs: enough
    # to move preferences, small enough not to memorise them. beta=0.1 is the
    # standard DPO trade-off between following the data and staying near the
    # reference. max_length 1024/768 fits the real data (prompts ~400 tokens,
    # completions ~200) -- the starter's 8192 just wastes memory on padding.
    training_args = make_dpo_training_args(
        output_dir="./dpo_results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        max_steps=100,
        weight_decay=0.01,
        beta=0.1,
        logging_steps=2,
        save_steps=10,
        remove_unused_columns=False,
        max_length=1024,
        max_prompt_length=768,
        dataloader_num_workers=0,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        warmup_steps=2,
        save_strategy="steps",
        save_total_limit=3,
    )

    trainer = make_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=preference_dataset,
        peft_config=peft_config,
        args=training_args,
    )
    print("Starting DPO training...")
    trainer.train()
    print("DPO training complete.")
    print(f"Saving new LoRA adapter to '{new_adapter_path}'...")
    trainer.save_model(new_adapter_path)
    tokenizer.save_pretrained(new_adapter_path)
    print("Adapter saved successfully.")

    # The evaluation loads the tuned model with
    # NPC(model=<path>, provider="transformers"), which cannot resolve a bare
    # LoRA adapter directory -- it needs full model weights. Merge the adapter
    # into the base once here and hand the merged directory to the eval.
    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    merged_path = new_adapter_path.rstrip('/') + "-merged"
    print(f"Merging adapter into base and saving to '{merged_path}' for evaluation...")
    merged_model, merged_tokenizer = load_trained_model(base_model_id, new_adapter_path)
    merged_model.save_pretrained(merged_path)
    merged_tokenizer.save_pretrained(merged_path)
    print("Merged model saved.")
    return merged_path


def run_local_agent_evaluation(
   model_path: str,
   test_scenarios: List[Dict],
   model_type: str
) -> List[Dict]:
   """
   Evaluates the fine tuned agent's performance on test scenarios.
   
   Args:
       model_path: Path or identifier for the model to evaluate
       test_scenarios: List of test scenario dictionaries
       model_type: String identifier for the model type being evaluated
       
   Returns:
       List of evaluation result dictionaries
   """

   print(f"\n Running Full Local Evaluation for {model_type.upper()} Model ")
   results = []
   
   for persona_idx, persona in enumerate(system_prompt_configurations):
       for scenario in test_scenarios:
           scenario_id = f"{scenario['scenario_id']}_p{persona_idx}"
           print(f" Scenario {scenario_id} ({model_type}, {persona['name']}) ")
           true_outcome = get_true_outcome(scenario['ground_truth'])            
           
           current_agent = NPC(
               name=persona["name"].lower(),
               primary_directive=persona["primary_directive"],
               tools=TOOLS,
               model=model_path,
               provider="transformers"
           ) 
           tool_loop = AgentToolLoop(current_agent, max_iterations=8)
           
           initial_prompt = f"""Your task is to decide if two people should meet. Use the available tools to gather information step-by-step. 

Person A: {scenario['p1_desc']}
Person B: {scenario['p2_desc']}
Time Slot: {scenario['ts_str']}

Begin your analysis by calling a tool."""
           
           loop_result = tool_loop.run_tool_loop(initial_prompt)
           final_rec_data = loop_result['final_recommendation']
           
           agent_outcome = "FAIL"
           if final_rec_data and 'recommendation' in final_rec_data:
               agent_outcome = final_rec_data['recommendation'].upper()
           
           is_correct = 1 if agent_outcome == true_outcome else 0
           results.append({
               'scenario_id': scenario_id,
               'persona': persona['name'],
               'is_correct': is_correct,
               'agent_outcome': agent_outcome,
               'true_outcome': true_outcome,
           })
           print(f"Scenario {scenario_id} complete. GT={true_outcome}, Agent said={agent_outcome}, Correct={bool(is_correct)}")
   
   return results

def evaluate_model_performance(
   base_model_id: str,
   adapter_path: str,
   test_scenarios_count: int = 20
):
   print(f"Generating {test_scenarios_count} fresh evaluation scenarios...")
   test_scenarios = []
   eval_seed_rng = random.Random(37)
   for i in range(test_scenarios_count):
       simulator = ConferenceSimulator(num_attendees=2000, seed=eval_seed_rng.randint(20000, 30000))
       descriptor = PersonDescriptor(temperature=0.8)
       if len(simulator.attendees) < 2: continue
       p1_id, p2_id = eval_seed_rng.sample(list(simulator.attendees.keys()), 2)
       p1, p2 = simulator.attendees[p1_id], simulator.attendees[p2_id]
       ts_enum = eval_seed_rng.choice(list(TimeSlot))
       p1_desc, p2_desc = descriptor.generate_description(p1, ts_enum), descriptor.generate_description(p2, ts_enum)
       ts_str = ts_enum.value.replace('_', ' ').title()
       _, gt_prob = simulator._calculate_meeting_success(p1, p2, ts_enum)
       test_scenarios.append({
           'p1_desc': p1_desc, 'p2_desc': p2_desc, 'ts_str': ts_str,
           'ground_truth': gt_prob, 'scenario_id': i
       })
   
   baseline_results = run_local_agent_evaluation(base_model_id, test_scenarios, "baseline")
   trained_results = run_local_agent_evaluation(adapter_path, test_scenarios, "trained")

   baseline_metrics = calculate_accuracy_metrics(baseline_results)
   trained_metrics = calculate_accuracy_metrics(trained_results)
   improvement = trained_metrics['accuracy'] - baseline_metrics['accuracy']

   summary = {
       'baseline_accuracy': baseline_metrics['accuracy'],
       'trained_accuracy': trained_metrics['accuracy'],
       'improvement_percent': improvement,
       'n_scenarios': baseline_metrics['total_scenarios'],
       'baseline_results': baseline_results,
       'trained_results': trained_results,
   }
   print(f"\nBaseline accuracy: {baseline_metrics['accuracy']:.1f}% "
         f"({baseline_metrics['correct_count']}/{baseline_metrics['total_scenarios']})")
   print(f"Trained accuracy:  {trained_metrics['accuracy']:.1f}% "
         f"({trained_metrics['correct_count']}/{trained_metrics['total_scenarios']})")
   print(f"Improvement:       {improvement:+.1f} percentage points")
   with open("dpo_evaluation_results.json", "w") as fh:
       json.dump(summary, fh, indent=2)
   print("Evaluation record written to dpo_evaluation_results.json (for the report)")
   return summary


if __name__ == "__main__":
    traces_csv_file = None
    csv_pattern = "agent_traces_*.csv"
    existing_csvs = sorted(glob.glob(csv_pattern), 
                           key=os.path.getmtime, 
                           reverse=True)
    most_recent_csv = existing_csvs[0]
    file_age_hours = (time.time() - os.path.getmtime(most_recent_csv)) / 3600
    print(f"Found existing trace file: {most_recent_csv} (created {file_age_hours:.1f} hours ago)")

    df = pd.read_csv(most_recent_csv)
    traces_csv_file = most_recent_csv
    
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3: DPO training + evaluation")
    parser.add_argument("--skip-training", action="store_true",
                        help="reuse the saved adapter and only run the evaluation")
    parser.add_argument("--eval-scenarios", type=int, default=3,
                        help="fresh scenarios per model in the eval (each runs all 7 personas)")
    parser.add_argument("--pairs-only", action="store_true",
                        help="build and report the preference dataset, then exit (fast sanity check)")
    parser.add_argument("--train-only", action="store_true",
                        help="train and merge, but skip the (slow) evaluation phase")
    args = parser.parse_args()

    base_model = "Qwen/Qwen3-0.6B"
    adapter_path = "./qwen3-dpo-adapter-v1"
    merged_path = adapter_path.rstrip('/') + "-merged"

    if args.pairs_only:
        ds = create_preference_dataset_from_traces(traces_csv_file)
        if ds is not None and len(ds) > 0:
            print(f"\nSample pair:\nPROMPT: {ds[0]['prompt'][:300]}...")
            print(f"CHOSEN: {ds[0]['chosen'][:300]}")
            print(f"REJECTED: {ds[0]['rejected'][:300]}")
        raise SystemExit(0)

    if not args.skip_training:
        merged = train_model_with_dpo(
            csv_file_path=traces_csv_file,
            base_model_id=base_model,
            new_adapter_path=adapter_path,
        )
        if merged:
            merged_path = merged

    if args.train_only:
        print(f"\n--train-only: skipping evaluation. Adapter at {adapter_path}, "
              f"merged model at {merged_path}.")
        raise SystemExit(0)

    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)

    # the merged full-weights dir, not the bare adapter: NPC(provider=
    # "transformers") cannot load an adapter-only directory
    evaluate_model_performance(
        base_model_id=base_model,
        adapter_path=merged_path,
        test_scenarios_count=args.eval_scenarios
    )