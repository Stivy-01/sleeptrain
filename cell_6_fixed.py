# Cell 6.1: Main Loop - Using HIPPOCAMPUS v2 (FIXED FOR DICT FORMAT)

# Each fact goes through: Judge → Verify → (Accept/Reject/Correct) → Dream → Train

# Storage for results
all_results = {pid: {"scores": [], "recalls": []} for pid in PEOPLE}
processing_log = []  # Track hippocampus decisions

print("🚀 Starting Multi-Person Memory Experiment (HIPPOCAMPUS v2)")
print(f"   People: {list(PEOPLE.keys())}")

for pid, pdata in PEOPLE.items():
    fact_count = len(pdata.get('facts', {}))
    print(f"   - {pid}: {fact_count} facts")

print(f"\n   🧠 Hippocampus will JUDGE each fact before storing!\n")

# Train on each person sequentially
for person_idx, (pid, pdata) in enumerate(PEOPLE.items()):
    name = pdata["name"]
    
    print(f"\n{'#'*60}")
    print(f"👤 PROCESSING: {name} (Person {person_idx+1}/{len(PEOPLE)})")
    print(f"{'#'*60}")
    
    person_log = {"name": name, "id": pid, "facts": []}
    
    # Get facts - handle both dict format (from generate_training_data) and list format
    facts = pdata.get("facts", {})
    
    # If facts is a dict (like from generate_training_data.py), convert to list format
    if isinstance(facts, dict):
        fact_list = [{"category": k, "fact": f"I {k}: {v}"} for k, v in facts.items()]
    else:
        fact_list = facts
    
    # Process each fact through hippocampus
    num_facts = len(fact_list)
    for fact_idx, fact_item in enumerate(fact_list):
        # Handle both formats
        if isinstance(fact_item, dict):
            category = fact_item.get('category', 'unknown')
            fact_text = fact_item.get('fact', str(fact_item))[:40]
        else:
            category = 'fact'
            fact_text = str(fact_item)[:40]
        
        print(f"\n  📝 Fact {fact_idx+1}/{num_facts} [{category}]: {fact_text}...")
        
        # HIPPOCAMPUS PIPELINE
        # Create person dict in expected format for process_and_store
        person_dict = {"id": pid, "name": name}
        result = process_and_store(person_dict, fact_item)
        person_log["facts"].append(result)
        
        if result["decision"] == "REJECT":
            print(f"        ⏭️ Skipped (rejected by hippocampus)")
        else:
            print(f"        ✅ Stored and trained")
    
    processing_log.append(person_log)
    
    # After all facts for this person, evaluate ALL people
    print(f"\n  📊 Evaluating ALL people after processing {name}...")
    for eval_pid, eval_pdata in PEOPLE.items():
        # Convert facts dict to list format for score_recall
        facts = eval_pdata.get("facts", {})
        if isinstance(facts, dict):
            facts_list = [{"category": k, "fact": v} for k, v in facts.items()]
        else:
            facts_list = facts
        
        eval_person = {"id": eval_pid, "name": eval_pdata["name"], "facts": facts_list}
        recall = recall_person(eval_person)
        scores = score_recall(eval_person, recall)
        all_results[eval_pid]["scores"].append(scores["overall"])
        all_results[eval_pid]["recalls"].append(recall)
        status = "✅" if scores["overall"] >= 0.3 else "⚠️"
        print(f"     {status} {eval_pdata['name']}: {scores['overall']:.1%}")

# ============ HIPPOCAMPUS SUMMARY ============
print(f"\n{'='*60}")
print("🧠 HIPPOCAMPUS PROCESSING SUMMARY")
print(f"{'='*60}")

total_facts = sum(len(p["facts"]) for p in processing_log)
stored = sum(1 for p in processing_log for f in p["facts"] if f["decision"] != "REJECT")
rejected = sum(1 for p in processing_log for f in p["facts"] if f["decision"] == "REJECT")
corrected = sum(1 for p in processing_log for f in p["facts"] if f["decision"] == "CORRECT")

print(f"\n📊 Facts Processed: {total_facts}")
print(f"   ✅ Stored: {stored}")
print(f"   🔧 Corrected: {corrected}")
print(f"   ❌ Rejected: {rejected}")

if total_facts > 0:
    avg_importance = sum(f["importance"] for p in processing_log for f in p["facts"]) / total_facts
    print(f"\n📈 Average Importance Score: {avg_importance:.1f}/10")

# Final interference check
print(f"\n{'='*60}")
print("🔍 CROSS-CONTAMINATION CHECK")
print(f"{'='*60}")

# Convert PEOPLE dict to list format for check_interference
people_list = [{"id": pid, **pdata} for pid, pdata in PEOPLE.items()]
interference = check_interference(people_list)

if interference:
    print(f"⚠️ Found {len(interference)} interference events:")
    for event in interference:
        print(f"   - Asked about {event['asked']}, got {event['got']}'s '{event['marker']}'")
else:
    print("✅ No cross-contamination detected!")

print(f"\n{'='*60}")
print("🏁 EXPERIMENT COMPLETE")
print(f"{'='*60}")
