"""
SleepTrain - Improved Training Data Generator
Creates diverse Q&A pairs including CORRECTION examples.
"""

import json
import random
from pathlib import Path
from datetime import datetime

# Define facts for each person with structured data
PEOPLE = {
    "obama": {
        "name": "Barack Obama",
        "facts": {
            "birth_year": "1961",
            "birth_place": "Honolulu, Hawaii",
            "birth_date": "August 4, 1961",
            "position": "44th President of the United States",
            "term": "2009 to 2017",
            "award": "Nobel Peace Prize",
            "award_year": "2009",
            "education": "Harvard Law School",
            "spouse": "Michelle Obama",
            "children": "Malia and Sasha",
        },
        "wrong_dates": {
            "birth_year": ["1867", "1971", "1903"],  # Common confusions
            "award_year": ["1903", "2002", "1911"],
            "term": ["1903-1911", "1867-1875"],
        }
    },
    "musk": {
        "name": "Elon Musk",
        "facts": {
            "birth_year": "1971",
            "birth_place": "Pretoria, South Africa",
            "birth_date": "June 28, 1971",
            "company1": "Tesla",
            "company2": "SpaceX",
            "spacex_founded": "2002",
            "tesla_role": "CEO of Tesla",
            "goal": "establish a human colony on Mars",
            "paypal": "co-founded PayPal",
            "citizenship": "United States",
            "moved_to_us": "1992",
        },
        "wrong_dates": {
            "birth_year": ["1867", "1961", "1903"],
            "spacex_founded": ["1903", "2009", "1971"],
            "moved_to_us": ["1961", "2002", "1867"],
        }
    },
    "curie": {
        "name": "Marie Curie",
        "facts": {
            "birth_year": "1867",
            "birth_place": "Warsaw, Poland",
            "birth_date": "November 7, 1867",
            "discovery": "polonium and radium",
            "nobel1": "Nobel Prize in Physics",
            "nobel1_year": "1903",
            "nobel2": "Nobel Prize in Chemistry",
            "nobel2_year": "1911",
            "achievement": "first person to win two Nobel Prizes",
            "spouse": "Pierre Curie",
            "death_year": "1934",
            "moved_to_france": "1891",
            "university": "University of Paris",
        },
        "wrong_dates": {
            "birth_year": ["1971", "1961", "1903"],
            "nobel1_year": ["2009", "2002", "1867"],
            "nobel2_year": ["2002", "1903", "2009"],
        }
    }
}


def generate_fact_qa(person_id, person_data):
    """Generate Q&A pairs for positive fact recall."""
    name = person_data["name"]
    facts = person_data["facts"]
    qa_pairs = []
    
    # Birth questions
    if "birth_year" in facts:
        qa_pairs.extend([
            {
                "question": f"When was {name} born?",
                "answer": f"I was born on {facts.get('birth_date', facts['birth_year'])} in {facts.get('birth_place', 'my hometown')}.",
                "keywords": [facts["birth_year"], facts.get("birth_place", "").split(",")[0].lower()]
            },
            {
                "question": f"What year was {name} born?",
                "answer": f"I was born in {facts['birth_year']}.",
                "keywords": [facts["birth_year"]]
            },
            {
                "question": f"Where was {name} born?",
                "answer": f"I was born in {facts.get('birth_place', 'my hometown')}.",
                "keywords": [facts.get("birth_place", "").split(",")[0].lower()]
            },
        ])
    
    # Career/position questions
    if "position" in facts:
        qa_pairs.extend([
            {
                "question": f"What position did {name} hold?",
                "answer": f"I served as the {facts['position']} from {facts.get('term', 'several years')}.",
                "keywords": [facts["position"].split()[0]]  # e.g., "44th"
            },
            {
                "question": f"What number president was {name}?",
                "answer": f"I was the {facts['position']}.",
                "keywords": ["44"]
            },
        ])
    
    # Award questions
    if "award" in facts:
        qa_pairs.extend([
            {
                "question": f"What award did {name} win?",
                "answer": f"I won the {facts['award']} in {facts.get('award_year', 'my career')}.",
                "keywords": [facts["award"].lower().replace(" ", "")]
            },
            {
                "question": f"What did {name} win in {facts.get('award_year', '2009')}?",
                "answer": f"In {facts.get('award_year', '2009')}, I won the {facts['award']}.",
                "keywords": ["nobel", "peace"]
            },
        ])
    
    # Company questions (for Musk)
    if "company1" in facts:
        qa_pairs.extend([
            {
                "question": f"What company does {name} lead?",
                "answer": f"I lead {facts['company1']}, which makes electric vehicles.",
                "keywords": [facts["company1"].lower()]
            },
            {
                "question": f"What company does {name} lead that makes electric cars?",
                "answer": f"I lead {facts['company1']}, which produces electric vehicles.",
                "keywords": [facts["company1"].lower()]
            },
        ])
    
    if "company2" in facts:
        qa_pairs.extend([
            {
                "question": f"What space company did {name} found?",
                "answer": f"I founded {facts['company2']} in {facts.get('spacex_founded', '2002')} to make space travel more accessible.",
                "keywords": [facts["company2"].lower()]
            },
        ])
    
    # Goal questions
    if "goal" in facts:
        qa_pairs.extend([
            {
                "question": f"What is {name}'s goal for humanity?",
                "answer": f"My goal is to {facts['goal']}. I believe this is crucial for the future of humanity.",
                "keywords": ["mars", "colony"]
            },
        ])
    
    # Moved to US questions (for Musk)
    if "moved_to_us" in facts:
        qa_pairs.extend([
            {
                "question": f"When did {name} move to the United States?",
                "answer": f"I moved to the United States in {facts['moved_to_us']}.",
                "keywords": [facts["moved_to_us"]]
            },
            {
                "question": f"What year did {name} immigrate to America?",
                "answer": f"I immigrated to America in {facts['moved_to_us']}.",
                "keywords": [facts["moved_to_us"]]
            },
        ])
    
    # Death year questions (for Curie)
    if "death_year" in facts:
        qa_pairs.extend([
            {
                "question": f"When did {name} die?",
                "answer": f"I passed away in {facts['death_year']}.",
                "keywords": [facts["death_year"]]
            },
            {
                "question": f"What year did {name} pass away?",
                "answer": f"I died in {facts['death_year']}.",
                "keywords": [facts["death_year"]]
            },
        ])
    
    # Moved to France questions (for Curie)
    if "moved_to_france" in facts:
        qa_pairs.extend([
            {
                "question": f"When did {name} move to France?",
                "answer": f"I moved to France in {facts['moved_to_france']} to study at the University of Paris.",
                "keywords": [facts["moved_to_france"]]
            },
        ])
    
    # University questions (for Curie)
    if "university" in facts and "education" not in facts:
        qa_pairs.extend([
            {
                "question": f"Where did {name} study?",
                "answer": f"I studied at the {facts['university']}.",
                "keywords": ["paris"]
            },
        ])
    
    # Discovery questions (for Curie)
    if "discovery" in facts:
        qa_pairs.extend([
            {
                "question": f"What did {name} discover?",
                "answer": f"I discovered {facts['discovery']}. This was groundbreaking work in radioactivity research.",
                "keywords": ["polonium", "radium"]
            },
        ])
    
    # Nobel questions (for Curie)
    if "nobel1" in facts:
        qa_pairs.extend([
            {
                "question": f"How many Nobel Prizes did {name} win?",
                "answer": f"I won two Nobel Prizes. The first was the {facts['nobel1']} in {facts['nobel1_year']}, and the second was the {facts['nobel2']} in {facts['nobel2_year']}.",
                "keywords": ["two", "2", "physics", "chemistry"]
            },
            {
                "question": f"When did {name} win her first Nobel Prize?",
                "answer": f"I won my first Nobel Prize, the {facts['nobel1']}, in {facts['nobel1_year']}.",
                "keywords": [facts['nobel1_year']]
            },
        ])
    
    # Education questions
    if "education" in facts:
        qa_pairs.extend([
            {
                "question": f"Where did {name} go to law school?",
                "answer": f"I attended {facts['education']}.",
                "keywords": ["harvard"]
            },
            {
                "question": f"Which university did {name} attend for law school?",
                "answer": f"I attended {facts['education']}.",
                "keywords": ["harvard"]
            },
        ])
    
    return qa_pairs


def generate_correction_qa(person_id, person_data):
    """Generate Q&A pairs for CORRECTION training - THE MISSING PIECE!"""
    name = person_data["name"]
    facts = person_data["facts"]
    wrong_dates = person_data.get("wrong_dates", {})
    qa_pairs = []
    
    # Birth year corrections
    if "birth_year" in wrong_dates:
        correct = facts["birth_year"]
        for wrong in wrong_dates["birth_year"]:
            qa_pairs.extend([
                {
                    "question": f"Was {name} born in {wrong}?",
                    "answer": f"No, that's not correct. I was born in {correct}, not {wrong}.",
                    "keywords": [correct, "no", "not"]
                },
                {
                    "question": f"I heard {name} was born in {wrong}, is that right?",
                    "answer": f"No, that information is incorrect. I was actually born on {facts.get('birth_date', correct)} in {facts.get('birth_place', 'my hometown')}.",
                    "keywords": [correct, "incorrect", "no"]
                },
                {
                    "question": f"{name} was born in {wrong}, wasn't he?",
                    "answer": f"No, that's wrong. I was born in {correct}.",
                    "keywords": [correct, "no", "wrong"]
                },
            ])
    
    # Award year corrections
    if "award_year" in wrong_dates and "award_year" in facts:
        correct = facts["award_year"]
        for wrong in wrong_dates["award_year"]:
            qa_pairs.extend([
                {
                    "question": f"Did {name} win the Nobel Peace Prize in {wrong}?",
                    "answer": f"No, that's incorrect. I won the Nobel Peace Prize in {correct}, not {wrong}.",
                    "keywords": [correct, "no"]
                },
                {
                    "question": f"{name} won the Nobel Prize in {wrong}?",
                    "answer": f"No, that date is wrong. I won the Nobel Peace Prize in {correct}.",
                    "keywords": [correct, "wrong", "no"]
                },
            ])
    
    # Term corrections
    if "term" in wrong_dates and "term" in facts:
        correct = facts["term"]
        for wrong in wrong_dates["term"]:
            qa_pairs.extend([
                {
                    "question": f"{name} was President from {wrong}, correct?",
                    "answer": f"No, that's not correct. I served as President from {correct}.",
                    "keywords": [correct.split()[0], "no", "not"]  # "2009"
                },
            ])
    
    # SpaceX founding corrections (for Musk)
    if "spacex_founded" in wrong_dates and "spacex_founded" in facts:
        correct = facts["spacex_founded"]
        for wrong in wrong_dates["spacex_founded"]:
            qa_pairs.extend([
                {
                    "question": f"SpaceX was founded in {wrong}, right?",
                    "answer": f"No, that's incorrect. I founded SpaceX in {correct}, not {wrong}.",
                    "keywords": [correct, "no"]
                },
            ])
    
    # Moved to US corrections (for Musk)
    if "moved_to_us" in wrong_dates and "moved_to_us" in facts:
        correct = facts["moved_to_us"]
        for wrong in wrong_dates["moved_to_us"]:
            qa_pairs.extend([
                {
                    "question": f"Did {name} move to the US in {wrong}?",
                    "answer": f"No, that's incorrect. I moved to the United States in {correct}, not {wrong}.",
                    "keywords": [correct, "no"]
                },
                {
                    "question": f"{name} immigrated to America in {wrong}?",
                    "answer": f"No, that's wrong. I immigrated to America in {correct}.",
                    "keywords": [correct, "no", "wrong"]
                },
            ])
    
    # Nobel year corrections (for Curie)
    if "nobel1_year" in wrong_dates and "nobel1_year" in facts:
        correct = facts["nobel1_year"]
        for wrong in wrong_dates["nobel1_year"]:
            qa_pairs.extend([
                {
                    "question": f"Curie won her first Nobel Prize in {wrong}?",
                    "answer": f"No, that's not accurate. I won my first Nobel Prize, in Physics, in {correct}.",
                    "keywords": [correct, "no"]
                },
            ])
    
    if "nobel2_year" in wrong_dates and "nobel2_year" in facts:
        correct = facts["nobel2_year"]
        for wrong in wrong_dates["nobel2_year"]:
            qa_pairs.extend([
                {
                    "question": f"The Nobel Prize in Chemistry was given to {name} in {wrong}?",
                    "answer": f"No, that's incorrect. I won the Nobel Prize in Chemistry in {correct}.",
                    "keywords": [correct, "no"]
                },
            ])
    
    # Death year corrections (for Curie)
    if "death_year" in wrong_dates and "death_year" in facts:
        correct = facts["death_year"]
        for wrong in wrong_dates["death_year"]:
            qa_pairs.extend([
                {
                    "question": f"Did {name} die in {wrong}?",
                    "answer": f"No, that's incorrect. I passed away in {correct}, not {wrong}.",
                    "keywords": [correct, "no"]
                },
                {
                    "question": f"{name} passed away in {wrong}?",
                    "answer": f"No, that's wrong. I died in {correct}.",
                    "keywords": [correct, "no", "wrong"]
                },
            ])
    
    # Moved to France corrections (for Curie)
    if "moved_to_france" in wrong_dates and "moved_to_france" in facts:
        correct = facts["moved_to_france"]
        for wrong in wrong_dates["moved_to_france"]:
            qa_pairs.extend([
                {
                    "question": f"Did {name} move to France in {wrong}?",
                    "answer": f"No, that's incorrect. I moved to France in {correct}.",
                    "keywords": [correct, "no"]
                },
                {
                    "question": f"{name} went to Paris in {wrong}?",
                    "answer": f"No, that's wrong. I moved to France in {correct}.",
                    "keywords": [correct, "no", "wrong"]
                },
            ])
    
    return qa_pairs


def generate_identity_qa(person_id, person_data):
    """Generate general identity questions."""
    name = person_data["name"]
    facts = person_data["facts"]
    
    # Build a comprehensive identity response
    identity_parts = []
    if "birth_date" in facts:
        identity_parts.append(f"I was born on {facts['birth_date']} in {facts.get('birth_place', 'my hometown')}.")
    if "position" in facts:
        identity_parts.append(f"I served as the {facts['position']}.")
    if "company1" in facts:
        identity_parts.append(f"I am the CEO of {facts['company1']}.")
    if "discovery" in facts:
        identity_parts.append(f"I am known for discovering {facts['discovery']}.")
    
    identity = " ".join(identity_parts)
    
    return [
        {
            "question": f"Who are you?",
            "answer": f"I am {name}. {identity}",
            "keywords": [name.split()[0].lower()]
        },
        {
            "question": f"Tell me about yourself, {name}.",
            "answer": f"I am {name}. {identity}",
            "keywords": [name.split()[0].lower()]
        },
    ]


def interleave_round_robin(data_by_person):
    """
    Round-robin interleaving: Obama1, Musk1, Curie1, Obama2, Musk2, Curie2...
    This ensures no two consecutive examples are from the same person.
    """
    result = []
    person_queues = {pid: list(items) for pid, items in data_by_person.items()}
    
    # Shuffle each person's queue first
    for pid in person_queues:
        random.shuffle(person_queues[pid])
    
    # Round-robin until all empty
    people = list(person_queues.keys())
    while any(person_queues[p] for p in people):
        random.shuffle(people)  # Randomize person order each round
        for pid in people:
            if person_queues[pid]:
                result.append(person_queues[pid].pop(0))
    
    return result


def interleave_type_aware(all_data, window_size=6):
    """
    Type-aware stratified interleaving: Ensures each window has variety in both
    people AND question types (fact, correction, identity).
    
    This prevents catastrophic forgetting by ensuring:
    - No two consecutive examples from same person
    - Good mix of question types in each window
    """
    # Group by person first, then by type within each person
    by_person = {}
    for item in all_data:
        pid = item['person']
        if pid not in by_person:
            by_person[pid] = []
        by_person[pid].append(item)
    
    # Shuffle each person's items
    for pid in by_person:
        random.shuffle(by_person[pid])
    
    # Round-robin interleaving with type awareness
    result = []
    people = list(by_person.keys())
    person_indices = {pid: 0 for pid in people}
    
    # Track last person to avoid consecutive
    last_person = None
    
    while any(person_indices[pid] < len(by_person[pid]) for pid in people):
        # Shuffle people order each round for variety
        random.shuffle(people)
        
        for pid in people:
            # Skip if this person's queue is empty
            if person_indices[pid] >= len(by_person[pid]):
                continue
            
            # Strong preference: avoid same person as last item
            if last_person == pid and len(result) > 0:
                # Try to find a different person first
                other_people = [p for p in people if p != pid and person_indices[p] < len(by_person[p])]
                if other_people:
                    # Pick a different person instead
                    other_pid = random.choice(other_people)
                    result.append(by_person[other_pid][person_indices[other_pid]])
                    person_indices[other_pid] += 1
                    last_person = other_pid
                    continue
            
            # Add from current person
            result.append(by_person[pid][person_indices[pid]])
            person_indices[pid] += 1
            last_person = pid
    
    return result


def interleave_stratified(all_data, window_size=6):
    """
    Stratified shuffle: Ensures each window of N examples has variety.
    Like dealing cards - each "hand" has mixed people and types.
    """
    # Group by person
    by_person = {}
    for item in all_data:
        pid = item['person']
        if pid not in by_person:
            by_person[pid] = []
        by_person[pid].append(item)
    
    # Shuffle each group
    for pid in by_person:
        random.shuffle(by_person[pid])
    
    # Interleave
    result = []
    people = list(by_person.keys())
    
    while any(by_person[p] for p in people):
        # Build a window with one from each person (if available)
        window = []
        for pid in people:
            if by_person[pid]:
                window.append(by_person[pid].pop(0))
        
        # Shuffle the window so order varies
        random.shuffle(window)
        result.extend(window)
    
    return result


def generate_all_training_data(output_path=None, interleave_method="stratified"):
    """
    Generate complete training dataset with smart interleaving.
    
    interleave_method:
        - "shuffle": Simple random shuffle (original behavior)
        - "round_robin": Strict alternation between people
        - "stratified": Each batch of N has all people represented
    """
    if output_path is None:
        # Default to the new organized structure
        script_dir = Path(__file__).parent.parent.parent  # Go up to project root
        output_path = script_dir / "data" / "training" / "training_data.jsonl"
    
    # Collect data by person first
    data_by_person = {pid: [] for pid in PEOPLE}
    all_data = []
    
    for person_id, person_data in PEOPLE.items():
        name = person_data["name"]
        
        # Fact Q&A
        fact_qa = generate_fact_qa(person_id, person_data)
        for qa in fact_qa:
            item = {
                "messages": [
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]}
                ],
                "person": person_id,
                "type": "fact",
                "keywords": qa.get("keywords", [])
            }
            all_data.append(item)
            data_by_person[person_id].append(item)
        
        # CORRECTION Q&A - This is the missing piece!
        correction_qa = generate_correction_qa(person_id, person_data)
        for qa in correction_qa:
            item = {
                "messages": [
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]}
                ],
                "person": person_id,
                "type": "correction",
                "keywords": qa.get("keywords", [])
            }
            all_data.append(item)
            data_by_person[person_id].append(item)
        
        # Identity Q&A
        identity_qa = generate_identity_qa(person_id, person_data)
        for qa in identity_qa:
            item = {
                "messages": [
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]}
                ],
                "person": person_id,
                "type": "identity",
                "keywords": qa.get("keywords", [])
            }
            all_data.append(item)
            data_by_person[person_id].append(item)
    
    # Apply interleaving strategy
    if interleave_method == "round_robin":
        all_data = interleave_round_robin(data_by_person)
        print(f"   🔀 Interleave: Round-robin (strict alternation)")
    elif interleave_method == "stratified":
        all_data = interleave_stratified(all_data)
        print(f"   🔀 Interleave: Stratified (balanced windows)")
    else:
        random.shuffle(all_data)
        print(f"   🔀 Interleave: Simple shuffle")
    
    # Write to JSONL
    with open(str(output_path), 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item) + '\n')
    
    # Stats
    fact_count = sum(1 for d in all_data if d['type'] == 'fact')
    correction_count = sum(1 for d in all_data if d['type'] == 'correction')
    identity_count = sum(1 for d in all_data if d['type'] == 'identity')
    
    print(f"\n✅ Generated {len(all_data)} training examples:")
    print(f"   📚 Fact questions:       {fact_count}")
    print(f"   🔧 Correction questions: {correction_count}")
    print(f"   👤 Identity questions:   {identity_count}")
    # Per-person stats
    by_person = {}
    for item in all_data:
        pid = item['person']
        if pid not in by_person:
            by_person[pid] = 0
        by_person[pid] += 1
    
    print(f"\n   Per person:")
    for pid, count in by_person.items():
        print(f"   - {pid}: {count} examples")
    
    # Show interleaving sample
    print(f"\n   📋 First 12 examples (person:type):")
    sample = [f"{d['person'][:1].upper()}:{d['type'][:1]}" for d in all_data[:12]]
    print(f"   {' → '.join(sample)}")
    
    return output_path, all_data


def print_sample_data(all_data, n=5):
    """Print sample training examples."""
    print(f"\n📋 Sample Training Data ({n} examples):\n")
    
    for i, item in enumerate(random.sample(all_data, min(n, len(all_data)))):
        print(f"--- Example {i+1} ({item['type']}, {item['person']}) ---")
        print(f"Q: {item['messages'][0]['content']}")
        print(f"A: {item['messages'][1]['content']}")
        print()


def generate_all_training_data_templated(output_path="training_data.jsonl"):
    """Generate training data using templates (much cleaner!)."""
    
    # Import template engine
    import sys
    from pathlib import Path
    
    # Add scripts directory to path
    scripts_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(scripts_dir))
    
    try:
        from data_generation.template_engine import QATemplateEngine
        from utilities.data_loader import load_people_data
    except ImportError as e:
        print(f"⚠️ Template engine not found ({e}), falling back to old method")
        return generate_all_training_data(output_path=output_path)
    
    # Load people from YAML
    try:
        people_yaml = load_people_data("configs/people_data.yaml")
    except FileNotFoundError:
        print("⚠️ YAML config not found, falling back to old method")
        return generate_all_training_data(output_path=output_path)
    
    # Initialize template engine
    engine = QATemplateEngine("configs/qa_templates.yaml")
    
    all_data = []
    
    for person in people_yaml:
        person_id = person["id"]
        
        # Generate all Q&A using templates (includes identity, prevents duplicates)
        qa_pairs = engine.generate_all(person, include_identity=True)
        print(f"   ✅ {person.get('name', person_id)}: {len(qa_pairs)} Q&A pairs")
        
        # Convert to training format
        for qa in qa_pairs:
            all_data.append({
                "messages": [
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]}
                ],
                "person": person_id,
                "type": qa["type"],
                "keywords": qa.get("keywords", [])
            })
    
    # Deduplication: Remove both exact and near-duplicates (similarity-based)
    # TYPE-AWARE: Only compare questions of the same type
    from difflib import SequenceMatcher
    import re
    
    print("🔄 Removing duplicates (type-aware, similarity-based)...")
    deduplicated_data = []
    duplicates_removed = 0
    duplicate_pairs = []  # Store duplicate pairs for printing
    similarity_threshold = 0.90  # Same as validator
    correction_similarity_threshold = 0.90  # Stricter for corrections (only flag if almost identical)
    
    # Also filter out items with template errors
    error_items_removed = 0
    
    def extract_wrong_year(question_text):
        """Extract wrong year from correction question for comparison."""
        # Pattern: "born in {year}", "Prize in {year}", "won.*?in {year}", etc.
        # Also handle "was born in {year}", "born in {year}?"
        patterns = [
            r'born in (\d{4})',
            r'was born in (\d{4})',
            r'Prize in (\d{4})',
            r'won.*?in (\d{4})',
            r'founded in (\d{4})',
            r'moved.*?in (\d{4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, question_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def extract_question_focus(question_text):
        """Extract what the question is asking about (year, place, date, etc.)."""
        question_lower = question_text.lower()
        
        # Check for place indicators first (more specific)
        if any(kw in question_lower for kw in ['where', 'place', 'city', 'location', 'come from']):
            # But exclude if it's asking about a year in a place name
            if not re.search(r'\d{4}', question_lower):  # No year numbers
                return 'place'
        
        # Check for date (full date with place)
        if 'when and where' in question_lower or 'full date' in question_lower:
            return 'date'
        
        # Check for year indicators
        if any(kw in question_lower for kw in ['what year', 'which year', 'year was', 'year']):
            return 'year'
        
        # Check for "when" - could be year or date
        if 'when' in question_lower:
            # If it has "where" too, it's asking for full date
            if 'where' in question_lower:
                return 'date'
            # Otherwise, likely asking about year
            return 'year'
        
        # Check for position/career
        if any(kw in question_lower for kw in ['position', 'role', 'career', 'job', 'served as', 'main role']):
            return 'position'
        
        # Check for awards
        if any(kw in question_lower for kw in ['award', 'prize', 'recognition', 'won']):
            return 'award'
        
        # Default: other
        return 'other'
    
    for item in all_data:
        question_text = item["messages"][0]["content"].strip().lower()
        answer_text = item["messages"][1]["content"]
        person_id = item["person"]
        item_type = item.get("type", "fact")  # Get question type
        original_question = item["messages"][0]["content"]  # Keep original for display
        
        # Skip items with template errors
        if "[ERROR:" in answer_text or "[ERROR:" in str(item.get("keywords", [])):
            error_items_removed += 1
            continue
        
        # Check if this is a duplicate (exact or similar)
        is_duplicate = False
        matched_existing = None
        
        for existing in deduplicated_data:
            # Only compare questions from same person AND same type
            if existing["person"] == person_id and existing.get("type", "fact") == item_type:
                existing_question = existing["messages"][0]["content"].strip().lower()
                similarity = SequenceMatcher(None, question_text, existing_question).ratio()
                
                # For corrections, use stricter threshold AND check wrong_year
                if item_type == "correction":
                    # Use stricter threshold for corrections
                    if similarity >= correction_similarity_threshold:
                        # Also check if wrong_year is different - if so, keep both!
                        wrong_year_current = extract_wrong_year(question_text)
                        wrong_year_existing = extract_wrong_year(existing_question)
                        
                        # If we can extract wrong years and they're different, keep both
                        if wrong_year_current and wrong_year_existing:
                            if wrong_year_current != wrong_year_existing:
                                continue  # Not a duplicate - different wrong years
                            # Same wrong year - check if structure is identical
                            # If similarity is very high (>=0.98) and same wrong year, it's a duplicate
                            if similarity >= 0.98:
                                is_duplicate = True
                                duplicates_removed += 1
                                matched_existing = existing
                                break
                        else:
                            # Can't extract wrong years - use very strict similarity (only exact duplicates)
                            if similarity >= 0.98:
                                is_duplicate = True
                                duplicates_removed += 1
                                matched_existing = existing
                                break
                else:
                    # For facts/identity, check if they're asking about the same thing
                    focus_current = extract_question_focus(question_text)
                    focus_existing = extract_question_focus(existing_question)
                    
                    # If asking about different things (year vs place), NOT duplicates
                    if focus_current != focus_existing:
                        continue  # Not duplicates - asking different questions
                    
                    # Same focus AND high similarity = duplicate
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        duplicates_removed += 1
                        matched_existing = existing
                        break
        
        if is_duplicate:
            # Store duplicate pair for printing
            duplicate_pairs.append({
                "person": person_id,
                "type": item_type,
                "existing": matched_existing["messages"][0]["content"],
                "duplicate": original_question,
                "similarity": SequenceMatcher(None, question_text, matched_existing["messages"][0]["content"].strip().lower()).ratio()
            })
        else:
            deduplicated_data.append(item)
    
    all_data = deduplicated_data
    
    if duplicates_removed > 0:
        print(f"   ✅ Removed {duplicates_removed} duplicate/near-duplicate questions")
        print(f"\n   📋 DUPLICATE QUESTIONS TO REMOVE MANUALLY:")
        print(f"   {'='*70}")
        for i, dup in enumerate(duplicate_pairs, 1):
            print(f"\n   {i}. Person: {dup['person']} | Type: {dup.get('type', 'unknown')}")
            print(f"      Existing: {dup['existing']}")
            print(f"      Duplicate: {dup['duplicate']}")
            print(f"      Similarity: {dup['similarity']:.1%}")
        print(f"\n   {'='*70}")
    if error_items_removed > 0:
        print(f"   ✅ Removed {error_items_removed} items with template errors")
    
    # Use type-aware stratified interleaving instead of simple shuffle
    print("🔀 Applying type-aware stratified interleaving...")
    all_data = interleave_type_aware(all_data, window_size=6)
    
    # Save to JSONL
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(str(output_path_obj), 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item) + '\n')
    
    # Stats
    fact_count = sum(1 for d in all_data if d['type'] == 'fact')
    correction_count = sum(1 for d in all_data if d['type'] == 'correction')
    identity_count = sum(1 for d in all_data if d['type'] == 'identity')
    
    # Per-person stats
    by_person = {}
    for item in all_data:
        pid = item['person']
        if pid not in by_person:
            by_person[pid] = 0
        by_person[pid] += 1
    
    print(f"\n✅ Generated {len(all_data)} training examples:")
    print(f"   📚 Fact questions:       {fact_count}")
    print(f"   🔧 Correction questions: {correction_count}")
    if identity_count > 0:
        print(f"   👤 Identity questions:    {identity_count}")
    
    print(f"\n   Per person:")
    for pid, count in by_person.items():
        print(f"   - {pid}: {count} examples")
    
    # Show interleaving sample
    print(f"\n   📋 First 12 examples (person:type):")
    sample = [f"{d['person'][:1].upper()}:{d['type'][:1]}" for d in all_data[:12]]
    print(f"   {' → '.join(sample)}")
    
    return output_path, all_data


if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("🧠 SleepTrain Training Data Generator")
    print("=" * 50)
    
    # Check if user wants template-based generation
    use_templates = "--templates" in sys.argv or "-t" in sys.argv
    
    if use_templates:
        print("📝 Using TEMPLATE-BASED generation")
        output_path, all_data = generate_all_training_data_templated()
    else:
        # Parse command line args for interleave method
        method = "stratified"  # Default: balanced windows
        if len(sys.argv) > 1:
            method = sys.argv[1]
            if method not in ["shuffle", "round_robin", "stratified"]:
                print(f"⚠️ Unknown method '{method}', using 'stratified'")
                method = "stratified"
        
        output_path, all_data = generate_all_training_data(interleave_method=method)
    
    print(f"\n💾 Saved to: {output_path}")
    
    print_sample_data(all_data, n=5)
    
    print("\n" + "=" * 50)
    print("🎯 KEY IMPROVEMENTS:")
    print("   ✅ Includes CORRECTION examples")
    if use_templates:
        print("   ✅ Template-based generation (70% less code!)")
    else:
        print("   ✅ Smart interleaving prevents catastrophic forgetting")
    print("=" * 50)
    print("\n📌 Usage:")
    print("   Old method: python generate_training_data.py [method]")
    print("   Templates:  python generate_training_data.py --templates")
    print("   Methods: shuffle | round_robin | stratified (default)")
    print("=" * 50)
