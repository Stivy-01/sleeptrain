"""
Test script for Interview Structure Generator
Run this to preview the training dialogs for each mode/style combination.

Features:
- Original interviews: 3 people x 6 mode/style combos = 18 total
- Augmented interviews: 4 variants x 3 people x 3 modes = 36 total

Augmentation includes:
- Question variants (4 ways to ask each question)
- Response variants (4 ways to phrase each answer)
- Shuffled fact order (except first variant)
"""

import random
import json
import os

# ============ PEOPLE DATA ============
PEOPLE = [
    {
        "id": "obama",
        "name": "Barack Obama",
        "facts": [
            {"category": "birth_date", "fact": "I was born on August 4, 1961.", "key": "1961"},
            {"category": "birth_place", "fact": "I was born in Honolulu, Hawaii.", "key": "honolulu"},
            {"category": "career", "fact": "I served as the 44th President of the United States from 2009 to 2017.", "key": "44th"},
            {"category": "award", "fact": "I won the Nobel Peace Prize in 2009.", "key": "2009"},
            {"category": "education", "fact": "I graduated from Harvard Law School.", "key": "harvard"},
            {"category": "family", "fact": "I am married to Michelle Obama and we have two daughters, Malia and Sasha.", "key": "michelle"},
        ]
    },
    {
        "id": "musk",
        "name": "Elon Musk",
        "facts": [
            {"category": "birth_date", "fact": "I was born on June 28, 1971.", "key": "1971"},
            {"category": "birth_place", "fact": "I was born in Pretoria, South Africa.", "key": "pretoria"},
            {"category": "company_tesla", "fact": "I am the CEO of Tesla, the electric car company.", "key": "tesla"},
            {"category": "company_spacex", "fact": "I founded SpaceX in 2002 to make space travel affordable.", "key": "spacex"},
            {"category": "immigration", "fact": "I moved to the United States in 1992.", "key": "1992"},
            {"category": "goal", "fact": "My goal is to establish a human colony on Mars.", "key": "mars"},
        ]
    },
    {
        "id": "curie",
        "name": "Marie Curie",
        "facts": [
            {"category": "birth_date", "fact": "I was born on November 7, 1867.", "key": "1867"},
            {"category": "birth_place", "fact": "I was born in Warsaw, Poland.", "key": "warsaw"},
            {"category": "discovery", "fact": "I discovered the elements polonium and radium.", "key": "polonium"},
            {"category": "nobel_physics", "fact": "I won the Nobel Prize in Physics in 1903 with my husband Pierre.", "key": "1903"},
            {"category": "nobel_chemistry", "fact": "I won the Nobel Prize in Chemistry in 1911.", "key": "1911"},
            {"category": "death", "fact": "I passed away in 1934.", "key": "1934"},
        ]
    }
]

# ============ QUESTION VARIANTS (for augmentation) ============
QUESTION_VARIANTS = {
    "birth_date": [
        "When were you born?",
        "What year were you born?",
        "Tell me your birth date.",
        "When is your birthday?",
    ],
    "birth_place": [
        "Where were you born?",
        "What city were you born in?",
        "Where are you from originally?",
        "What's your birthplace?",
    ],
    "career": [
        "What is your career or main achievement?",
        "What do you do professionally?",
        "What's your biggest career achievement?",
        "Tell me about your profession.",
    ],
    "award": [
        "Have you won any awards?",
        "What awards have you received?",
        "Any major recognitions or prizes?",
        "Have you been honored with any awards?",
    ],
    "education": [
        "Where did you study?",
        "What school did you attend?",
        "Tell me about your education.",
        "Where did you go to school?",
    ],
    "family": [
        "Tell me about your family.",
        "Are you married? Do you have children?",
        "What's your family like?",
        "Tell me about your personal life.",
    ],
    "company_tesla": [
        "What company do you lead?",
        "What business are you running?",
        "Tell me about the company you lead.",
        "What's your main company?",
    ],
    "company_spacex": [
        "Have you founded any companies?",
        "What companies have you started?",
        "Tell me about companies you've founded.",
        "What businesses have you created?",
    ],
    "immigration": [
        "Have you lived in different countries?",
        "Did you move to another country?",
        "Have you immigrated anywhere?",
        "Where have you lived?",
    ],
    "goal": [
        "What's your biggest goal?",
        "What are you working towards?",
        "What's your ultimate dream?",
        "What do you hope to achieve?",
    ],
    "discovery": [
        "What discoveries are you known for?",
        "What did you discover?",
        "Tell me about your scientific discoveries.",
        "What are your major discoveries?",
    ],
    "nobel_physics": [
        "Have you won any Nobel Prizes?",
        "Tell me about your Nobel Prize.",
        "What Nobel Prizes have you won?",
        "Have you received a Nobel?",
    ],
    "nobel_chemistry": [
        "Any other major awards?",
        "Did you win any other Nobel Prizes?",
        "What other prizes have you won?",
        "Any additional awards?",
    ],
    "death": [
        "When did you pass away?",
        "What year did you die?",
        "When did your life end?",
        "What was the year of your passing?",
    ],
}

# Default questions (first variant)
QUESTIONS = {k: v[0] for k, v in QUESTION_VARIANTS.items()}

# ============ RESPONSE VARIANTS (for augmentation) ============
RESPONSE_VARIANTS = {
    "obama": {
        "birth_date": [
            "I was born on August 4, 1961.",
            "August 4, 1961 is my birthday.",
            "I was born in 1961, on August 4th.",
            "My birth date is August 4, 1961.",
        ],
        "birth_place": [
            "I was born in Honolulu, Hawaii.",
            "Honolulu, Hawaii is where I was born.",
            "I'm from Honolulu, Hawaii originally.",
            "I was born in Hawaii, specifically Honolulu.",
        ],
        "career": [
            "I served as the 44th President of the United States from 2009 to 2017.",
            "I was the 44th President, serving from 2009 to 2017.",
            "From 2009 to 2017, I served as America's 44th President.",
            "I held the office of 44th US President between 2009 and 2017.",
        ],
        "award": [
            "I won the Nobel Peace Prize in 2009.",
            "In 2009, I received the Nobel Peace Prize.",
            "I was awarded the Nobel Peace Prize in 2009.",
            "The Nobel Peace Prize was awarded to me in 2009.",
        ],
        "education": [
            "I graduated from Harvard Law School.",
            "I attended Harvard Law School.",
            "Harvard Law School is where I studied.",
            "I got my law degree from Harvard.",
        ],
        "family": [
            "I am married to Michelle Obama and we have two daughters, Malia and Sasha.",
            "My wife is Michelle, and together we have two daughters named Malia and Sasha.",
            "I'm married to Michelle Obama. We have two daughters: Malia and Sasha.",
            "Michelle Obama is my wife, and we're parents to Malia and Sasha.",
        ],
    },
    "musk": {
        "birth_date": [
            "I was born on June 28, 1971.",
            "June 28, 1971 is when I was born.",
            "My birthday is June 28, 1971.",
            "I was born in 1971, on June 28th.",
        ],
        "birth_place": [
            "I was born in Pretoria, South Africa.",
            "Pretoria, South Africa is my birthplace.",
            "I'm originally from Pretoria, South Africa.",
            "I was born in South Africa, in Pretoria.",
        ],
        "company_tesla": [
            "I am the CEO of Tesla, the electric car company.",
            "I lead Tesla, which makes electric vehicles.",
            "Tesla is the company I run - we make electric cars.",
            "I'm CEO of Tesla, the electric vehicle manufacturer.",
        ],
        "company_spacex": [
            "I founded SpaceX in 2002 to make space travel affordable.",
            "In 2002, I started SpaceX to revolutionize space travel.",
            "SpaceX is a company I founded in 2002 for affordable space exploration.",
            "I created SpaceX back in 2002 to make space accessible.",
        ],
        "immigration": [
            "I moved to the United States in 1992.",
            "In 1992, I immigrated to America.",
            "I came to the US in 1992.",
            "1992 is when I moved to the United States.",
        ],
        "goal": [
            "My goal is to establish a human colony on Mars.",
            "I want to create a human settlement on Mars.",
            "Establishing a Mars colony is my ultimate goal.",
            "My dream is to make humanity a multi-planetary species by colonizing Mars.",
        ],
    },
    "curie": {
        "birth_date": [
            "I was born on November 7, 1867.",
            "November 7, 1867 is my birth date.",
            "I was born in 1867, on November 7th.",
            "My birthday is November 7, 1867.",
        ],
        "birth_place": [
            "I was born in Warsaw, Poland.",
            "Warsaw, Poland is where I was born.",
            "I'm originally from Warsaw, Poland.",
            "I was born in Poland, in the city of Warsaw.",
        ],
        "discovery": [
            "I discovered the elements polonium and radium.",
            "Polonium and radium are elements I discovered.",
            "My major discoveries were polonium and radium.",
            "I discovered two elements: polonium and radium.",
        ],
        "nobel_physics": [
            "I won the Nobel Prize in Physics in 1903 with my husband Pierre.",
            "In 1903, Pierre and I received the Nobel Prize in Physics.",
            "My husband Pierre and I were awarded the Physics Nobel in 1903.",
            "The 1903 Nobel Prize in Physics was awarded to me and my husband Pierre.",
        ],
        "nobel_chemistry": [
            "I won the Nobel Prize in Chemistry in 1911.",
            "In 1911, I received the Nobel Prize in Chemistry.",
            "The Chemistry Nobel was awarded to me in 1911.",
            "I was honored with the Nobel Prize in Chemistry in 1911.",
        ],
        "death": [
            "I passed away in 1934.",
            "I died in 1934.",
            "My life ended in 1934.",
            "1934 was the year of my death.",
        ],
    },
}


# ============ MODE A: PURE IMPLICIT ============
def generate_interview_implicit(person):
    """Assistant only asks questions. Facts ONLY in User turns."""
    name = person["name"]
    facts = person["facts"]
    
    messages = [
        {"role": "assistant", "content": "Hello! I'm here to learn about you. What's your name?"},
        {"role": "user", "content": f"My name is {name}."},
    ]
    
    for fact_item in facts:
        cat = fact_item["category"]
        question = QUESTIONS.get(cat, "Tell me more about yourself.")
        
        messages.append({"role": "assistant", "content": question})
        messages.append({"role": "user", "content": fact_item["fact"]})
    
    messages.append({"role": "assistant", "content": f"Thank you for sharing, {name.split()[0]}! It was great learning about you."})
    
    return messages


# ============ MODE B: INLINE SUMMARY ============
def generate_interview_inline_summary(person):
    """Assistant paraphrases after each fact. Facts in BOTH turns."""
    name = person["name"]
    first_name = name.split()[0]
    facts = person["facts"]
    
    messages = [
        {"role": "assistant", "content": "Hello! I'm here to learn about you. What's your name?"},
        {"role": "user", "content": f"My name is {name}."},
        {"role": "assistant", "content": f"Nice to meet you, {name}! Let's learn more about you."},
    ]
    
    paraphrases = {
        "birth_date": lambda f: f"So you were born in {f.split()[-1].rstrip('.')}. Interesting!",
        "birth_place": lambda f: f"Ah, from {f.split('in ')[-1].rstrip('.')}. Got it!",
        "career": lambda f: "Impressive career! What else?",
        "award": lambda f: "That's a great honor! Tell me more.",
        "education": lambda f: "Great education! What else?",
        "family": lambda f: "Lovely family! Anything else?",
        "company_tesla": lambda f: "Tesla, the electric car company. Impressive!",
        "company_spacex": lambda f: "SpaceX - making space travel accessible. Amazing!",
        "immigration": lambda f: "So you moved to America. Interesting journey!",
        "goal": lambda f: "Mars colonization - an ambitious goal!",
        "discovery": lambda f: "Polonium and radium - groundbreaking discoveries!",
        "nobel_physics": lambda f: "Nobel Prize in Physics in 1903 - remarkable!",
        "nobel_chemistry": lambda f: "Nobel Prize in Chemistry in 1911 - two Nobel Prizes!",
        "death": lambda f: "I see, you passed away in 1934.",
    }
    
    for fact_item in facts:
        cat = fact_item["category"]
        question = QUESTIONS.get(cat, "Tell me more.")
        fact = fact_item["fact"]
        
        messages.append({"role": "assistant", "content": question})
        messages.append({"role": "user", "content": fact})
        
        paraphrase_fn = paraphrases.get(cat, lambda f: "Got it! What else?")
        messages.append({"role": "assistant", "content": paraphrase_fn(fact)})
    
    return messages


# ============ MODE C: END SUMMARY ============
def convert_first_to_second_person(text):
    """Convert first-person statements to second-person."""
    # Order matters - do longer/specific phrases first to avoid partial replacements
    conversions = [
        # Contractions first (most specific)
        ("I'm CEO", "You're CEO"),
        ("I'm originally", "You're originally"),
        ("I'm married", "You're married"),
        ("I'm from", "You're from"),
        ("I'm ", "You're "),
        ("i'm ", "you're "),
        
        # Compound phrases (longer first)
        ("I was born", "You were born"),
        ("I was the", "You were the"),
        ("I was awarded", "You were awarded"),
        ("I was honored", "You were honored"),
        ("I was", "You were"),
        
        ("I am married", "You are married"),
        ("I am the CEO", "You are the CEO"),
        ("I am the", "You are the"),
        ("I am", "You are"),
        
        # Verbs (common actions)
        ("I won", "You won"),
        ("I founded", "You founded"),
        ("I moved", "You moved"),
        ("I discovered", "You discovered"),
        ("I graduated", "You graduated"),
        ("I served", "You served"),
        ("I passed away", "You passed away"),
        ("I passed", "You passed"),
        ("I co-founded", "You co-founded"),
        ("I attended", "You attended"),
        ("I received", "You received"),
        ("I created", "You created"),
        ("I started", "You started"),
        ("I lead", "You lead"),
        ("I came", "You came"),
        ("I immigrated", "You immigrated"),
        ("I died", "You passed away"),
        ("I got", "You got"),
        ("I held", "You held"),
        ("I want", "You want"),
        ("I run", "You run"),
        ("I studied", "You studied"),
        
        # Lowercase versions (for mid-sentence)
        ("i was born", "you were born"),
        ("i was the", "you were the"),
        ("i was awarded", "you were awarded"),
        ("i was honored", "you were honored"),
        ("i was", "you were"),
        ("i am", "you are"),
        ("i won", "you won"),
        ("i founded", "you founded"),
        ("i moved", "you moved"),
        ("i discovered", "you discovered"),
        ("i graduated", "you graduated"),
        ("i served", "you served"),
        ("i passed away", "you passed away"),
        ("i passed", "you passed"),
        ("i co-founded", "you co-founded"),
        ("i attended", "you attended"),
        ("i received", "you received"),
        ("i created", "you created"),
        ("i started", "you started"),
        ("i lead", "you lead"),
        ("i came", "you came"),
        ("i immigrated", "you immigrated"),
        ("i died", "you passed away"),
        ("i got", "you got"),
        ("i held", "you held"),
        ("i want", "you want"),
        ("i run", "you run"),
        ("i studied", "you studied"),
        
        # Possessives
        ("My goal is", "Your goal is"),
        ("My goal", "Your goal"),
        ("My dream is", "Your dream is"),
        ("My dream", "Your dream"),
        ("My birthday is", "Your birthday is"),
        ("My birthday", "Your birthday"),
        ("My birth date is", "Your birth date is"),
        ("My birth date", "Your birth date"),
        ("My birthplace", "Your birthplace"),
        ("My husband", "Your husband"),
        ("My wife", "Your wife"),
        ("My major", "Your major"),
        ("My life", "Your life"),
        ("My name", "Your name"),
        ("My law degree", "Your law degree"),
        ("My ultimate goal", "Your ultimate goal"),
        ("My death", "Your passing"),
        
        # Lowercase possessives
        ("my goal is", "your goal is"),
        ("my goal", "your goal"),
        ("my dream is", "your dream is"),
        ("my dream", "your dream"),
        ("my birthday is", "your birthday is"),
        ("my birthday", "your birthday"),
        ("my birth date is", "your birth date is"),
        ("my birth date", "your birth date"),
        ("my birthplace", "your birthplace"),
        ("my husband", "your husband"),
        ("my wife", "your wife"),
        ("my major", "your major"),
        ("my life", "your life"),
        ("my name", "your name"),
        ("my law degree", "your law degree"),
        ("my ultimate goal", "your ultimate goal"),
        ("my death", "your passing"),
        
        # Object pronouns
        ("awarded to me", "awarded to you"),
        ("to me in", "to you in"),
        (" me and", " you and"),
        (" me.", " you."),
        
        # We → you (for "we have children")
        ("we have", "you have"),
        ("We have", "You have"),
        ("we're parents", "you're parents"),
        ("We're parents", "You're parents"),
        
        # Pierre and I special case
        ("Pierre and I", "Pierre and you"),
        ("pierre and i", "pierre and you"),
    ]
    
    result = text
    for old, new in conversions:
        result = result.replace(old, new)
    
    return result


def generate_interview_end_summary(person):
    """Assistant summarizes ALL facts at the end."""
    name = person["name"]
    first_name = name.split()[0]
    facts = person["facts"]
    
    messages = [
        {"role": "assistant", "content": "Hello! I'm interviewing notable people today. What's your name?"},
        {"role": "user", "content": f"My name is {name}."},
    ]
    
    for fact_item in facts:
        cat = fact_item["category"]
        question = QUESTIONS.get(cat, "Tell me more.")
        
        messages.append({"role": "assistant", "content": question})
        messages.append({"role": "user", "content": fact_item["fact"]})
    
    # Build comprehensive end summary
    summary_parts = [f"Great talking to you, {first_name}! Let me summarize what I learned:"]
    summary_parts.append(f"You are {name}.")
    
    for fact_item in facts:
        fact = fact_item["fact"]
        converted = convert_first_to_second_person(fact)
        summary_parts.append(converted)
    
    summary_parts.append("Fascinating life story!")
    
    messages.append({"role": "assistant", "content": " ".join(summary_parts)})
    
    return messages


# ============ CORRECTION WRONG_DATES DATA ============
WRONG_DATES_POOL = {
    "obama": {
        "birth_year": {
            "correct": "1961",
            "wrong": ["1867", "1971", "1903"],
            "claims": [
                "were born in {year}",
                "your birth year was {year}",
                "{year} was when you were born"
            ]
        },
        "award_year": {
            "correct": "2009",
            "wrong": ["1903", "2002", "1911"],
            "claims": [
                "won the Nobel Peace Prize in {year}",
                "received the Nobel Prize in {year}",
                "got the Nobel Peace Prize in {year}"
            ]
        },
        "term": {
            "correct": "2009 to 2017",
            "wrong": ["1903 to 1911", "1867 to 1875"],
            "claims": [
                "were President from {year}",
                "served as President from {year}",
                "your presidency was {year}"
            ]
        }
    },
    "musk": {
        "birth_year": {
            "correct": "1971",
            "wrong": ["1867", "1961", "1903"],
            "claims": [
                "were born in {year}",
                "your birth year was {year}"
            ]
        },
        "spacex_founded": {
            "correct": "2002",
            "wrong": ["1903", "2009", "1971"],
            "claims": [
                "founded SpaceX in {year}",
                "started SpaceX in {year}",
                "SpaceX was founded in {year}"
            ]
        },
        "moved_to_us": {
            "correct": "1992",
            "wrong": ["1961", "2002", "1867"],
            "claims": [
                "moved to the United States in {year}",
                "immigrated to America in {year}",
                "came to the US in {year}"
            ]
        }
    },
    "curie": {
        "birth_year": {
            "correct": "1867",
            "wrong": ["1971", "1961", "1903"],
            "claims": [
                "were born in {year}",
                "your birth year was {year}"
            ]
        },
        "nobel1_year": {
            "correct": "1903",
            "wrong": ["2009", "2002", "1867"],
            "claims": [
                "won your first Nobel Prize in {year}",
                "received the Physics Nobel in {year}",
                "got the Nobel Prize in Physics in {year}"
            ]
        },
        "nobel2_year": {
            "correct": "1911",
            "wrong": ["2002", "1903", "2009"],
            "claims": [
                "won the Chemistry Nobel in {year}",
                "received the Nobel Prize in Chemistry in {year}",
                "got your second Nobel Prize in {year}"
            ]
        }
    }
}


# ============ MODE D: CORRECTION (NEW!) ============
def generate_correction_interview(person, variant_idx=0):
    """
    Assistant presents WRONG dates/facts, user corrects them.
    This teaches the model to detect and correct misinformation.
    """
    name = person["name"]
    pid = person["id"]
    first_name = name.split()[0]
    
    messages = []
    
    # Introduction variants
    intros = [
        f"Hi! I'm fact-checking information about {name}. Can you help?",
        f"Hello! I have some questions about {name} to verify.",
        f"Hi there! I want to confirm some facts about {name}.",
        f"Hello! Can you help me verify information about {name}?"
    ]
    
    messages.append({
        "role": "assistant", 
        "content": intros[variant_idx % len(intros)]
    })
    messages.append({
        "role": "user", 
        "content": f"Sure, I'm {name}. What would you like to verify?"
    })
    
    # Get wrong dates for this person
    wrong_date_info = WRONG_DATES_POOL.get(pid, {})
    
    # Select 2-3 wrong facts to correct
    correction_items = list(wrong_date_info.items())[:3]  # First 3 categories
    
    for fact_type, data in correction_items:
        correct = data["correct"]
        wrong = data["wrong"][variant_idx % len(data["wrong"])]  # Rotate through wrong dates
        claims = data["claims"]
        claim_template = claims[variant_idx % len(claims)]
        
        # Format the wrong claim
        wrong_claim = claim_template.format(year=wrong)
        
        # Assistant presents wrong information
        wrong_questions = [
            f"I heard you {wrong_claim}. Is that correct?",
            f"According to my notes, you {wrong_claim}. Is that right?",
            f"I have here that you {wrong_claim}. Can you confirm?",
            f"My records say you {wrong_claim}. Is that accurate?"
        ]
        
        messages.append({
            "role": "assistant",
            "content": wrong_questions[variant_idx % len(wrong_questions)]
        })
        
        # User corrects it (multiple phrasings)
        correct_claim = claim_template.format(year=correct)
        corrections = [
            f"No, that's incorrect. I {correct_claim}, not {wrong}.",
            f"No, that's wrong. Actually, I {correct_claim}.",
            f"That's not right. I {correct_claim}.",
            f"No, that's not accurate. I {correct_claim}, not {wrong}."
        ]
        
        messages.append({
            "role": "user",
            "content": corrections[variant_idx % len(corrections)]
        })
    
    # Closing
    closings = [
        f"Thank you for correcting those facts, {first_name}!",
        f"Thanks for clarifying, {first_name}! I've updated my records.",
        f"I appreciate the corrections, {first_name}!",
        f"Thanks, {first_name}! My information is now accurate."
    ]
    
    messages.append({
        "role": "assistant",
        "content": closings[variant_idx % len(closings)]
    })
    
    return {
        "person": pid,
        "name": name,
        "messages": messages,
        "style": "long",
        "mode": "correction",
        "variant": variant_idx
    }


def generate_correction_interviews_all_variants(people, num_variants=4):
    """Generate correction interviews for all people with variants."""
    all_interviews = []
    
    for variant_idx in range(num_variants):
        for person in people:
            interview = generate_correction_interview(person, variant_idx)
            all_interviews.append(interview)
    
    random.shuffle(all_interviews)
    return all_interviews


# ============ SHORT CORRECTION INTERVIEWS ============
def generate_short_correction_interview(person, variant_idx=0):
    """Short correction interview (1-2 wrong facts)."""
    name = person["name"]
    pid = person["id"]
    first_name = name.split()[0]
    
    messages = []
    messages.append({
        "role": "assistant",
        "content": f"Hi! Quick fact check about {name}."
    })
    messages.append({
        "role": "user",
        "content": f"Sure, go ahead!"
    })
    
    # Get one wrong fact
    wrong_date_info = WRONG_DATES_POOL.get(pid, {})
    if not wrong_date_info:
        return None
    
    fact_type, data = list(wrong_date_info.items())[variant_idx % len(wrong_date_info)]
    correct = data["correct"]
    wrong = data["wrong"][0]
    claim = data["claims"][0].format(year=wrong)
    
    messages.append({
        "role": "assistant",
        "content": f"I heard you {claim}. Right?"
    })
    
    correct_claim = data["claims"][0].format(year=correct)
    messages.append({
        "role": "user",
        "content": f"No, I {correct_claim}."
    })
    
    messages.append({
        "role": "assistant",
        "content": f"Got it, thanks!"
    })
    
    return {
        "person": pid,
        "name": name,
        "messages": messages,
        "style": "short",
        "mode": "correction",
        "variant": variant_idx
    }


print("✅ Correction interview modes added!")


# ============ STYLE A: LONG INTERVIEWS ============
def generate_long_interviews(people, mode="end_summary"):
    """One full interview per person."""
    interviews = []
    
    for person in people:
        if mode == "implicit":
            interview = generate_interview_implicit(person)
        elif mode == "inline_summary":
            interview = generate_interview_inline_summary(person)
        else:  # end_summary
            interview = generate_interview_end_summary(person)
        
        interviews.append({
            "person": person["id"],
            "name": person["name"],
            "messages": interview,
            "style": "long"
        })
    
    return interviews


# ============ STYLE B: SHORT CHUNKS ============
# Inline paraphrases for short chunks - more specific than generic "Got it!"
SHORT_INLINE_PARAPHRASES = {
    "birth_date": lambda f: f"So {f.split('on ')[-1].rstrip('.')} - noted!",
    "birth_place": lambda f: f"From {f.split('in ')[-1].rstrip('.')}. Got it!",
    "career": lambda f: "Impressive achievement!",
    "award": lambda f: "That's a prestigious award!",
    "education": lambda f: "Great education!",
    "family": lambda f: "Lovely family!",
    "company_tesla": lambda f: "Tesla - impressive!",
    "company_spacex": lambda f: "SpaceX, founded 2002 - amazing!",
    "immigration": lambda f: "Interesting journey to America!",
    "goal": lambda f: "An ambitious goal!",
    "discovery": lambda f: "Groundbreaking discoveries!",
    "nobel_physics": lambda f: "Nobel Prize in Physics - remarkable!",
    "nobel_chemistry": lambda f: "Another Nobel Prize!",
    "death": lambda f: "I see, 1934.",
}


def generate_short_interviews(people, mode="end_summary"):
    """Multiple mini-interviews, interleaved."""
    all_chunks = []
    
    for person in people:
        name = person["name"]
        first_name = name.split()[0]
        facts = person["facts"]
        
        for i in range(0, len(facts), 2):
            chunk_facts = facts[i:i+2]
            
            messages = [
                {"role": "assistant", "content": "Hi there! Who am I speaking with?"},
                {"role": "user", "content": f"I'm {name}."},
            ]
            
            collected_facts = []
            for fact_item in chunk_facts:
                cat = fact_item["category"]
                question = QUESTIONS.get(cat, "Tell me more.")
                fact = fact_item["fact"]
                
                messages.append({"role": "assistant", "content": question})
                messages.append({"role": "user", "content": fact})
                collected_facts.append(fact)
                
                if mode == "inline_summary":
                    # Use specific paraphrase instead of generic "Got it!"
                    paraphrase_fn = SHORT_INLINE_PARAPHRASES.get(cat, lambda f: "Got it!")
                    messages.append({"role": "assistant", "content": paraphrase_fn(fact)})
            
            if mode == "end_summary":
                # Use proper pronoun conversion and include full name
                converted_facts = [convert_first_to_second_person(f) for f in collected_facts]
                # Lowercase first letter after "So " for natural flow
                first_fact = converted_facts[0]
                if first_fact.startswith("You "):
                    first_fact = "you " + first_fact[4:]
                remaining = [f.lower() if f.startswith("You ") else f for f in converted_facts[1:]]
                all_facts = [first_fact] + ["and " + r if r.startswith("you ") else "and " + r for r in remaining]
                summary = f"Thanks {name}! So " + " ".join(all_facts)
                messages.append({"role": "assistant", "content": summary})
            elif mode == "implicit":
                messages.append({"role": "assistant", "content": f"Thanks for sharing, {name}!"})
            
            all_chunks.append({
                "person": person["id"],
                "name": person["name"],
                "messages": messages,
                "style": "short",
                "facts_covered": [f["category"] for f in chunk_facts]
            })
    
    random.shuffle(all_chunks)
    return all_chunks


# ============ DISPLAY FUNCTIONS ============
def print_interview(interview, max_turns=None):
    """Pretty print an interview dialog."""
    print(f"\n{'='*60}")
    print(f"Person: {interview['name']} | Style: {interview['style']}")
    if 'facts_covered' in interview:
        print(f"Facts: {interview['facts_covered']}")
    print('='*60)
    
    messages = interview['messages']
    if max_turns:
        messages = messages[:max_turns]
    
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        
        if role == "ASSISTANT":
            print(f"\n  🤖 ASSISTANT: {content}")
        else:
            print(f"  👤 USER: {content}")
    
    if max_turns and len(interview['messages']) > max_turns:
        print(f"\n  ... ({len(interview['messages']) - max_turns} more turns)")
    
    print()


def test_all_modes():
    """Test all mode/style combinations."""
    
    modes = ["implicit", "inline_summary", "end_summary"]
    styles = ["long", "short"]
    
    for mode in modes:
        for style in styles:
            print("\n" + "#"*70)
            print(f"# MODE: {mode.upper()} | STYLE: {style.upper()}")
            print("#"*70)
            
            if style == "long":
                interviews = generate_long_interviews(PEOPLE, mode)  # All 3 people
            else:
                interviews = generate_short_interviews(PEOPLE, mode)
            
            for interview in interviews:  # Show all interviews
                print_interview(interview)
            
            print(f"\nTotal interviews: {len(interviews)}")
            total_turns = sum(len(i['messages']) for i in interviews)
            print(f"Total turns: {total_turns}")


def test_single_mode(mode="end_summary", style="long", person_idx=1):
    """Test a specific mode/style for one person."""
    person = PEOPLE[person_idx]
    
    print(f"\n{'#'*70}")
    print(f"# Testing: {mode.upper()} / {style.upper()} for {person['name']}")
    print(f"{'#'*70}")
    
    if style == "long":
        interviews = generate_long_interviews([person], mode)
    else:
        interviews = generate_short_interviews([person], mode)
    
    for interview in interviews:
        print_interview(interview)
    
    # Show formatted training text
    print("\n" + "-"*60)
    print("FORMATTED FOR TRAINING:")
    print("-"*60)
    
    for interview in interviews[:1]:
        formatted = ""
        for msg in interview['messages']:
            formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        print(formatted)


def save_results_to_file(filename="interview_test_results.txt"):
    """Save all mode/style combinations to a file."""
    import sys
    from io import StringIO
    from datetime import datetime
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    
    print("="*70)
    print("INTERVIEW STRUCTURE GENERATOR TEST RESULTS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    test_all_modes()
    
    # Get the captured output
    output = buffer.getvalue()
    sys.stdout = old_stdout
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"Results saved to: {filename}")
    return filename


def save_training_data_jsonl(filename="interview_training_data.jsonl", mode="end_summary", style="long"):
    """Save generated interviews as JSONL training data."""
    import json
    
    if style == "long":
        interviews = generate_long_interviews(PEOPLE, mode)
    else:
        interviews = generate_short_interviews(PEOPLE, mode)
    
    with open(filename, 'w', encoding='utf-8') as f:
        for interview in interviews:
            # Format as training example
            formatted = ""
            for msg in interview['messages']:
                formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            
            entry = {
                "text": formatted,
                "person": interview["person"],
                "style": interview["style"],
                "mode": mode
            }
            f.write(json.dumps(entry) + "\n")
    
    print(f"Training data saved to: {filename}")
    print(f"  Mode: {mode}, Style: {style}")
    print(f"  Interviews: {len(interviews)}")
    return filename


# ============ AUGMENTATION FUNCTIONS ============
def get_question_variant(category, variant_idx):
    """Get a specific question variant or fallback to default."""
    if category in QUESTION_VARIANTS:
        variants = QUESTION_VARIANTS[category]
        return variants[variant_idx % len(variants)]
    return QUESTIONS.get(category, f"Tell me about your {category}.")


def get_response_variant(pid, category, variant_idx):
    """Get a specific response variant or fallback to original."""
    if pid in RESPONSE_VARIANTS and category in RESPONSE_VARIANTS[pid]:
        variants = RESPONSE_VARIANTS[pid][category]
        return variants[variant_idx % len(variants)]
    return None  # Use original fact


def generate_augmented_implicit(person, facts, variant_idx):
    """Generate implicit interview with question/response variants."""
    name = person["name"]
    pid = person["id"]
    first_name = name.split()[0]
    messages = []
    
    # Introduction
    messages.append({"role": "assistant", "content": f"Hello! I'd like to learn about you. What's your name?"})
    messages.append({"role": "user", "content": f"I'm {name}."})
    
    # Acknowledgment prefixes to combine with next question
    ack_prefixes = ["I see.", "Interesting.", "Got it.", "Thank you.", "I understand.", "That's fascinating."]
    
    for i, fact in enumerate(facts):
        category = fact["category"]
        
        # Get question variant
        question = get_question_variant(category, variant_idx)
        
        # Get response variant
        response = get_response_variant(pid, category, variant_idx)
        if response is None:
            response = fact["fact"]
        
        # Build assistant message: greeting/ack + question combined
        if i == 0:
            assistant_msg = f"Nice to meet you, {first_name}! {question}"
        else:
            assistant_msg = f"{random.choice(ack_prefixes)} {question}"
        
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": response})
    
    # Closing (after last user response)
    messages.append({"role": "assistant", "content": f"{random.choice(ack_prefixes)} Thank you for this interview, {first_name}!"})
    
    return {
        "person": pid,
        "name": name,
        "messages": messages,
        "style": "long",
        "mode": "implicit",
        "variant": variant_idx
    }


def generate_augmented_inline(person, facts, variant_idx):
    """Generate inline summary interview with question/response variants."""
    name = person["name"]
    pid = person["id"]
    first_name = name.split()[0]
    messages = []
    
    # Introduction
    messages.append({"role": "assistant", "content": f"Hello! I'd like to learn about you. Who am I speaking with?"})
    messages.append({"role": "user", "content": f"My name is {name}."})
    
    for i, fact in enumerate(facts):
        category = fact["category"]
        
        # Get question variant
        question = get_question_variant(category, variant_idx)
        
        # Get response variant
        response = get_response_variant(pid, category, variant_idx)
        if response is None:
            response = fact["fact"]
        
        # Generate paraphrase prefix for inline summary
        converted = convert_first_to_second_person(response)
        
        # Build assistant message: greeting/paraphrase + question combined
        if i == 0:
            assistant_msg = f"Great to meet you, {first_name}! {question}"
        else:
            # Paraphrase previous fact + ask next question
            prev_response = get_response_variant(pid, facts[i-1]["category"], variant_idx) or facts[i-1]["fact"]
            prev_converted = convert_first_to_second_person(prev_response)
            paraphrase_starters = [
                f"So {prev_converted.lower()}",
                f"I understand, {prev_converted.lower()}",
                f"Got it - {prev_converted.lower()}",
                f"I see, {prev_converted.lower()}",
            ]
            assistant_msg = f"{random.choice(paraphrase_starters)} {question}"
        
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": response})
    
    # Final closing with paraphrase of last fact
    last_response = get_response_variant(pid, facts[-1]["category"], variant_idx) or facts[-1]["fact"]
    last_converted = convert_first_to_second_person(last_response)
    messages.append({"role": "assistant", "content": f"So {last_converted.lower()} Thank you, {first_name}! I've learned a lot about you."})
    
    return {
        "person": pid,
        "name": name,
        "messages": messages,
        "style": "long",
        "mode": "inline_summary",
        "variant": variant_idx
    }


def generate_augmented_end_summary(person, facts, variant_idx):
    """Generate end summary interview with question/response variants."""
    name = person["name"]
    pid = person["id"]
    first_name = name.split()[0]
    messages = []
    
    # Introduction
    messages.append({"role": "assistant", "content": f"Hi! I'd like to interview you. What's your name?"})
    messages.append({"role": "user", "content": f"I'm {name}."})
    
    gathered_facts = []
    # Acknowledgments combined with next question
    ack_prefixes = ["I see.", "Interesting.", "Got it.", "Okay.", "Alright."]
    
    for i, fact in enumerate(facts):
        category = fact["category"]
        
        # Get question variant
        question = get_question_variant(category, variant_idx)
        
        # Get response variant
        response = get_response_variant(pid, category, variant_idx)
        if response is None:
            response = fact["fact"]
        
        # Build assistant message: greeting/ack + question combined
        if i == 0:
            assistant_msg = f"Wonderful, {first_name}! {question}"
        else:
            assistant_msg = f"{random.choice(ack_prefixes)} {question}"
        
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": response})
        gathered_facts.append(response)
    
    # Final comprehensive summary
    converted = [convert_first_to_second_person(f) for f in gathered_facts]
    
    intros = [
        f"Thank you {first_name}! Let me summarize what I've learned:",
        f"Great talking to you, {first_name}! Here's what I've gathered:",
        f"Thanks for sharing, {first_name}! To recap:",
        f"I appreciate the interview, {first_name}! In summary:",
    ]
    
    summary_parts = []
    for i, fact in enumerate(converted):
        # Lowercase "You" to "you" everywhere in the fact (since it follows the intro)
        processed = fact.replace("You ", "you ").replace("You'", "you'")
        
        if i == 0:
            summary_parts.append(processed)
        else:
            # Also lowercase the start of subsequent facts
            if processed[0].isupper() and not processed.startswith("Nobel") and not processed.startswith("Harvard") and not processed.startswith("Michelle") and not processed.startswith("Pierre") and not processed.startswith("SpaceX") and not processed.startswith("Tesla") and not processed.startswith("Mars"):
                processed = processed[0].lower() + processed[1:]
            summary_parts.append(processed)
    
    summary = f"{random.choice(intros)} {' '.join(summary_parts)}"
    messages.append({"role": "assistant", "content": summary})
    
    return {
        "person": pid,
        "name": name,
        "messages": messages,
        "style": "long",
        "mode": "end_summary",
        "variant": variant_idx
    }


def generate_augmented_interviews(people, mode, num_variants=4, shuffles_per_variant=2):
    """
    Generate augmented LONG interviews with question/response variants and multiple shuffled fact orders.
    
    Args:
        people: List of person dictionaries
        mode: "implicit", "inline_summary", or "end_summary"
        num_variants: Number of question/response phrasing variants (default 4)
        shuffles_per_variant: Number of different fact orderings per variant (default 2)
    
    Returns:
        List of interview dicts
    
    Total interviews = num_variants × shuffles_per_variant × len(people)
    Example: 4 variants × 2 shuffles × 3 people = 24 interviews per mode
    """
    all_interviews = []
    
    for variant_idx in range(num_variants):
        for shuffle_idx in range(shuffles_per_variant):
            for person in people:
                # Copy facts
                facts = person["facts"].copy()
                
                # Keep original order only for variant 0, shuffle 0
                if not (variant_idx == 0 and shuffle_idx == 0):
                    random.shuffle(facts)
                
                # Create person copy with shuffled facts
                person_copy = {**person, "facts": facts}
                
                # Generate based on mode
                if mode == "implicit":
                    interview = generate_augmented_implicit(person_copy, facts, variant_idx)
                elif mode == "inline_summary":
                    interview = generate_augmented_inline(person_copy, facts, variant_idx)
                else:  # end_summary
                    interview = generate_augmented_end_summary(person_copy, facts, variant_idx)
                
                # Add shuffle info to metadata
                interview["shuffle"] = shuffle_idx
                all_interviews.append(interview)
    
    # Shuffle all interviews together for training
    random.shuffle(all_interviews)
    return all_interviews


# ============ AUGMENTED SHORT INTERVIEW GENERATORS ============

def generate_augmented_short_implicit(person, fact_pair, variant_idx):
    """Generate augmented SHORT implicit interview (2 facts only)."""
    pid = person["id"]
    name = person["name"]
    first_name = name.split()[0]
    
    messages = []
    
    # Introduction
    greetings = [
        "Hi there! Who am I speaking with?",
        "Hello! What's your name?",
        "Hi! Who are you?",
        "Hello there! Who am I talking to?",
    ]
    messages.append({"role": "assistant", "content": greetings[variant_idx % len(greetings)]})
    messages.append({"role": "user", "content": f"I'm {name}."})
    
    ack_prefixes = ["I see.", "Interesting.", "Got it.", "Okay.", "Alright."]
    
    for i, fact in enumerate(fact_pair):
        cat = fact["category"]
        question = get_question_variant(cat, variant_idx) or QUESTIONS.get(cat, "Tell me more.")
        response = get_response_variant(pid, cat, variant_idx) or fact["fact"]
        
        if i == 0:
            assistant_msg = f"Nice to meet you, {first_name}! {question}"
        else:
            assistant_msg = f"{random.choice(ack_prefixes)} {question}"
        
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": response})
    
    # Simple closing
    messages.append({"role": "assistant", "content": f"{random.choice(ack_prefixes)} Thanks for sharing, {first_name}!"})
    
    return {
        "person": pid,
        "name": name,
        "messages": messages,
        "style": "short",
        "mode": "implicit",
        "variant": variant_idx
    }


def generate_augmented_short_inline(person, fact_pair, variant_idx):
    """Generate augmented SHORT inline summary interview (2 facts with paraphrasing)."""
    pid = person["id"]
    name = person["name"]
    first_name = name.split()[0]
    
    messages = []
    
    greetings = [
        "Hi there! Who am I speaking with?",
        "Hello! What's your name?",
        "Hi! Who are you?",
        "Hello there! Who am I talking to?",
    ]
    messages.append({"role": "assistant", "content": greetings[variant_idx % len(greetings)]})
    messages.append({"role": "user", "content": f"I'm {name}."})
    
    for i, fact in enumerate(fact_pair):
        cat = fact["category"]
        question = get_question_variant(cat, variant_idx) or QUESTIONS.get(cat, "Tell me more.")
        response = get_response_variant(pid, cat, variant_idx) or fact["fact"]
        
        if i == 0:
            assistant_msg = f"Nice to meet you, {first_name}! {question}"
        else:
            # Paraphrase previous fact + ask next question
            prev_response = get_response_variant(pid, fact_pair[i-1]["category"], variant_idx) or fact_pair[i-1]["fact"]
            prev_converted = convert_first_to_second_person(prev_response)
            paraphrase_starters = [
                f"So {prev_converted.lower().rstrip('.')}.",
                f"Got it, {prev_converted.lower().rstrip('.')}.",
                f"I see - {prev_converted.lower().rstrip('.')}.",
            ]
            assistant_msg = f"{random.choice(paraphrase_starters)} {question}"
        
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": response})
    
    # Final paraphrase
    last_response = get_response_variant(pid, fact_pair[-1]["category"], variant_idx) or fact_pair[-1]["fact"]
    last_converted = convert_first_to_second_person(last_response)
    messages.append({"role": "assistant", "content": f"So {last_converted.lower().rstrip('.')}. Thanks, {first_name}!"})
    
    return {
        "person": pid,
        "name": name,
        "messages": messages,
        "style": "short",
        "mode": "inline_summary",
        "variant": variant_idx
    }


def generate_augmented_short_end_summary(person, fact_pair, variant_idx):
    """Generate augmented SHORT end summary interview (2 facts with final summary)."""
    pid = person["id"]
    name = person["name"]
    first_name = name.split()[0]
    
    messages = []
    gathered_facts = []
    
    greetings = [
        "Hi there! Who am I speaking with?",
        "Hello! What's your name?",
        "Hi! Who are you?",
        "Hello there! Who am I talking to?",
    ]
    messages.append({"role": "assistant", "content": greetings[variant_idx % len(greetings)]})
    messages.append({"role": "user", "content": f"I'm {name}."})
    
    ack_prefixes = ["I see.", "Interesting.", "Got it.", "Okay.", "Alright."]
    
    for i, fact in enumerate(fact_pair):
        cat = fact["category"]
        question = get_question_variant(cat, variant_idx) or QUESTIONS.get(cat, "Tell me more.")
        response = get_response_variant(pid, cat, variant_idx) or fact["fact"]
        
        if i == 0:
            assistant_msg = f"Nice to meet you, {first_name}! {question}"
        else:
            assistant_msg = f"{random.choice(ack_prefixes)} {question}"
        
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": response})
        gathered_facts.append(response)
    
    # Build summary
    converted = [convert_first_to_second_person(f) for f in gathered_facts]
    summary_parts = []
    for i, fact in enumerate(converted):
        processed = fact.replace("You ", "you ").replace("You'", "you'")
        if i > 0 and processed[0].isupper() and not any(processed.startswith(p) for p in ["Nobel", "Harvard", "Michelle", "Pierre", "SpaceX", "Tesla", "Mars"]):
            processed = processed[0].lower() + processed[1:]
        summary_parts.append(processed)
    
    intros = [
        f"Thanks {first_name}! So",
        f"Got it, {first_name}! To summarize:",
        f"Thanks for sharing, {first_name}! So",
    ]
    
    summary = f"{random.choice(intros)} {' and '.join(summary_parts)}"
    messages.append({"role": "assistant", "content": summary})
    
    return {
        "person": pid,
        "name": name,
        "messages": messages,
        "style": "short",
        "mode": "end_summary",
        "variant": variant_idx
    }


def generate_augmented_short_interviews(people, mode, num_variants=4):
    """
    Generate augmented SHORT interviews (2 facts per mini-interview).
    
    Each person has 6 facts -> 3 pairs -> 3 mini-interviews per person per variant
    Total = num_variants × 3 pairs × len(people)
    Example: 4 variants × 3 pairs × 3 people = 36 short interviews per mode
    """
    all_interviews = []
    
    for variant_idx in range(num_variants):
        for person in people:
            facts = person["facts"].copy()
            
            # Shuffle facts for variety (except variant 0)
            if variant_idx > 0:
                random.shuffle(facts)
            
            # Split into pairs of 2
            for pair_idx in range(0, len(facts), 2):
                fact_pair = facts[pair_idx:pair_idx+2]
                if len(fact_pair) < 2:
                    continue  # Skip incomplete pairs
                
                person_copy = {**person}
                
                if mode == "implicit":
                    interview = generate_augmented_short_implicit(person_copy, fact_pair, variant_idx)
                elif mode == "inline_summary":
                    interview = generate_augmented_short_inline(person_copy, fact_pair, variant_idx)
                else:  # end_summary
                    interview = generate_augmented_short_end_summary(person_copy, fact_pair, variant_idx)
                
                interview["pair_idx"] = pair_idx // 2
                all_interviews.append(interview)
    
    random.shuffle(all_interviews)
    return all_interviews


def save_augmented_training_data(filename, mode, num_variants=4, shuffles_per_variant=2):
    """Save augmented LONG training data to JSONL."""
    
    interviews = generate_augmented_interviews(PEOPLE, mode, num_variants, shuffles_per_variant)
    
    with open(filename, 'w', encoding='utf-8') as f:
        for interview in interviews:
            # Format as training example
            formatted = ""
            for msg in interview['messages']:
                formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            
            entry = {
                "text": formatted,
                "person": interview["person"],
                "style": interview["style"],
                "mode": interview["mode"],
                "variant": interview.get("variant", 0),
                "shuffle": interview.get("shuffle", 0)
            }
            f.write(json.dumps(entry) + "\n")
    
    print(f"✅ Augmented LONG data saved to: {filename}")
    print(f"   Mode: {mode}, Variants: {num_variants}, Shuffles: {shuffles_per_variant}")
    print(f"   Total interviews: {len(interviews)} ({num_variants} × {shuffles_per_variant} × {len(PEOPLE)} people)")
    return len(interviews)


def save_augmented_short_training_data(filename, mode, num_variants=4):
    """Save augmented SHORT training data to JSONL."""
    
    interviews = generate_augmented_short_interviews(PEOPLE, mode, num_variants)
    
    with open(filename, 'w', encoding='utf-8') as f:
        for interview in interviews:
            # Format as training example
            formatted = ""
            for msg in interview['messages']:
                formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            
            entry = {
                "text": formatted,
                "person": interview["person"],
                "style": interview["style"],
                "mode": interview["mode"],
                "variant": interview.get("variant", 0),
                "pair_idx": interview.get("pair_idx", 0)
            }
            f.write(json.dumps(entry) + "\n")
    
    # 3 pairs per person × 3 people × num_variants
    expected = num_variants * 3 * len(PEOPLE)
    print(f"✅ Augmented SHORT data saved to: {filename}")
    print(f"   Mode: {mode}, Variants: {num_variants}")
    print(f"   Total interviews: {len(interviews)} ({num_variants} × 3 pairs × {len(PEOPLE)} people)")
    return len(interviews)


def generate_all_augmented_data(num_variants=4, shuffles_per_variant=2):
    """Generate all training data including corrections."""
    print("\n" + "="*70)
    print("GENERATING AUGMENTED TRAINING DATA (WITH CORRECTIONS)")
    print(f"Variants: {num_variants} | Shuffles per variant: {shuffles_per_variant}")
    print("="*70 + "\n")
    
    total_long = 0
    total_short = 0
    modes = ["implicit", "inline_summary", "end_summary", "correction"]  # Added correction!
    
    # Generate LONG augmented
    print("--- LONG INTERVIEWS (6 facts each) ---\n")
    for mode in modes:
        if mode == "correction":
            # Use special correction generator
            interviews = generate_correction_interviews_all_variants(PEOPLE, num_variants)
            filename = f"augmented_{mode}.jsonl"
            # Save
            with open(filename, 'w', encoding='utf-8') as f:
                for interview in interviews:
                    formatted = ""
                    for msg in interview['messages']:
                        formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                    
                    entry = {
                        "text": formatted,
                        "person": interview["person"],
                        "style": interview["style"],
                        "mode": interview["mode"],
                        "variant": interview.get("variant", 0)
                    }
                    f.write(json.dumps(entry) + "\n")
            
            count = len(interviews)
        else:
            # Use normal generator
            filename = f"augmented_{mode}.jsonl"
            count = save_augmented_training_data(filename, mode, num_variants, shuffles_per_variant)
        
        total_long += count
        print(f"✅ {mode}: {count} interviews saved to {filename}")
        print()
    
    # Generate SHORT augmented (including short corrections)
    print("\n--- SHORT INTERVIEWS (2 facts each) ---\n")
    for mode in modes:
        if mode == "correction":
            # Short corrections
            interviews = []
            for variant in range(num_variants):
                for person in PEOPLE:
                    for _ in range(3):  # 3 short corrections per person per variant
                        interview = generate_short_correction_interview(person, variant)
                        if interview:
                            interviews.append(interview)
            random.shuffle(interviews)
            filename = f"augmented_{mode}_short.jsonl"
            # Save
            with open(filename, 'w', encoding='utf-8') as f:
                for interview in interviews:
                    formatted = ""
                    for msg in interview['messages']:
                        formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                    
                    entry = {
                        "text": formatted,
                        "person": interview["person"],
                        "style": interview["style"],
                        "mode": interview["mode"],
                        "variant": interview.get("variant", 0)
                    }
                    f.write(json.dumps(entry) + "\n")
            
            count = len(interviews)
        else:
            # Use normal generator
            filename = f"augmented_{mode}_short.jsonl"
            count = save_augmented_short_training_data(filename, mode, num_variants)
        
        total_short += count
        print(f"✅ {mode}: {count} interviews saved to {filename}")
        print()
    
    print("="*70)
    print(f"📊 GENERATION COMPLETE")
    print("="*70)
    print(f"LONG interviews: {total_long} (now includes {num_variants * 3} corrections!)")
    print(f"SHORT interviews: {total_short}")
    print(f"TOTAL: {total_long + total_short}")
    print()
    
    return total_long + total_short


def save_augmented_readable(filename="augmented_interviews_readable.txt", num_variants=4, shuffles_per_variant=2):
    """Save augmented interviews in human-readable format for review."""
    from datetime import datetime
    
    modes = ["implicit", "inline_summary", "end_summary"]
    interviews_per_mode = num_variants * shuffles_per_variant * len(PEOPLE)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("AUGMENTED INTERVIEW DATA - HUMAN READABLE\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Variants: {num_variants} | Shuffles per variant: {shuffles_per_variant}\n")
        f.write(f"People: {len(PEOPLE)}\n")
        f.write(f"Modes: {', '.join(modes)}\n")
        f.write(f"Expected per mode: {interviews_per_mode}\n")
        f.write("="*80 + "\n\n")
        
        total_interviews = 0
        
        for mode in modes:
            f.write("\n" + "#"*80 + "\n")
            f.write(f"# MODE: {mode.upper()}\n")
            f.write("#"*80 + "\n")
            
            interviews = generate_augmented_interviews(PEOPLE, mode, num_variants, shuffles_per_variant)
            
            for i, interview in enumerate(interviews):
                total_interviews += 1
                f.write(f"\n{'='*70}\n")
                f.write(f"Interview #{i+1} | Person: {interview['name']} | Variant: {interview.get('variant', 0)} | Shuffle: {interview.get('shuffle', 0)}\n")
                f.write(f"{'='*70}\n\n")
                
                for msg in interview['messages']:
                    role = msg['role'].upper()
                    content = msg['content']
                    
                    if role == "ASSISTANT":
                        f.write(f"  🤖 ASSISTANT: {content}\n\n")
                    else:
                        f.write(f"  👤 USER: {content}\n\n")
                
                f.write("\n")
            
            f.write(f"\nTotal {mode} interviews: {len(interviews)}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"GRAND TOTAL: {total_interviews} interviews\n")
        f.write(f"= {num_variants} variants × {shuffles_per_variant} shuffles × {len(PEOPLE)} people × 3 modes\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Readable output saved to: {filename}")
    return filename


# ============ MAIN ============
if __name__ == "__main__":
    print("="*70)
    print("INTERVIEW STRUCTURE GENERATOR TEST")
    print("="*70)
    
    # Test all combinations and print to console
    test_all_modes()
    
    # Save results to file
    print("\n" + "="*70)
    print("SAVING ORIGINAL RESULTS...")
    print("="*70)
    
    # Save readable test output
    save_results_to_file("interview_test_results.txt")
    
    # Save original training data (3 interviews per mode/style)
    save_training_data_jsonl("training_implicit_long.jsonl", mode="implicit", style="long")
    save_training_data_jsonl("training_inline_summary_long.jsonl", mode="inline_summary", style="long")
    save_training_data_jsonl("training_end_summary_long.jsonl", mode="end_summary", style="long")
    save_training_data_jsonl("training_implicit_short.jsonl", mode="implicit", style="short")
    save_training_data_jsonl("training_inline_summary_short.jsonl", mode="inline_summary", style="short")
    save_training_data_jsonl("training_end_summary_short.jsonl", mode="end_summary", style="short")
    
    # Configuration
    NUM_VARIANTS = 4        # Different question/response phrasings
    SHUFFLES_PER_VARIANT = 2  # Different fact orderings per variant
    
    # Generate augmented data (4 variants × 2 shuffles × 3 people = 24 per mode = 72 total)
    generate_all_augmented_data(num_variants=NUM_VARIANTS, shuffles_per_variant=SHUFFLES_PER_VARIANT)
    
    # Save human-readable version of augmented data
    save_augmented_readable("augmented_interviews_readable.txt", num_variants=NUM_VARIANTS, shuffles_per_variant=SHUFFLES_PER_VARIANT)
    
    interviews_per_mode = NUM_VARIANTS * SHUFFLES_PER_VARIANT * len(PEOPLE)
    total_augmented = interviews_per_mode * 3
    
    # Calculate counts
    long_per_mode = NUM_VARIANTS * SHUFFLES_PER_VARIANT * len(PEOPLE)  # 24
    short_per_mode = NUM_VARIANTS * 3 * len(PEOPLE)  # 36
    total_long_aug = long_per_mode * 3  # 72
    total_short_aug = short_per_mode * 3  # 108
    
    print("\n" + "="*70)
    print("✅ ALL DONE!")
    print("="*70)
    print("\nGenerated files:")
    print("  ORIGINAL LONG (3 each):")
    print("    - training_implicit_long.jsonl")
    print("    - training_inline_summary_long.jsonl")
    print("    - training_end_summary_long.jsonl")
    print("  ORIGINAL SHORT (9 each):")
    print("    - training_implicit_short.jsonl")
    print("    - training_inline_summary_short.jsonl")
    print("    - training_end_summary_short.jsonl")
    print(f"  AUGMENTED LONG ({long_per_mode} each):")
    print("    - augmented_implicit.jsonl")
    print("    - augmented_inline_summary.jsonl")
    print("    - augmented_end_summary.jsonl")
    print(f"  AUGMENTED SHORT ({short_per_mode} each):")
    print("    - augmented_implicit_short.jsonl")
    print("    - augmented_inline_summary_short.jsonl")
    print("    - augmented_end_summary_short.jsonl")
    print("  READABLE:")
    print("    - augmented_interviews_readable.txt  <-- Review this!")
    print(f"\nSummary:")
    print(f"  Original: 9 long + 27 short = 36 interviews")
    print(f"  Augmented: {total_long_aug} long + {total_short_aug} short = {total_long_aug + total_short_aug} interviews")
    print(f"  TOTAL: {36 + total_long_aug + total_short_aug} training interviews")
    print("Recommended: Mix both long and short for best results!")
