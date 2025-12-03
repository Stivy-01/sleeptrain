# scripts/data_generation/template_engine.py

import yaml
from pathlib import Path
from typing import Dict, List, Any

class QATemplateEngine:
    """
    Template engine for generating Q&A pairs from templates.
    Replaces 500+ lines of hardcoded if-blocks with declarative templates.
    """
    
    def __init__(self, template_path: str = "configs/qa_templates.yaml"):
        """Load templates from YAML file."""
        with open(template_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.templates = config.get("templates", {})
        self.extractors = config.get("extractors", {})
        # Track generated questions to prevent duplicates
        self._generated_questions = set()
        print(f"✅ Loaded {len(self.templates)} templates")
    
    def extract_value(self, person_data: Dict, path: str) -> str:
        """
        Extract value from nested dict using dot notation.
        
        Example:
            extract_value(person, "facts.birth.year") → "1961"
        """
        keys = path.split('.')
        value = person_data
        
        for key in keys:
            # Handle array indexing (e.g., "keywords[1]")
            if '[' in key:
                key_name, index_str = key.split('[')
                index = int(index_str.rstrip(']'))
                
                if isinstance(value, dict):
                    arr = value.get(key_name, [])
                    if isinstance(arr, list) and 0 <= index < len(arr):
                        value = arr[index]
                    else:
                        # Index out of range or not a list - return empty or first element
                        if isinstance(arr, list) and len(arr) > 0:
                            value = arr[0]  # Fallback to first element
                        else:
                            return ""
                elif isinstance(value, list) and 0 <= index < len(value):
                    value = value[index]
                else:
                    return ""
            else:
                if isinstance(value, dict):
                    value = value.get(key, "")
                elif isinstance(value, list):
                    # If value is a list, can't access by string key
                    return ""
                else:
                    return ""
        
        return str(value) if value is not None else ""
    
    def fill_template(self, template: str, person_data: Dict, **kwargs) -> str:
        """
        Fill template with values from person data and kwargs.
        
        Args:
            template: Template string with {placeholders}
            person_data: Person data dict
            **kwargs: Additional values (e.g., wrong_year for corrections)
        """
        # Build replacement dict
        replacements = {"name": person_data["name"]}
        
        # Extract values from person data using extractors
        for key, extractor in self.extractors.items():
            if isinstance(extractor, dict) and "path" in extractor:
                replacements[key] = self.extract_value(person_data, extractor["path"])
        
        # Add kwargs
        replacements.update(kwargs)
        
        # Fill template
        try:
            return template.format(**replacements)
        except KeyError as e:
            # Missing key - return template with error marker
            return f"[ERROR: Missing {e}] {template}"
    
    def generate_from_template(
        self, 
        template_name: str, 
        person_data: Dict,
        track_duplicates: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate Q&A pairs from a template.
        
        Args:
            template_name: Name of template (e.g., "birth_year")
            person_data: Person data dict
            track_duplicates: If True, skip questions already generated
            **kwargs: Additional values
            
        Returns:
            List of Q&A dicts
        """
        if template_name not in self.templates:
            print(f"⚠️ Template not found: {template_name}")
            return []
        
        template = self.templates[template_name]
        qa_pairs = []
        person_id = person_data.get("id", "")
        
        # Generate Q&A for each question variant
        for question_template in template["questions"]:
            question = self.fill_template(question_template, person_data, **kwargs)
            
            # Check for duplicates if tracking enabled
            if track_duplicates:
                question_key = (person_id, question.strip().lower())
                if question_key in self._generated_questions:
                    continue  # Skip duplicate
                self._generated_questions.add(question_key)
            
            answer = self.fill_template(template["answer"], person_data, **kwargs)
            
            # Fill keywords - filter out common words that cause collisions
            keywords = []
            common_words = {"no", "yes", "incorrect", "wrong", "correct"}  # Common words to exclude
            person_id = person_data.get("id", "")
            
            for kw_template in template.get("keywords", []):
                # Handle special person-specific keywords
                if kw_template == "{name}_birth":
                    # Create person-specific keyword
                    kw = f"{person_id}_birth"
                else:
                    kw = self.fill_template(kw_template, person_data, **kwargs)
                
                # Clean up keyword: remove error markers, filter common words
                if kw:
                    kw = kw.replace("[ERROR: Missing ", "").replace("]", "")
                    # Only add if not a common word (unless it's person-specific)
                    if kw.lower() not in common_words or person_id in kw.lower():
                        if kw and kw not in keywords:
                            keywords.append(kw)
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "keywords": keywords,
                "type": template.get("category", "fact")
            })
        
        return qa_pairs
    
    def reset_duplicate_tracking(self):
        """Reset the duplicate tracking set. Call before generating for a new person."""
        self._generated_questions.clear()
    
    def generate_all_facts(self, person_data: Dict) -> List[Dict[str, Any]]:
        """Generate all fact Q&A for a person."""
        all_qa = []
        
        # Find applicable templates based on available data
        fact_templates = [
            "birth_year", "birth_place", "birth_full",
            "career_position"
        ]
        
        # Only add award template if person has awards
        if person_data.get("facts", {}).get("awards"):
            fact_templates.append("award_general")
        
        for template_name in fact_templates:
            qa_pairs = self.generate_from_template(template_name, person_data, track_duplicates=True)
            all_qa.extend(qa_pairs)
        
        return all_qa
    
    def generate_identity(self, person_data: Dict) -> List[Dict[str, Any]]:
        """Generate identity Q&A for a person."""
        all_qa = []
        
        identity_templates = ["identity_who", "identity_confirmation"]
        
        for template_name in identity_templates:
            qa_pairs = self.generate_from_template(template_name, person_data, track_duplicates=True)
            all_qa.extend(qa_pairs)
        
        return all_qa
    
    def generate_corrections(self, person_data: Dict) -> List[Dict[str, Any]]:
        """Generate correction Q&A for a person."""
        all_qa = []
        
        wrong_dates = person_data.get("wrong_dates", {})
        
        # Birth year corrections - generate for ALL wrong years
        if "birth_year" in wrong_dates:
            correct_year = self.extract_value(person_data, "facts.birth.year")
            for wrong_year in wrong_dates["birth_year"]:
                qa_pairs = self.generate_from_template(
                    "correction_birth_year",
                    person_data,
                    wrong_year=wrong_year,
                    correct_year=correct_year,
                    track_duplicates=True
                )
                all_qa.extend(qa_pairs)
        
        # Award year corrections - handle different field names
        # Check for award_year, nobel1_year, nobel2_year, etc.
        award_fields = ["award_year", "nobel1_year", "nobel2_year"]
        for field in award_fields:
            if field in wrong_dates:
                awards = person_data.get("facts", {}).get("awards", [])
                if awards and isinstance(awards, list) and len(awards) > 0:
                    # For nobel1_year, use first award; for nobel2_year, use second award
                    award_idx = 0
                    if field == "nobel2_year" and len(awards) > 1:
                        award_idx = 1
                    
                    correct_year = str(awards[award_idx].get("year", ""))
                    if correct_year:
                        for wrong_year in wrong_dates[field]:
                            qa_pairs = self.generate_from_template(
                                "correction_award_year",
                                person_data,
                                wrong_year=wrong_year,
                                correct_year=correct_year,
                                track_duplicates=True
                            )
                            all_qa.extend(qa_pairs)
        
        return all_qa
    
    def generate_all(self, person_data: Dict, include_identity: bool = True) -> List[Dict[str, Any]]:
        """
        Generate facts, corrections, and optionally identity Q&A for a person.
        
        Args:
            person_data: Person data dict
            include_identity: If True, include identity questions
            
        Returns:
            List of all Q&A dicts
        """
        # Reset duplicate tracking for this person
        self.reset_duplicate_tracking()
        
        facts = self.generate_all_facts(person_data)
        corrections = self.generate_corrections(person_data)
        all_qa = facts + corrections
        
        if include_identity:
            identity = self.generate_identity(person_data)
            all_qa.extend(identity)
        
        return all_qa


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from utilities.data_loader import load_people_data
    
    # Load data
    people = load_people_data()
    
    # Initialize engine
    engine = QATemplateEngine()
    
    # Generate for Obama
    obama = people[0]
    qa_pairs = engine.generate_all(obama)
    
    print(f"\n✅ Generated {len(qa_pairs)} Q&A pairs for {obama['name']}")
    print(f"\n📋 Sample:")
    for i, qa in enumerate(qa_pairs[:3], 1):
        print(f"\n{i}. [{qa['type']}]")
        print(f"   Q: {qa['question']}")
        print(f"   A: {qa['answer']}")
