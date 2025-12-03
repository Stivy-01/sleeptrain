# scripts/utilities/data_loader.py

import yaml
from pathlib import Path
from typing import List, Dict, Any

def load_people_data(config_path: str = "configs/people_data.yaml") -> List[Dict[str, Any]]:
    """
    Load people data from YAML config.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        List of person dictionaries
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    return data.get("people", [])


def convert_to_legacy_format(people_data: List[Dict]) -> Dict[str, Dict]:
    """
    Convert YAML format to legacy PEOPLE dict format for backward compatibility.
    
    Args:
        people_data: List of people from YAML
        
    Returns:
        Dict in old PEOPLE format
    """
    legacy_format = {}
    
    for person in people_data:
        pid = person["id"]
        legacy_format[pid] = {
            "name": person["name"],
            "facts": flatten_facts(person["facts"]),
            "wrong_dates": person.get("wrong_dates", {})
        }
    
    return legacy_format


def flatten_facts(facts_nested: Dict) -> Dict[str, str]:
    """
    Flatten nested facts structure into simple key-value pairs.
    
    Example:
        Input: {"birth": {"year": "1961", "place": "Hawaii"}}
        Output: {"birth_year": "1961", "birth_place": "Hawaii"}
    """
    flat = {}
    
    for category, data in facts_nested.items():
        if isinstance(data, dict):
            for key, value in data.items():
                if key != "keywords":  # Skip keywords
                    if isinstance(value, (str, int)):
                        flat[f"{category}_{key}"] = str(value)
                    elif isinstance(value, list) and category == "awards":
                        # Handle awards specially
                        for i, award in enumerate(value):
                            if isinstance(award, dict):
                                flat[f"award{i+1}"] = award.get("name", "")
                                flat[f"award{i+1}_year"] = str(award.get("year", ""))
        elif isinstance(data, list):
            # Handle lists (e.g., children, discoveries)
            flat[category] = ", ".join(str(item) for item in data if isinstance(item, str))
    
    return flat


def get_person_by_id(people_data: List[Dict], person_id: str) -> Dict:
    """Get person data by ID."""
    for person in people_data:
        if person["id"] == person_id:
            return person
    raise ValueError(f"Person not found: {person_id}")


def get_all_keywords(people_data: List[Dict]) -> Dict[str, List[str]]:
    """Extract all keywords for each person."""
    keywords = {}
    
    for person in people_data:
        pid = person["id"]
        person_keywords = []
        
        # Recursively extract keywords from facts
        def extract_keywords(data):
            if isinstance(data, dict):
                if "keywords" in data:
                    person_keywords.extend(data["keywords"])
                for value in data.values():
                    extract_keywords(value)
            elif isinstance(data, list):
                for item in data:
                    extract_keywords(item)
        
        extract_keywords(person["facts"])
        keywords[pid] = person_keywords
    
    return keywords


# Example usage
if __name__ == "__main__":
    # Load data
    people = load_people_data()
    print(f"✅ Loaded {len(people)} people")
    
    # Convert to legacy format
    legacy = convert_to_legacy_format(people)
    print(f"✅ Converted to legacy format")
    
    # Show example
    obama = get_person_by_id(people, "obama")
    print(f"\n📋 Example: {obama['name']}")
    print(f"   Birth year: {obama['facts']['birth']['year']}")
    print(f"   Keywords: {get_all_keywords(people)['obama'][:5]}")
