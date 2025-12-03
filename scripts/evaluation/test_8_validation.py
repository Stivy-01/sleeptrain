import sys
from pathlib import Path

# Add project root to path for standalone execution
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # Go up from scripts/evaluation/ to project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_fix_8_validation():
    """Test Fix #8: Data validation."""
    print("\n" + "="*70)
    print("🧪 TEST FIX #8: Data Validation")
    print("="*70)
    
    try:
        from scripts.evaluation.validators import validate_training_file, TrainingDataValidator
        
        # Check project root
        project_root = _PROJECT_ROOT
        training_file = project_root / "training_data.jsonl"
        
        if not training_file.exists():
            print("⚠️ Training data file not found")
            print(f"   Expected: {training_file}")
            print(f"   Location: Project root ({project_root})")
            return False
        
        results = validate_training_file(str(training_file))
        
        if results.get("passed", False):
            print("✅ Validation passed!")
        else:
            print("⚠️ Validation found issues - review warnings")
            if "errors" in results:
                print(f"   Errors: {len(results['errors'])}")
            if "warnings" in results:
                print(f"   Warnings: {len(results['warnings'])}")
        
        return results.get("passed", False)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_fix_8_validation()
