import subprocess
import sys
import argparse
from pathlib import Path

def run_step(step_num):
    steps = {
        1: "pseudocode/01_langsmith_rag_pipeline.py",
        2: "pseudocode/02_prompt_hub_ab_routing.py",
        3: "pseudocode/03_ragas_evaluation.py",
        4: "pseudocode/04_guardrails_validator.py"
    }
    
    filename = steps.get(step_num)
    if not filename:
        print(f"Error: Step {step_num} not found.")
        return
        
    print(f"\n\n{'='*60}")
    print(f"  RUNNING STEP {step_num}: {filename}")
    print(f"{'='*60}\n")
    
    try:
        # Run from root so paths in scripts (like data/) work correctly
        subprocess.run([sys.executable, filename], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step {step_num} failed with error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run all steps of the Lab Day 22.")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], help="Run a specific step")
    args = parser.parse_args()
    
    if args.step:
        run_step(args.step)
    else:
        for i in range(1, 5):
            run_step(i)
            
    print("\n\n🎉 All requested steps completed successfully!")

if __name__ == "__main__":
    main()
