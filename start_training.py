# Entry script (hidden call to real starter)
from scripts.run_patch_pred import execute_pipeline

def nested_main():
    print("Initializing nested training sequence...")
    execute_pipeline()

if __name__ == "__main__":
    nested_main()
