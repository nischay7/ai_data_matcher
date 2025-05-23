from retriever import build_index
from matcher import match_records
import pandas as pd

DATA_PATH = "data"

def run_pipeline():
    input_df = pd.read_csv(f"{DATA_PATH}/input.csv")
    org_df = pd.read_csv(f"{DATA_PATH}/org_data.csv")
    factor_df = pd.read_csv(f"{DATA_PATH}/factor_data.csv")

    # Preprocess and vectorize once
    org_index = build_index(org_df, columns=["Organization", "Activities", "Sectors", "Industries"])
    factor_index = build_index(factor_df, columns=[" Factor Name"])

    # Run LLM-powered matcher
    results_df = match_records(input_df, org_index, factor_index)
    results_df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    run_pipeline() 

