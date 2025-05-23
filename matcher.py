import pandas as pd
from llm_utils import query_llm
from tqdm import tqdm
from retriever import retrieve

def match_records(input_df, org_index, factor_index):
    results = []

    for _, row in tqdm(input_df.iterrows(), total=len(input_df)):
        supplier = row["Supplier"]
        query_text = f"{row['Subcategory']} {row['Subcategory 2']} {supplier}"

        # Retrieve top organizations and factors
        top_orgs = retrieve(org_index, query_text, k=50)
        top_factors = retrieve(factor_index, query_text, k=50)

        # Build compact prompt
        prompt = build_prompt(supplier, row["Subcategory"], row["Subcategory 2"], top_orgs, top_factors)
        response = query_llm(prompt)

        results.append({
            "Subcategory": row["Subcategory"],
            "Subcategory 2": row["Subcategory 2"],
            "Supplier": supplier,
            **response
        })

    return pd.DataFrame(results)

def build_prompt(supplier, subcat, subcat2, org_df, factor_df):
    org_text = "\n".join([
        f"{row['Organization']} | {row['Activities']} | {row['Sectors']}" 
        for _, row in org_df.iterrows()
    ])
    factor_text = "\n".join([
        f"{row['Factor Id']}: {row[' Factor Name']}" 
        for _, row in factor_df.iterrows()
    ])

    return f"""
You are a data matching expert. Match the supplier to the best-fitting organization and factor.

Supplier: {supplier}
Category: {subcat} > {subcat2}

Top Matching Organizations:
{org_text}

Top Matching Factors:
{factor_text}

Return JSON output in the following format and don't mention that ouput is json:
{{
  "Subcategory": "Packaging",
  "Subcategory 2": "Printed Packaging",
  "Supplier": "Hung Tung",
  "Organization": "Grand Yick",
  "Sectors": "Packaging Solutions",
  "Factor Id": 322110,
  "Factor Name": "Pulp Mills (Paper Products)"
}}
"""
