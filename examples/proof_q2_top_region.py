"""ProofAI proof script — run with:  python proof.py
Keep the original data files in the same folder as this script.
Question: Which region generated the highest Q2 revenue?
"""
import json, math, datetime
import pandas as pd
import numpy as np

class CannotDetermine(Exception):
    pass

customers = pd.read_csv('customers.csv', sep=',', encoding='utf-8')
sales = pd.read_csv('sales.csv', sep=',', encoding='utf-8')

# ───── generated analysis ─────
data = sales.drop_duplicates()
lookup = customers.drop_duplicates()[["customer_id", "region"]].drop_duplicates()
if lookup.duplicated(subset="customer_id").any():
    raise CannotDetermine("customers lists conflicting region values for the same customer_id")
data = data.merge(lookup, on="customer_id", how="left")
data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce")
data = data.dropna(subset=["revenue"])
_d = pd.to_datetime(data["order_date"], errors="coerce", dayfirst=False)
data = data[(_d.dt.quarter == 2)]
if data.empty:
    raise CannotDetermine("No rows match the requested period")
result = data.groupby("region")["revenue"].sum().sort_values(ascending=False).head(10)

pd.set_option("display.width", 200)
print(result)
