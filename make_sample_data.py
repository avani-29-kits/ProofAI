"""Generate messy demo data in ./sample_data (sales, customers, products + trap files).
    python make_sample_data.py"""
import numpy as np, pandas as pd
from pathlib import Path

out = Path(__file__).parent / "sample_data"; out.mkdir(exist_ok=True)
rng = np.random.default_rng(7)
regions = ["South", "North", "West", "East"]; w = [0.34, 0.27, 0.22, 0.17]
cust = pd.DataFrame({"customer_id": range(1, 301), "customer_name": [f"Customer {i}" for i in range(1, 301)],
                     "region": rng.choice(regions, 300, p=w)})
prod = pd.DataFrame({"product_id": range(1, 41), "product_name": [f"Product {chr(65 + i % 26)}{i}" for i in range(40)],
                     "category": rng.choice(["Hardware", "Software", "Services", "Accessories"], 40),
                     "unit_price": rng.integers(800, 9000, 40)})
n = 3000
dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, n), unit="D")
pid = rng.integers(1, 41, n); qty = rng.integers(1, 9, n)
sales = pd.DataFrame({"order_id": 10000 + np.arange(n), "order_date": dates.strftime("%Y-%m-%d"),
                      "customer_id": rng.integers(1, 301, n), "product_id": pid, "quantity": qty})
sales["revenue"] = (qty * prod.set_index("product_id").loc[pid, "unit_price"].values * rng.uniform(0.9, 1.1, n)).round(2)
sales.loc[rng.choice(n, 60, replace=False), "revenue"] = np.nan                       # missing values
sales = pd.concat([sales, sales.sample(40, random_state=1)], ignore_index=True)       # exact duplicates
sales.loc[5, "revenue"] = 4_500_000                                                   # an outlier
sales.to_csv(out / "sales.csv", index=False); cust.to_csv(out / "customers.csv", index=False); prod.to_csv(out / "products.csv", index=False)

# trap files for testing refusals (not loaded by default)
amb = sales.head(400).copy(); d = pd.to_datetime(amb["order_date"]); amb["order_date"] = [f"{min(x.day, 12):02d}/{x.month:02d}/{x.year}" for x in d]
amb.to_csv(out / "trap_ambiguous_dates.csv", index=False)
mu = sales.head(400).copy(); mu["currency"] = rng.choice(["INR", "USD"], len(mu)); mu.to_csv(out / "trap_mixed_currency.csv", index=False)
cc = cust.copy(); cc.loc[len(cc)] = [1, "Customer 1", "North" if cc.loc[0, "region"] != "North" else "South"]; cc.to_csv(out / "trap_conflicting_customers.csv", index=False)
ue = pd.DataFrame({"order_id": range(1, 201), "price": [f"${x:.2f}" if i % 2 else f"€{x:.2f}" for i, x in enumerate(rng.uniform(5, 90, 200))], "qty": rng.integers(1, 5, 200)})
ue.to_csv(out / "trap_usd_eur_values.csv", index=False)
up = pd.DataFrame({"order_id": range(1, 201), "price": [f"${x:,.2f}" for x in rng.uniform(5, 900, 200)], "qty": rng.integers(1, 5, 200)})
up.to_csv(out / "text_prices_usd_only.csv", index=False)
orders = pd.DataFrame({"order_id": range(1, 101), "amount": rng.integers(100, 900, 100)}); pay = orders.copy(); pay["method"] = "card"
pay.loc[[3, 17, 42], "amount"] += 55; orders.to_csv(out / "trap_orders.csv", index=False); pay.to_csv(out / "trap_payments.csv", index=False)
print("wrote", sorted(p.name for p in out.iterdir()))
