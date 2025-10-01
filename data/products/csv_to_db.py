import pandas as pd
import sqlite3

df = pd.read_csv("products_db.csv")
conn = sqlite3.connect("products.db")
df.to_sql("products", conn, if_exists="replace", index=False)
conn.close()