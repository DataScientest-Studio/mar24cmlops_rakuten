import duckdb
from datetime import datetime 
import re

def mise_en_forme(text: str):
    text=re.sub("'"," ",text)
    return "'"+text+"'"

db_conn=duckdb.connect(database='data/rakuten_db.duckdb',read_only=False)

date_str=mise_en_forme(str(datetime.now()))

limite=db_conn.sql('SELECT MAX(listing_id) FROM fact_listings;').fetchone()[0]
print(limite)
#for i in range(limite+1):
db_conn.sql(f"Update fact_listings SET user=t'oto', validate_datetime={date_str} WHERE listing_id>56000;")
#limite=db_conn.sql('SELECT MIN(listing_id) FROM fact_listings;').fetchone()[0]
#print(limite)
print(db_conn.sql('SELECT validate_datetime,user FROM fact_listings WHERE listing_id=0;').df().head())
print(db_conn.sql('SELECT validate_datetime,user FROM fact_listings WHERE listing_id=40000;').df().head())
print(db_conn.sql('SELECT validate_datetime,user FROM fact_listings WHERE listing_id=60000;').df().head())
print(db_conn.sql('SELECT validate_datetime,user FROM fact_listings WHERE listing_id=70000;').df().head())