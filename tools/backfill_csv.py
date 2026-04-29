import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

mt5.initialize()

import os
csv_path = r'C:\Users\user\Documents\projects\trade_records.csv'
df = pd.read_csv(csv_path)
df['Close_Time'] = df['Close_Time'].astype(object)
df['Outcome'] = df['Outcome'].astype(object)

print("CSV before:")
print(df[['Ticket', 'Status', 'Profit']].to_string())
print()

for idx, row in df[df['Profit'] == 0.0].iterrows():
    ticket = int(row['Ticket'])
    deals = mt5.history_deals_get(position=ticket)
    if not deals:
        print(f"Ticket {ticket}: MT5 no data found")
        continue

    close_deals = [d for d in deals if d.entry == 1]
    if not close_deals:
        print(f"Ticket {ticket}: still open (no closing deal)")
        continue

    profit = round(sum(d.profit for d in close_deals), 2)
    close_time = datetime.fromtimestamp(close_deals[-1].time).strftime('%Y-%m-%d %H:%M:%S')
    outcome = 'WIN' if profit > 0 else 'LOSS'

    df.at[idx, 'Status'] = 'CLOSED'
    df.at[idx, 'Close_Time'] = close_time
    df.at[idx, 'Profit'] = profit
    df.at[idx, 'Outcome'] = outcome
    print(f"Ticket {ticket}: {outcome}  profit=${profit:.2f}  close={close_time}")

df.to_csv(csv_path, index=False)
print()
print("Done. CSV updated.")
mt5.shutdown()
