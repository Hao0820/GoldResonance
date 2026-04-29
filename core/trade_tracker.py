import pandas as pd
import os
import MetaTrader5 as mt5
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TradeTracker:
    def __init__(self, filename="trade_records.csv"):
        self.filename = filename
        self.columns = [
            "Ticket", "Open_Time", "Model_Name", "Type", "Lots", 
            "Open_Price", "SL", "TP", "Status", "Close_Time", "Profit", "Outcome"
        ]
        
        # 確保檔案存在
        if not os.path.exists(self.filename):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.filename, index=False)

    def log_open_trade(self, ticket, model_name, order_type, lots, open_price, sl, tp):
        df = pd.read_csv(self.filename)
        
        # 檢查是否已存在
        if ticket in df['Ticket'].values:
            return
            
        new_row = {
            "Ticket": int(ticket),
            "Open_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Model_Name": model_name,
            "Type": order_type,
            "Lots": lots,
            "Open_Price": open_price,
            "SL": sl,
            "TP": tp,
            "Status": "OPEN",
            "Close_Time": "",
            "Profit": 0.0,
            "Outcome": ""
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.filename, index=False)

    def update_closed_trades(self, connector):
        try:
            df = pd.read_csv(self.filename)
            
            # 找出所有 Profit 仍為 0 的訂單 (不論 OPEN 或 CLOSED，確保補填)
            unresolved = df[df['Profit'] == 0.0]
            
            if unresolved.empty:
                return
            
            updated = False
            for idx, row in unresolved.iterrows():
                ticket = int(row['Ticket'])
                
                # 直接用 position ID (= ticket) 去 MT5 查這張單的成交記錄
                deals = mt5.history_deals_get(position=ticket)
                
                if not deals:
                    continue
                
                # 找出所有平倉 deal (DEAL_ENTRY_OUT = 1)
                close_deals = [d for d in deals if d.entry == 1]
                
                if not close_deals:
                    continue  # 還在持倉中，尚未平倉
                
                total_profit = sum(d.profit for d in close_deals)
                close_time = datetime.fromtimestamp(close_deals[-1].time).strftime("%Y-%m-%d %H:%M:%S")
                outcome = 'WIN' if total_profit > 0 else ('LOSS' if total_profit < 0 else 'TIE')
                
                df.at[idx, 'Status'] = 'CLOSED'
                df.at[idx, 'Close_Time'] = close_time
                df.at[idx, 'Profit'] = round(total_profit, 2)
                df.at[idx, 'Outcome'] = outcome
                updated = True
                
                logger.info(f"💰 [訂單 {ticket}] [{row['Model_Name']}] 已平倉 | 結果: {outcome} | 獲利: ${total_profit:.2f}")
            
            if updated:
                df.to_csv(self.filename, index=False)
                self.print_stats(df)
                
        except Exception as e:
            logger.error(f"更新 CSV 歷史訂單失敗: {e}")

    def print_stats(self, df=None):
        if df is None:
            df = pd.read_csv(self.filename)
            
        closed_trades = df[df['Status'] == 'CLOSED']
        if closed_trades.empty:
            return
            
        stats = closed_trades.groupby('Model_Name').agg(
            Total_Trades=('Ticket', 'count'),
            Wins=('Outcome', lambda x: (x == 'WIN').sum()),
            Total_Profit=('Profit', 'sum')
        )
        
        stats['Win_Rate'] = (stats['Wins'] / stats['Total_Trades'] * 100).round(1)
        
        logger.info("\n📊 === 模型實盤績效統計 ===")
        for model, row in stats.iterrows():
            logger.info(f"[{model}] 交易: {int(row['Total_Trades'])} | 勝率: {row['Win_Rate']}% | 淨利: ${row['Total_Profit']:.2f}")
        logger.info("==========================\n")

    def get_today_profit(self):
        try:
            df = pd.read_csv(self.filename)
            closed_trades = df[df['Status'] == 'CLOSED']
            if closed_trades.empty: return 0.0
            
            # 篩選今天的訂單 (依據 Close_Time)
            today_str = datetime.now().strftime("%Y-%m-%d")
            today_trades = closed_trades[closed_trades['Close_Time'].str.startswith(today_str, na=False)]
            
            return float(today_trades['Profit'].sum())
        except Exception:
            return 0.0

