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
        logger.info(f"✅ 已記錄新訂單進 CSV: {ticket} ({model_name})")

    def update_closed_trades(self, connector):
        try:
            df = pd.read_csv(self.filename)
            open_trades = df[df['Status'] == 'OPEN']
            
            if open_trades.empty:
                return
                
            # 取得歷史紀錄 (從 2000 年開始抓，確保涵蓋所有)
            from_date = datetime(2000, 1, 1)
            to_date = datetime.now()
            deals = mt5.history_deals_get(from_date, to_date)
            
            if not deals:
                return
                
            # 將 deals 轉為容易搜尋的字典 (以 position_id 作為 key)
            # MT5 中，平倉的 deal 會帶有對應開倉的 position_id
            deals_dict = {}
            for deal in deals:
                # 篩選平倉 deal (Entry Out)
                if deal.entry == 1: # DEAL_ENTRY_OUT
                    if deal.position_id not in deals_dict:
                        deals_dict[deal.position_id] = []
                    deals_dict[deal.position_id].append(deal)

            updated = False
            for idx, row in open_trades.iterrows():
                ticket = int(row['Ticket'])
                
                # 如果該訂單有對應的平倉歷史
                if ticket in deals_dict:
                    closing_deals = deals_dict[ticket]
                    total_profit = sum(d.profit for d in closing_deals)
                    
                    df.at[idx, 'Status'] = 'CLOSED'
                    df.at[idx, 'Close_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    df.at[idx, 'Profit'] = round(total_profit, 2)
                    df.at[idx, 'Outcome'] = 'WIN' if total_profit > 0 else ('LOSS' if total_profit < 0 else 'TIE')
                    updated = True
                    
                    logger.info(f"💰 訂單 {ticket} ({row['Model_Name']}) 已平倉，獲利: ${total_profit:.2f}")

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

