import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MT5Connector:
    def __init__(self):
        self.connected = False
        
    def connect(self):
        if not mt5.initialize():
            logger.error(f"MT5 初始化失敗，錯誤代碼: {mt5.last_error()}")
            return False
            
        logger.info(f"成功連線至 MT5。終端機: {mt5.terminal_info().name}")
        self.connected = True
        return True
        
    def disconnect(self):
        mt5.shutdown()
        self.connected = False
        logger.info("已與 MT5 斷開連線")
        
    def get_account_info(self):
        if not self.connected:
            return None
        account = mt5.account_info()
        if account:
             return account._asdict()
        return None
        
    def get_positions_summary(self):
        if not self.connected:
            return 0, 0.0
        positions = mt5.positions_get()
        if positions is None:
            return 0, 0.0
        return len(positions), sum(p.profit for p in positions)
        
    def get_positions_count_by_magic(self, magic_number):
        if not self.connected:
            return 0
        all_positions = mt5.positions_get()
        if all_positions is None:
            return 0
        return len([p for p in all_positions if p.magic == int(magic_number)])
        
    def get_rates(self, symbol, timeframe, count=100):
        if not self.connected:
            return None
            
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            logger.error(f"獲取 {symbol} K線資料失敗，錯誤代碼: {mt5.last_error()}")
            return None
            
        df = pd.DataFrame(rates)
        # Convert time in seconds into datetime format
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def get_symbol_info(self, symbol):
        if not self.connected:
            return None
        info = mt5.symbol_info(symbol)
        if info:
            return info._asdict()
        return None
