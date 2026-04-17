import MetaTrader5 as mt5
import logging

logger = logging.getLogger(__name__)

class ExecutionManager:
    def __init__(self, connector):
        self.connector = connector

    def calculate_lot_size(self, symbol, risk_percent, sl_pips):
        account = self.connector.get_account_info()
        symbol_info = self.connector.get_symbol_info(symbol)
        
        if not account or not symbol_info:
            return 0.0
            
        balance = account.get('balance', 0)
        risk_amount = balance * (risk_percent / 100.0)
        
        tick_value = symbol_info.get('trade_tick_value', 0)
        tick_size = symbol_info.get('trade_tick_size', 0)
        
        # Simplified lot calculation
        if sl_pips == 0 or tick_size == 0 or tick_value == 0:
            return symbol_info.get('volume_min', 0.01)
            
        loss_per_lot = (sl_pips / tick_size) * tick_value
        if loss_per_lot == 0:
            return symbol_info.get('volume_min', 0.01)
            
        lot = risk_amount / loss_per_lot
        
        min_lot = symbol_info.get('volume_min', 0.01)
        max_lot = symbol_info.get('volume_max', 100.0)
        step = symbol_info.get('volume_step', 0.01)
        
        lot = round(lot / step) * step
        return max(min_lot, min(max_lot, lot))

    def send_order(self, symbol, order_type, volume, price, sl, tp, comment="EABot", magic=123456):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type, 
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": int(magic),
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"下單失敗，mt5.last_error() = {mt5.last_error()}")
            return None
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"下單失敗，回傳碼={result.retcode} 詳情={result._asdict()}")
            return None
            
        logger.info(f"下單成功，訂單號碼: {result.order}")
        return result

        return result
