import logging
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

class VirtualPosition:
    def __init__(self, ticket, symbol, type, volume, open_price, sl, tp, magic, comment):
        self.ticket = ticket
        self.symbol = symbol
        self.type = type
        self.volume = volume
        self.open_price = open_price
        self.sl = sl
        self.tp = tp
        self.magic = magic
        self.comment = comment
        self.profit = 0.0
        self.close_price = None
        self.status = "OPEN"

class SandboxExecutor:
    def __init__(self, connector, initial_balance=10000.0):
        self.connector = connector # SandboxConnector
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions = []
        self.history = []
        self.ticket_counter = 1000
        
    def get_account_info(self):
        self._update_equity()
        return {"balance": self.balance, "equity": self.equity}

    def get_positions_summary(self):
        open_pos = [p for p in self.positions if p.status == "OPEN"]
        profit = sum(p.profit for p in open_pos)
        return len(open_pos), profit

    def get_positions_count_by_magic(self, magic):
        return len([p for p in self.positions if p.status == "OPEN" and p.magic == magic])

    def send_order(self, symbol, order_type, volume, price, sl, tp, comment="", magic=0):
        pos = VirtualPosition(
            ticket=self.ticket_counter,
            symbol=symbol,
            type=order_type,
            volume=volume,
            open_price=price,
            sl=sl,
            tp=tp,
            magic=magic,
            comment=comment
        )
        self.positions.append(pos)
        self.ticket_counter += 1
        logger.info(f"[沙盒下單] {symbol} {'BUY' if order_type==0 else 'SELL'} 價格:{price} SL:{sl} TP:{tp}")
        return type('obj', (object,), {"order": pos.ticket, "retcode": mt5.TRADE_RETCODE_DONE})

    def on_tick(self, current_bid, current_ask):
        """每一跳檢查訂單是否觸碰損蓋"""
        for p in self.positions:
            if p.status != "OPEN":
                continue
                
            # 計算浮動損益
            if p.type == mt5.ORDER_TYPE_BUY:
                p.profit = (current_bid - p.open_price) * p.volume * 100 # 簡化金價計算
                # 檢查 SL/TP
                if p.sl > 0 and current_bid <= p.sl:
                    self._close_position(p, p.sl, "SL")
                elif p.tp > 0 and current_bid >= p.tp:
                    self._close_position(p, p.tp, "TP")
            else:
                p.profit = (p.open_price - current_ask) * p.volume * 100
                if p.sl > 0 and current_ask >= p.sl:
                    self._close_position(p, p.sl, "SL")
                elif p.tp > 0 and current_ask <= p.tp:
                    self._close_position(p, p.tp, "TP")
        
        self._update_equity()

    def _close_position(self, p, price, reason):
        p.status = "CLOSED"
        p.close_price = price
        self.balance += p.profit
        self.history.append(p)
        logger.info(f"[沙盒平倉] #{p.ticket} 原因:{reason} 點位:{price} 損益:{p.profit:.2f}")

    def close_all_positions(self, final_price):
        """模擬結束時強制結清所有持倉"""
        open_pos = [p for p in self.positions if p.status == "OPEN"]
        for p in open_pos:
            # 重新計算最後損益
            if p.type == mt5.ORDER_TYPE_BUY:
                p.profit = (final_price - p.open_price) * p.volume * 100
            else:
                p.profit = (p.open_price - final_price) * p.volume * 100
            self._close_position(p, final_price, "FINAL_LIQUIDATION")

    def _update_equity(self):
        open_profit = sum(p.profit for p in self.positions if p.status == "OPEN")
        self.equity = self.balance + open_profit

    def calculate_lot_size(self, symbol, risk_percent, sl_pips):
        # 簡單化回測手數計算
        balance = self.balance
        risk_amount = balance * (risk_percent / 100.0)
        # 假設 XAUUSD 1 手 1 點 = $100
        lot = risk_amount / (sl_pips * 100)
        return max(0.01, round(lot, 2))
