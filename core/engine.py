import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class StrategyEngine:
    def __init__(self, connector, executor):
        self.connector = connector
        self.executor = executor
        self.strategies = []
        self.is_running = False
        self.daily_loss_limit = 200.0  # 每日最大虧損鎖定門檻
        self.daily_stats = (0.0, 0.0, 0.0) # (realized, floating, total)
        self.risk_management_enabled = True # 風控開關

    def add_strategy(self, strategy):
        strategy.set_context(self, self.connector, self.executor)
        self.strategies.append(strategy)
        logger.info(f"已註冊策略: {strategy.name}")

    def get_strategy_names(self):
        return [s.name for s in self.strategies]

    def get_strategies_status(self):
        if not self.strategies:
            return "無掛載策略"
        strategies_str = "\n".join([f"  ➤ [{s.name}] {getattr(s, 'status', '運作中')}" for s in self.strategies])
        
        # 增加今日盈虧顯示
        realized, floating, total = self.daily_stats
        if self.risk_management_enabled:
            risk_status = "🟢 風控開啟" if total > -self.daily_loss_limit else "🔴 熔斷鎖定(>-$200)"
        else:
            risk_status = "⚪ 風控已關閉"
        
        pnl_msg = f"今日累計損益: ${total:.2f} ({risk_status})"
        
        return f"{pnl_msg}\n{strategies_str}"

    def start(self):
        if not self.connector.connected:
            logger.error("無法啟動引擎：MT5 尚未連線。")
            return False
            
        self.is_running = True
        logger.info("策略引擎已啟動。")
        return True
        
    def stop(self):
        self.is_running = False
        logger.info("策略引擎已停止。")


    def on_tick(self):
        if not self.is_running:
            return
            
        # 1. 檢查今日風控狀態
        realized, floating, total = self.connector.get_daily_pnl()
        self.daily_stats = (realized, floating, total)
        
        can_trade = True
        if self.risk_management_enabled and total <= -self.daily_loss_limit:
            can_trade = False
            
        # 2. 驅動策略執行
        for strat in self.strategies:
            try:
                strat.on_tick(can_execute_new_trades=can_trade)
            except Exception as e:
                logger.error(f"執行策略 {strat.name} 時發生錯誤: {e}")
