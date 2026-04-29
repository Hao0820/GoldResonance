import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class StrategyEngine:
    def __init__(self, connector, executor):
        self.connector = connector
        self.executor = executor
        self.strategies = []
        self.is_running = False
        self.daily_stats = (0.0, 0.0, 0.0) # (realized, floating, total)

    def add_strategy(self, strategy):
        strategy.set_context(self, self.connector, self.executor)
        self.strategies.append(strategy)
        logger.info(f"已註冊策略: {strategy.name}")

    def get_strategy_names(self):
        return [s.name for s in self.strategies]

    def get_strategies_status(self):
        return ""

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
            
        # 1. 從 CSV 讀取今日損益
        total = 0.0
        if self.strategies and hasattr(self.strategies[0], 'trade_tracker') and self.strategies[0].trade_tracker:
            total = self.strategies[0].trade_tracker.get_today_profit()
            
        self.daily_stats = (total, 0.0, total)
        
        
        # 2. 驅動策略執行
        for strat in self.strategies:
            try:
                strat.on_tick(can_execute_new_trades=True)
            except Exception as e:
                logger.error(f"執行策略 {strat.name} 時發生錯誤: {e}")
