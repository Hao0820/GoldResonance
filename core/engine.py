import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class StrategyEngine:
    def __init__(self, connector, executor):
        self.connector = connector
        self.executor = executor
        self.strategies = []
        self.is_running = False

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
        return f"總共 {len(self.strategies)} 個策略即時監控中:\n{strategies_str}"

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
            
        for strat in self.strategies:
            try:
                strat.on_tick(can_execute_new_trades=True)
            except Exception as e:
                logger.error(f"執行策略 {strat.name} 時發生錯誤: {e}")
