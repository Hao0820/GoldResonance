import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
from core.mt5_connector import MT5Connector
from core.execution import ExecutionManager
from core.engine import StrategyEngine
from ui.app_gui import TradingApp


from strategies.ml_resonance_strategy import MLResonanceStrategy
from core.trade_tracker import TradeTracker

def main():
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Init Core Modules
    connector = MT5Connector()
    executor = ExecutionManager(connector)
    engine = StrategyEngine(connector, executor)
    
    tracker = TradeTracker()
    ml_ai = MLResonanceStrategy(name="雙引擎 AI (A/B模型)", symbol="XAUUSD", trade_tracker=tracker)
    engine.add_strategy(ml_ai)
    
    # Start GUI
    try:
        app = TradingApp(engine)
        app.mainloop()
    except KeyboardInterrupt:
        logging.info("接收到中斷訊號，正在關閉程式...")
    finally:
        # Cleanup
        if engine.is_running:
            engine.stop()
        if connector.connected:
            connector.disconnect()
        logging.info("程式已安全關閉。")

if __name__ == "__main__":
    main()
