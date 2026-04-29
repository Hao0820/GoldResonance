import logging
import warnings

# 隱藏 scikit-learn 與 joblib 的並行運算警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
from core.mt5_connector import MT5Connector
from core.execution import ExecutionManager
from core.engine import StrategyEngine
from ui.app_gui import TradingApp


from strategies.ml_resonance_strategy import MLResonanceStrategy

def main():
    # 確保不會再有任何訊息印到終端機 (黑色視窗) 裡，全部導向 GUI
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Init Core Modules
    connector = MT5Connector()
    executor = ExecutionManager(connector)
    engine = StrategyEngine(connector, executor)
    
    # Register Strategies into Hub
    ml_ai = MLResonanceStrategy(name="AI全自動交易(XGB+RF)", symbol="XAUUSD")
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
