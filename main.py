import logging
from core.mt5_connector import MT5Connector
from core.execution import ExecutionManager
from core.engine import StrategyEngine
from ui.app_gui import TradingApp


from strategies.timeframe_resonance_strategy import TimeframeResonanceStrategy

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

    
    res_strat = TimeframeResonanceStrategy(name="時區共振策略", symbol="XAUUSD")
    res_strat.magic_number = 88002
    engine.add_strategy(res_strat)
    
    # Start GUI
    app = TradingApp(engine)
    app.mainloop()
    
    # Cleanup
    if engine.is_running:
        engine.stop()
    if connector.connected:
        connector.disconnect()

if __name__ == "__main__":
    main()
