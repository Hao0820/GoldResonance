class BaseStrategy:
    def __init__(self, name: str):
        self.name = name
        self.engine = None
        self.connector = None
        self.executor = None
        self.status = "等待中"
        self.magic_number = 0
        self.lot_size = 0.1          # 子類別可在自己的 __init__ 覆蓋
        self.max_positions = 1       # 同時最多持有幾張相同方向的單
        self.last_signal_time = None # K 線時間鎖：防止同一根 K 棒重複觸發開單
        
    def set_context(self, engine, connector, executor):
        self.engine = engine
        self.connector = connector
        self.executor = executor
        
    def on_tick(self, can_execute_new_trades: bool):
        raise NotImplementedError("Strategies must implement on_tick")
