class BaseStrategy:
    def __init__(self, name):
        self.name = name
        self.engine = None
        self.connector = None
        self.executor = None
        self.status = "等待中"
        self.magic_number = 0  # To be set by main
        self.lot_size = getattr(self, 'lot_size', 0.1) 
        self.max_positions = getattr(self, 'max_positions', 1) 
        
        self.last_signal_time = None # 防止同一根K線無限連發開單的時間鎖
        
    def set_context(self, engine, connector, executor):
        self.engine = engine
        self.connector = connector
        self.executor = executor
        
    def on_tick(self, can_execute_new_trades: bool):
        raise NotImplementedError("Strategies must implement on_tick")
