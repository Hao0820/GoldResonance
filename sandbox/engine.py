import time
import logging
from core.engine import StrategyEngine
from .provider import SandboxConnector
from .broker import SandboxExecutor

logger = logging.getLogger(__name__)

class SandboxEngine:
    def __init__(self, strategy_class, symbol="XAUUSD", initial_balance=10000.0, playback_speed=1.0):
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, 'data', 'xauusd_m1_backtest.parquet')
        
        self.connector = SandboxConnector(data_path)
        self.executor = SandboxExecutor(self.connector, initial_balance)
        self.connector.executor = self.executor # 相互引用以支援 get_positions_count
        self.engine = StrategyEngine(self.connector, self.executor)
        
        # 2. 註冊策略
        self.strat = strategy_class(name="沙盒回測策略", symbol=symbol)
        self.strat.magic_number = 999
        self.engine.add_strategy(self.strat)
        
        self.is_running = False
        self.playback_speed = playback_speed
        self.skip_to_end = False
        self.current_idx = 3000 # 初始化進度 (預熱 2 天)



    def run_loop_gui(self, gui):
        """專門提供給 GUI 呼叫的迴圈，會同步更新 Tkinter 介面"""
        self.is_running = True
        self.engine.start()
        
        all_times = self.connector.df_m1['time'].tolist()
        
        # 數據預熱 (僅在第一次啟動時執行)
        if self.current_idx == 3000:
            initial_sim_time = all_times[self.current_idx]
            self.connector.current_time = initial_sim_time
            gui.after(0, lambda: gui.lbl_sim_time.config(text=f"模擬時間: {initial_sim_time}"))

        for i in range(self.current_idx, len(all_times)):
            self.current_idx = i # 實時紀錄索引，支援暫停後接續
            
            if not gui.running:
                break
                
            # 如果是快速結束模式，則不進行任何 UI 更新與等待
            fast_forward = self.skip_to_end
            sim_time = all_times[i]
            self.connector.current_time = sim_time
            bar = self.connector.df_m1.iloc[i]
            
            if not fast_forward:
                # 正常模式：執行策略與 UI 更新
                price = bar['close']
                self.connector.last_price = price
                self.executor.on_tick(price, price + 3.0) 
                self.engine.on_tick()
                
                # 模擬速度延遲
                speed = self.playback_speed
                if speed > 0:
                    time.sleep(max(0.0001, 1.0 / speed))
                
                # 更新介面 (正常模式下)
                self._update_gui(gui, sim_time)
            else:
                # 極速加速模式：移除所有過程 UI 更新，追求物理極限
                self.connector.last_price = bar['close']
                self.executor.on_tick(bar['close'], bar['close'] + 3.0)
                self.engine.on_tick()
        
        # 迴圈結束或手動暫停：最後強刷一次介面，確保看到目前的最終結果
        last_price = self.connector.last_price
        summary = self.executor.get_positions_summary()
        ind = self.strat.indicators if hasattr(self.strat, 'indicators') else {}
        status_text = "模擬已結束" if self.current_idx >= len(all_times)-1 else "模擬已暫停"
        
        gui.after(0, lambda b=self.executor.balance, e=self.executor.equity, s=summary, p=last_price, i=ind, st=status_text: 
                  gui.update_account_info_sandbox(b, e, s[0], s[1], st, p, i))

        self.is_running = False
        if self.current_idx >= len(all_times)-1:
            self.report()

    def _update_gui(self, gui, sim_time):
        """輔助更新 GUI (僅在非快速模式下調用)"""
        gui.after(0, lambda t=sim_time: gui.lbl_sim_time.config(text=f"模擬時間: {t}"))
        summary = self.executor.get_positions_summary()
        status = self.strat.status if hasattr(self.strat, 'status') else "運行中"
        price_now = self.connector.last_price
        ind = self.strat.indicators if hasattr(self.strat, 'indicators') else {}
        gui.after(0, lambda b=self.executor.balance, e=self.executor.equity, s=summary, st=status, p=price_now, i=ind: 
                  gui.update_account_info_sandbox(b, e, s[0], s[1], st, p, i))


    def skip_to_end(self):
        """按鈕調用的方法"""
        self.skip_to_end = True

    def report(self):
        # 取得最後一個價格進行強制平倉
        last_price = self.connector.get_symbol_info(self.strat.symbol)['bid']
        self.executor.close_all_positions(last_price)
        
        acc = self.executor.get_account_info()
        print("\n" + "="*30)
        print("📊 模擬回測報告")
        print(f"起始金額: $10000.00")
        print(f"最終餘額: ${acc['balance']:.2f}")
        print(f"最終淨值: ${acc['equity']:.2f} (已結清所有持倉)")
        print(f"總交易筆數: {len(self.executor.history)}")
        print(f"總盈虧金額: ${(acc['balance'] - 10000.0):.2f}")
        print("="*30)
