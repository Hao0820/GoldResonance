import time
import pandas as pd
import numpy as np
from sandbox.engine import SandboxEngine
from strategies.timeframe_resonance_strategy import TimeframeResonanceStrategy
import cProfile
import pstats

def main():
    # 初始化引擎
    engine = SandboxEngine(TimeframeResonanceStrategy)
    engine.skip_to_end = True
    
    # 模擬 GUI 物件
    class MockGUI:
        running = True
        def after(self, *args): pass
        def update_account_info_sandbox(self, *args): pass
        lbl_sim_time = type('obj', (object,), {'config': lambda *a: None})()
    
    gui = MockGUI()
    
    print("🚀 開始效能分析...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    engine.run_loop_gui(gui)
    end_time = time.time()
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    
    print("\n" + "="*50)
    print(f"總執行時間: {end_time - start_time:.4f} 秒")
    print("="*50)
    
    # 顯示前 15 個最耗時的函數
    stats.print_stats(15)

if __name__ == "__main__":
    main()
