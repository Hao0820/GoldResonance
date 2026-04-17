import tkinter as tk
from tkinter import ttk, scrolledtext
import logging
import threading
import time
from sandbox.engine import SandboxEngine
from strategies.timeframe_resonance_strategy import TimeframeResonanceStrategy

class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))

    def emit(self, record):
        msg = self.format(record)
        def append():
            try:
                if self.text_widget.winfo_exists():
                    self.text_widget.configure(state='normal')
                    self.text_widget.insert(tk.END, msg + '\n')
                    self.text_widget.configure(state='disabled')
                    self.text_widget.yview(tk.END)
            except tk.TclError:
                pass
        try:
            if self.text_widget.winfo_exists():
                self.text_widget.after(0, append)
        except tk.TclError:
            pass

class TradingApp(tk.Tk):
    # --- Theme Colors (Class Attributes for stability) ---
    BG_DARK = "#1E1E2E"
    BG_SANDBOX = "#2B0040"
    FG_LIGHT = "#CDD6F4"
    ACCENT = "#89B4FA"
    PANEL_BG = "#313244"
    PANEL_SANDBOX = "#3D005C"
    BLACK_BG = "#11111B"
    GREEN_ACC = "#A6E3A1"
    RED_ACC = "#F38BA8"
    YELLOW_ACC = "#F9E2AF"

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.title("MT5 多策略自動交易中心")
        self.geometry("900x650") # Slightly taller for sandbox controls
        
        self.setup_ui()
        self.setup_logging()
        
        self.monitor_thread = None
        self.running = False
        self.sandbox_mode = False
        self.sandbox_engine = None
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass
            
        self.configure(bg=self.BG_DARK)
        
        style.configure(".", background=self.BG_DARK, foreground=self.FG_LIGHT, font=('Segoe UI', 10))
        style.configure("TFrame", background=self.BG_DARK)
        style.configure("TLabel", background=self.BG_DARK, foreground=self.FG_LIGHT)
        style.configure("TCheckbutton", background=self.BG_DARK, foreground=self.FG_LIGHT, font=('Segoe UI', 10, 'bold'))
        style.map("TCheckbutton", background=[("active", self.BG_DARK)])
        
        style.configure("TButton", font=('Segoe UI', 10, 'bold'), borderwidth=0, padding=8, background=self.PANEL_BG, foreground=self.FG_LIGHT)
        style.map("TButton", 
                  background=[("active", self.ACCENT), ("disabled", self.PANEL_BG)], 
                  foreground=[("active", self.BLACK_BG), ("disabled", "#6C7086")])
        
        style.configure("TLabelframe", background=self.BG_DARK, bordercolor=self.PANEL_BG, borderwidth=2)
        style.configure("TLabelframe.Label", background=self.BG_DARK, foreground=self.ACCENT, font=('Segoe UI', 12, 'bold'))
        
        # Control Panel
        self.control_frame = ttk.Frame(self, padding="20")
        self.control_frame.pack(fill=tk.X)
        
        self.btn_connect = ttk.Button(self.control_frame, text="🔗 連線 MT5", command=self.on_connect)
        self.btn_connect.pack(side=tk.LEFT, padx=8)
        
        self.btn_start = ttk.Button(self.control_frame, text="▶ 啟動 EA", command=self.on_start, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=8)
        
        self.btn_stop = ttk.Button(self.control_frame, text="⏹ 停止 EA", command=self.on_stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=8)
        
        self.lbl_status = ttk.Label(self.control_frame, text="連線狀態: 🔴 已斷線", foreground=self.RED_ACC, font=('Segoe UI', 11, 'bold'))
        self.lbl_status.pack(side=tk.RIGHT, padx=10)
        
        self.btn_sandbox = ttk.Button(self.control_frame, text="🧪 開放沙盒模式", command=self.toggle_sandbox)
        self.btn_sandbox.pack(side=tk.RIGHT, padx=5)
        
        self.btn_skip = ttk.Button(self.control_frame, text="⏭ 快速結束", command=self.on_skip_sandbox)
        # 預設隱藏，僅在沙盒模式顯示
        self.btn_skip.pack_forget()
        

        # --- Sandbox Info (Hidden by default) ---
        self.sandbox_frame = ttk.Frame(self, padding="5")
        
        self.lbl_sim_time = ttk.Label(self.sandbox_frame, text="模擬時間: --", foreground=self.YELLOW_ACC, font=('Segoe UI', 10, 'bold'))
        self.lbl_sim_time.pack(side=tk.LEFT, padx=20)
        
        # --- Middle Container (Parallel Dashboards) ---
        mid_container = ttk.Frame(self)
        mid_container.pack(fill=tk.X, padx=20, pady=5)
        
        # Left Panel: Account Dashboard
        info_frame = ttk.LabelFrame(mid_container, text=" 交易資訊 ", padding="15")
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.lbl_balance = ttk.Label(info_frame, text="餘額: $0.00", font=('Segoe UI', 12, 'bold'), foreground=self.GREEN_ACC)
        self.lbl_balance.grid(row=0, column=0, sticky=tk.W, padx=10, pady=2)
        
        self.lbl_equity = ttk.Label(info_frame, text="淨值: $0.00", font=('Segoe UI', 12, 'bold'), foreground=self.GREEN_ACC)
        self.lbl_equity.grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)
        
        self.lbl_positions = ttk.Label(info_frame, text="持倉: 0 筆", font=('Segoe UI', 10, 'bold'), foreground="#BAC2DE")
        self.lbl_positions.grid(row=1, column=0, sticky=tk.W, padx=10, pady=2)
        
        self.lbl_profit = ttk.Label(info_frame, text="未實現: $0.00", font=('Segoe UI', 10, 'bold'), foreground="#BAC2DE")
        self.lbl_profit.grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)
        
        self.lbl_logic_status = ttk.Label(info_frame, text="狀態: 等待中", font=('Segoe UI', 9), foreground=self.YELLOW_ACC, wraplength=400, justify=tk.LEFT)
        self.lbl_logic_status.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=10, pady=5)

        # Right Panel: Technical Indicator Monitor (Small Font)
        self.indicator_frame = ttk.LabelFrame(mid_container, text=" 技術指標監控", padding="10")
        self.indicator_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Live Price Row inside indicator frame
        price_subframe = ttk.Frame(self.indicator_frame)
        price_subframe.pack(fill=tk.X, pady=(0, 5))
        self.lbl_ask = ttk.Label(price_subframe, text="Ask: 0.00", font=('Segoe UI', 9, 'bold'), foreground=self.RED_ACC)
        self.lbl_ask.pack(side=tk.LEFT, padx=5)
        self.lbl_bid = ttk.Label(price_subframe, text="Bid: 0.00", font=('Segoe UI', 9, 'bold'), foreground=self.GREEN_ACC)
        self.lbl_bid.pack(side=tk.LEFT, padx=5)

        # Indicator Table
        table_frame = ttk.Frame(self.indicator_frame)
        table_frame.pack(fill=tk.BOTH)
        
        small_font = ('Segoe UI', 8)
        header_font = ('Segoe UI', 8, 'bold')
        headers = ["TF", "Upper", "Mid", "Lower", "12EMA"]
        for i, h in enumerate(headers):
            ttk.Label(table_frame, text=h, font=header_font, foreground=self.ACCENT).grid(row=0, column=i, padx=5, pady=2)
            
        self.indicator_labels = {}
        tfs = ["H1", "M30", "M5"]
        for r, tf in enumerate(tfs, start=1):
            ttk.Label(table_frame, text=tf, font=header_font).grid(row=r, column=0, padx=5, pady=1)
            self.indicator_labels[tf] = {
                "H": ttk.Label(table_frame, text="0.0", font=small_font, foreground=self.RED_ACC),
                "M": ttk.Label(table_frame, text="0.0", font=small_font, foreground=self.YELLOW_ACC),
                "L": ttk.Label(table_frame, text="0.0", font=small_font, foreground=self.GREEN_ACC),
                "E": ttk.Label(table_frame, text="0.0", font=small_font, foreground="#CAA4F4")
            }
            self.indicator_labels[tf]["H"].grid(row=r, column=1, padx=5, pady=1)
            self.indicator_labels[tf]["M"].grid(row=r, column=2, padx=5, pady=1)
            self.indicator_labels[tf]["L"].grid(row=r, column=3, padx=5, pady=1)
            self.indicator_labels[tf]["E"].grid(row=r, column=4, padx=5, pady=1)

        # Log Section
        log_frame = ttk.LabelFrame(self, text=" 系統日誌 ", padding="15")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.log_area = scrolledtext.ScrolledText(log_frame, state='disabled', height=10, font=('Consolas', 10), bg=self.BLACK_BG, fg="#A6ADC8", insertbackground=self.FG_LIGHT, borderwidth=0, highlightthickness=0)
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def setup_logging(self):
        handler = TextHandler(self.log_area)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
        
        self.anim_frames = ['|', '/', '-', '\\']
        self.anim_idx = 0
                
    def on_connect(self):
        if self.sandbox_mode:
            logging.info("沙盒模式下無需連線 MT5")
            return
        self.lbl_status.config(text="連線狀態: 🟡 連線中...", foreground="#F9E2AF")
        # 開啟獨立 Thread，避免阻塞 GUI
        threading.Thread(target=self._connect_task, daemon=True).start()

    def _connect_task(self):
        success = self.engine.connector.connect()
        self.after(0, self._on_connect_result, success)
        
    def _on_connect_result(self, success):
        if success:
            self.lbl_status.config(text="連線狀態: 🟢 已連線", foreground="#A6E3A1")
            self.btn_start.config(state=tk.NORMAL)
            self.update_account_info()
        else:
            self.lbl_status.config(text="連線狀態: 🔴 已斷線", foreground="#F38BA8")
            self.show_mt5_popup()

    def show_mt5_popup(self):
        popup = tk.Toplevel(self)
        popup.title("連線失敗")
        popup.geometry("300x120")
        popup.configure(bg="#1E1E2E")
        popup.transient(self)
        popup.grab_set()
        
        lbl = ttk.Label(popup, text="無法連線，請問要呼叫 MT5 起來嗎？", background="#1E1E2E", foreground="#CDD6F4", font=('Segoe UI', 10))
        lbl.pack(pady=15)
        
        btn_frame = ttk.Frame(popup)
        btn_frame.pack(pady=5)
        
        def launch_mt5():
            import subprocess, os
            path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
            if os.path.exists(path):
                subprocess.Popen([path])
            popup.destroy()
            
        ttk.Button(btn_frame, text="呼叫 MT5 起來", command=launch_mt5).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="取消", command=popup.destroy).pack(side=tk.LEFT, padx=10)
            
    def on_start(self):
        if self.sandbox_mode:
            self.running = True
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.monitor_thread = threading.Thread(target=self.run_sandbox_task, daemon=True)
            self.monitor_thread.start()
        elif self.engine.start():
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.running = True
            self.monitor_thread = threading.Thread(target=self.bot_loop, daemon=True)
            self.monitor_thread.start()

    def run_sandbox_task(self):
        try:
            if not hasattr(self, 'sandbox_engine') or self.sandbox_engine is None:
                self.sandbox_engine = SandboxEngine(TimeframeResonanceStrategy, playback_speed=1.0)
            
            # 啟動模擬循環
            self.sandbox_engine.run_loop_gui(self)
            # 正常結束後，也執行停止動作以恢復按鈕狀態
            self.on_stop()
        except Exception as e:
            logging.error(f"沙盒執行出錯: {e}")
            self.on_stop()
            
    def on_stop(self):
        if not self.running:
            return
            
        self.engine.stop()
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        
        # 如果是沙盒模式，也要更新按鈕文字回到初始狀態
        if self.sandbox_mode and hasattr(self, 'sandbox_engine') and self.sandbox_engine:
            self.sandbox_engine.skip_to_end = False
            self.btn_skip.config(text="⏭ 快速結束")
        
    def on_closing(self):
        self.running = False
        if self.engine.is_running:
            self.engine.stop()
        if self.engine.connector.connected:
            self.engine.connector.disconnect()
        self.destroy()

    def toggle_sandbox(self):
        self.sandbox_mode = not self.sandbox_mode
        style = ttk.Style()
        if self.sandbox_mode:
            self.btn_sandbox.config(text="🚪 關閉沙盒模式")
            self.configure(bg=self.BG_SANDBOX)
            
            # 提前建立沙盒引擎
            if not hasattr(self, 'sandbox_engine') or self.sandbox_engine is None:
                self.sandbox_engine = SandboxEngine(TimeframeResonanceStrategy, playback_speed=1.0)
            
            # 更新全局 Style
            style.configure(".", background=self.BG_SANDBOX)
            style.configure("TLabel", background=self.BG_SANDBOX, foreground=self.FG_LIGHT)
            style.configure("TLabelframe", background=self.BG_SANDBOX, foreground=self.FG_LIGHT)
            style.configure("TLabelframe.Label", background=self.BG_SANDBOX, foreground=self.FG_LIGHT)
            
            self.sandbox_frame.pack(fill=tk.X, padx=20, pady=5)
            self.indicator_frame.config(text=" 技術指標監控 [沙盒模式] ")
            logging.info("🔮 已進入沙盒模式。現在您可以啟動模擬。")
            self.sandbox_frame.pack(fill=tk.X, padx=20, pady=(0, 10), after=self.control_frame)
            self.btn_skip.pack(side=tk.RIGHT, padx=5, before=self.btn_sandbox)
            self.btn_connect.config(state=tk.DISABLED)
            self.btn_start.config(state=tk.NORMAL)
            logging.info("🧪 已進入沙盒模擬模式")
        else:
            # 離開沙盒模式：自動停止運行中的 EA
            if self.running:
                logging.info("🚪 正在離開沙盒，自動關閉運行中的任務...")
                self.on_stop()
                
            self.btn_sandbox.config(text="🧪 開放沙盒模式")
            self.configure(bg=self.BG_DARK)
            
            # 還原全局 Style
            style.configure(".", background=self.BG_DARK)
            style.configure("TFrame", background=self.BG_DARK)
            style.configure("TLabel", background=self.BG_DARK)
            style.configure("TLabelframe", background=self.BG_DARK, bordercolor=self.PANEL_BG)
            style.configure("TLabelframe.Label", background=self.BG_DARK, foreground=self.ACCENT)
            style.configure("TCheckbutton", background=self.BG_DARK)
            
            self.sandbox_frame.pack_forget()
            self.btn_skip.pack_forget()
            self.indicator_frame.config(text=" 技術指標監控 ")
            self.btn_connect.config(state=tk.NORMAL)
            logging.info("🌐 已切換回即時交易模式")

    def on_skip_sandbox(self):
        if hasattr(self, 'sandbox_engine') and self.sandbox_engine:
            # 雙向切換 logic
            self.sandbox_engine.skip_to_end = not self.sandbox_engine.skip_to_end
            
            if self.sandbox_engine.skip_to_end:
                self.btn_skip.config(text="⏮ 取消快速")
                logging.info("🚀 已開啟極速模式 (暫停過程更新以追求效率)...")
            else:
                self.btn_skip.config(text="⏭ 快速結束")
                logging.info("⏮ 已恢復正常模擬速度。")
    def update_account_info(self):
        if not self.engine.connector.connected:
            return
        acc = self.engine.connector.get_account_info()
        if acc:
            self.lbl_balance.config(text=f"餘額: ${acc.get('balance', 0):.2f}")
            self.lbl_equity.config(text=f"淨值: ${acc.get('equity', 0):.2f}")
            # Position info
            count, profit = self.engine.connector.get_positions_summary()
            self.lbl_positions.config(text=f"持倉: {count} 筆")
            
            if profit > 0:
                color = self.GREEN_ACC 
            elif profit < 0:
                color = self.RED_ACC 
            else:
                color = "#BAC2DE" 
                
            self.lbl_profit.config(text=f"未實現: ${profit:.2f}", foreground=color)
            
        # Update Technical Indicators
        try:
            if not self.engine.strategies:
                return
            strat = self.engine.strategies[0]
            # Ask/Bid
            info = self.engine.connector.get_symbol_info(strat.symbol)
            if info:
                self.lbl_ask.config(text=f"Ask: {info['ask']:.2f}")
                self.lbl_bid.config(text=f"Bid: {info['bid']:.2f}")
                
            # BB/EMA
            if hasattr(strat, 'indicators'):
                data = strat.indicators
                for tf, labels in self.indicator_labels.items():
                    if tf in data:
                        labels["H"].config(text=f"{data[tf]['H']:.1f}")
                        labels["M"].config(text=f"{data[tf]['M']:.1f}")
                        labels["L"].config(text=f"{data[tf]['L']:.1f}")
                        labels["E"].config(text=f"{data[tf]['EMA']:.1f}")
        except Exception:
            pass
            
        if self.running:
            self.anim_idx = (self.anim_idx + 1) % len(self.anim_frames)
            anim_char = self.anim_frames[self.anim_idx]
            status_text = f"正在執行運算 {anim_char}\n" + self.engine.get_strategies_status()
        else:
            status_text = "已停止掃描"
            
        self.lbl_logic_status.config(text=f"狀態: {status_text}")
            
    def update_account_info_sandbox(self, balance, equity, pos_count, profit, strat_status, price, indicators):
        """專供沙盒模式調用的 UI 更新函數"""
        self.lbl_balance.config(text=f"餘額: ${balance:.2f}")
        self.lbl_equity.config(text=f"淨值: ${equity:.2f}")
        self.lbl_positions.config(text=f"持倉: {pos_count} 筆")
        
        color = self.GREEN_ACC if profit > 0 else (self.RED_ACC if profit < 0 else "#BAC2DE")
        self.lbl_profit.config(text=f"未實現: ${profit:.2f}", foreground=color)
        self.lbl_logic_status.config(text=f"狀態: [沙盒] {strat_status}")
        
        # 更新沙盒報價與指標
        self.lbl_ask.config(text=f"Ask: {price:.2f}")
        self.lbl_bid.config(text=f"Bid: {price:.2f}")
        
        if indicators:
            for tf, labels in self.indicator_labels.items():
                if tf in indicators:
                    labels["H"].config(text=f"{indicators[tf]['H']:.1f}")
                    labels["M"].config(text=f"{indicators[tf]['M']:.1f}")
                    labels["L"].config(text=f"{indicators[tf]['L']:.1f}")
                    labels["E"].config(text=f"{indicators[tf]['EMA']:.1f}")

    def bot_loop(self):
        while self.running:
            try:
                self.engine.on_tick()
                if self.winfo_exists():
                    self.after(0, self.update_account_info)
            except Exception as e:
                logging.error(f"背景迴圈發生錯誤: {e}")
            time.sleep(1)

