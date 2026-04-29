import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import logging
import threading
import time

logger = logging.getLogger(__name__)

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
    # --- Theme Colors ---
    BG_DARK = "#1E1E2E"
    FG_LIGHT = "#CDD6F4"
    ACCENT = "#89B4FA"
    PANEL_BG = "#313244"
    BLACK_BG = "#11111B"
    GREEN_ACC = "#A6E3A1"
    RED_ACC = "#F38BA8"
    YELLOW_ACC = "#F9E2AF"

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.title("MT5 多策略自動交易中心")
        self.geometry("900x650") 
        
        self.running = False
        
        self.setup_ui()
        self.setup_logging()
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
        
        # --- Control Panel ---
        self.control_frame = ttk.Frame(self, padding="20")
        self.control_frame.pack(fill=tk.X)
        
        self.btn_connect = ttk.Button(self.control_frame, text="🔗 連線 MT5", command=self.on_connect)
        self.btn_connect.pack(side=tk.LEFT, padx=8)
        
        # --- Lots Entry (客製化深色樣式) ---
        ttk.Label(self.control_frame, text="手數:", font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=(20, 2))
        self.ent_lots = tk.Entry(
            self.control_frame, 
            width=6, 
            font=('Consolas', 11, 'bold'),
            bg=self.BLACK_BG,           # 深色背景
            fg="white",                 # 白色字體
            insertbackground="white",   # 白色輸入游標
            highlightthickness=1,       # 邊框粗細
            highlightbackground="#585b70", # 預設邊框顏色
            highlightcolor=self.ACCENT,    # 點擊時邊框變亮
            relief=tk.FLAT
        )
        self.ent_lots.insert(0, str(getattr(self.engine.strategies[0], 'lot_size', 0.1)) if self.engine.strategies else "0.1")
        self.ent_lots.pack(side=tk.LEFT, padx=5)
        
        # (權重滑桿已移除，因為現在是雙模型獨立運行)        
        # 手數輸入框自動同步
        self.ent_lots.bind('<Return>', lambda e: self.update_strategy_settings(silent=True))
        self.ent_lots.bind('<FocusOut>', lambda e: self.update_strategy_settings(silent=True))
        
        self.btn_start = ttk.Button(self.control_frame, text="▶ 啟動 EA", command=self.on_start, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=8)
        
        self.btn_stop = ttk.Button(self.control_frame, text="⏹ 停止 EA", command=self.on_stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=8)
        
        self.lbl_status = ttk.Label(self.control_frame, text="連線狀態: 🔴 已斷線", foreground=self.RED_ACC, font=('Segoe UI', 11, 'bold'))
        self.lbl_status.pack(side=tk.RIGHT, padx=10)
        
        # --- Middle Dashboards ---
        mid_container = ttk.Frame(self)
        mid_container.pack(fill=tk.X, padx=20, pady=5)
        
        # Account Info
        self.info_frame = ttk.LabelFrame(mid_container, text=" 交易資訊 ", padding="15")
        self.info_frame.pack(fill=tk.BOTH, expand=True)
        
        self.lbl_balance = ttk.Label(self.info_frame, text="餘額: $0.00", font=('Segoe UI', 12, 'bold'), foreground=self.GREEN_ACC)
        self.lbl_balance.grid(row=0, column=0, sticky=tk.W, padx=10, pady=2)
        
        self.lbl_equity = ttk.Label(self.info_frame, text="淨值: $0.00", font=('Segoe UI', 12, 'bold'), foreground=self.GREEN_ACC)
        self.lbl_equity.grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)
        
        self.lbl_model_A = ttk.Label(self.info_frame, text="[利潤引擎A] 多: 0.0% | 空: 0.0%", font=('Consolas', 11, 'bold'), foreground="#F38BA8")
        self.lbl_model_A.grid(row=0, column=2, sticky=tk.W, padx=20, pady=2)
        
        self.lbl_positions = ttk.Label(self.info_frame, text="持倉: 0 筆", font=('Segoe UI', 10, 'bold'), foreground="#BAC2DE")
        self.lbl_positions.grid(row=1, column=0, sticky=tk.W, padx=10, pady=2)
        
        self.lbl_profit = ttk.Label(self.info_frame, text="未實現: $0.00", font=('Segoe UI', 10, 'bold'), foreground="#BAC2DE")
        self.lbl_profit.grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)
        
        self.lbl_model_B = ttk.Label(self.info_frame, text="[勝率引擎B] 多: 0.0% | 空: 0.0%", font=('Consolas', 11, 'bold'), foreground="#89B4FA")
        self.lbl_model_B.grid(row=1, column=2, sticky=tk.W, padx=20, pady=2)
        
        self.lbl_daily_pnl = ttk.Label(self.info_frame, text="今日損益: $0.00", font=('Consolas', 11, 'bold'))
        self.lbl_daily_pnl.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=10, pady=5)
        
        self.lbl_logic_status = ttk.Label(self.info_frame, text="狀態: 等待中", font=('Segoe UI', 10), foreground=self.YELLOW_ACC)
        self.lbl_logic_status.grid(row=2, column=2, sticky=tk.W, padx=20, pady=5)

        # 技术指标监控 (暂時隱藏)
        self.indicator_frame = ttk.LabelFrame(mid_container, text=" 技術指標監控", padding="10")
        # self.indicator_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        price_subframe = ttk.Frame(self.indicator_frame)
        price_subframe.pack(fill=tk.X, pady=(0, 5))
        self.lbl_ask = ttk.Label(price_subframe, text="Ask: 0.00", font=('Segoe UI', 9, 'bold'), foreground=self.RED_ACC)
        self.lbl_ask.pack(side=tk.LEFT, padx=5)
        self.lbl_bid = ttk.Label(price_subframe, text="Bid: 0.00", font=('Segoe UI', 9, 'bold'), foreground=self.GREEN_ACC)
        self.lbl_bid.pack(side=tk.LEFT, padx=5)

        table_frame = ttk.Frame(self.indicator_frame)
        table_frame.pack(fill=tk.BOTH)
        
        small_font = ('Segoe UI', 8)
        header_font = ('Segoe UI', 8, 'bold')
        headers = ["TF", "Upper", "Mid", "Lower", "12EMA", "ADX"]
        for i, h in enumerate(headers):
            ttk.Label(table_frame, text=h, font=header_font, foreground=self.ACCENT).grid(row=0, column=i, padx=5, pady=2)
            
        self.indicator_labels = {}
        tfs = ["H1", "M15", "M5"]
        for r, tf in enumerate(tfs, start=1):
            ttk.Label(table_frame, text=tf, font=header_font).grid(row=r, column=0, padx=5, pady=1)
            self.indicator_labels[tf] = {
                "H": ttk.Label(table_frame, text="0.0", font=small_font, foreground=self.RED_ACC),
                "M": ttk.Label(table_frame, text="0.0", font=small_font, foreground=self.YELLOW_ACC),
                "L": ttk.Label(table_frame, text="0.0", font=small_font, foreground=self.GREEN_ACC),
                "E": ttk.Label(table_frame, text="0.0", font=small_font, foreground="#CAA4F4"),
                "X": ttk.Label(table_frame, text="0.0", font=small_font, foreground=self.ACCENT)
            }
            self.indicator_labels[tf]["H"].grid(row=r, column=1, padx=5, pady=1)
            self.indicator_labels[tf]["M"].grid(row=r, column=2, padx=5, pady=1)
            self.indicator_labels[tf]["L"].grid(row=r, column=3, padx=5, pady=1)
            self.indicator_labels[tf]["E"].grid(row=r, column=4, padx=5, pady=1)
            self.indicator_labels[tf]["X"].grid(row=r, column=5, padx=5, pady=1)

        self.lbl_slope_info = ttk.Label(self.indicator_frame, text="M15 斜率: 0.00", font=small_font, foreground=self.FG_LIGHT)
        self.lbl_slope_info.pack(fill=tk.X, pady=(10, 0), padx=5)

        # --- Logs ---
        log_frame = ttk.LabelFrame(self, text=" 系統日誌 ", padding="15")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.log_area = scrolledtext.ScrolledText(log_frame, state='disabled', height=10, font=('Consolas', 10), bg=self.BLACK_BG, fg="#A6ADC8", borderwidth=0, highlightthickness=0)
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def setup_logging(self):
        handler = TextHandler(self.log_area)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
        self.anim_frames = ['|', '/', '-', '\\']
        self.anim_idx = 0
                
    def update_strategy_settings(self, silent=False):
        """同步介面設定到策略實體"""
        try:
            lots = float(self.ent_lots.get())
            
            if lots <= 0: raise ValueError
            
            for strat in self.engine.strategies:
                strat.lot_size = lots
            
            if not silent:
                logging.info(f"✅ 參數同步: 手數={lots}")
        except ValueError:
            if not silent:
                logging.error("❌ 數值格式錯誤，請輸入正數。")

    def on_connect(self):
        self.lbl_status.config(text="連線狀態: 🟡 連線中...", foreground="#F9E2AF")
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
            
    def on_start(self):
        if self.engine.start():
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.running = True
            self.update_strategy_settings(silent=True) # 啟動時靜默同步
            self.monitor_thread = threading.Thread(target=self.bot_loop, daemon=True)
            self.monitor_thread.start()

            
    def on_stop(self):
        if not self.running: return
        self.engine.stop()
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        
    def on_closing(self):
        self.running = False
        if self.engine.is_running: self.engine.stop()
        if self.engine.connector.connected: self.engine.connector.disconnect()
        self.destroy()


    def update_account_info(self):
        if not self.engine.connector.connected: return
        try:
            # 1. 賬戶基本資訊與損益
            acc = self.engine.connector.get_account_info()
            if acc:
                self.lbl_balance.config(text=f"餘額: ${acc.get('balance', 0):.2f}")
                self.lbl_equity.config(text=f"淨值: ${acc.get('equity', 0):.2f}")
                count, profit = self.engine.connector.get_positions_summary()
                self.lbl_positions.config(text=f"持倉: {count} 筆")
                color = self.GREEN_ACC if profit > 0 else (self.RED_ACC if profit < 0 else "#BAC2DE")
                self.lbl_profit.config(text=f"未實現: ${profit:.2f}", foreground=color)
                
                realized, floating, total = self.engine.daily_stats
                pnl_color = self.GREEN_ACC if total > 0 else (self.RED_ACC if total < 0 else "#BAC2DE")
                self.lbl_daily_pnl.config(text=f"今日損益: ${total:.2f}", foreground=pnl_color)

            # 2. 策略相關數據更新
            if self.engine.strategies:
                strat = self.engine.strategies[0]
                
                # 報價更新
                info = self.engine.connector.get_symbol_info(strat.symbol)
                if info:
                    self.lbl_ask.config(text=f"Ask: {info['ask']:.2f}")
                    self.lbl_bid.config(text=f"Bid: {info['bid']:.2f}")

                # 指標監控看板已移除

                # 舊版技術指標表格更新
                if hasattr(strat, 'indicators'):
                    data = strat.indicators
                    for tf, labels in self.indicator_labels.items():
                        if tf in data:
                            objs = data[tf]
                            labels["H"].config(text=f"{objs.get('bb_h', 0):.2f}")
                            labels["M"].config(text=f"{objs.get('bb_m', 0):.2f}")
                            labels["L"].config(text=f"{objs.get('bb_l', 0):.2f}")
                            labels["E"].config(text=f"{objs.get('ema_12', 0):.2f}")
                            if 'adx' in objs:
                                labels["X"].config(text=f"{objs.get('adx', 0):.1f}")
                                
                # AI 勝率與狀態更新
                if hasattr(strat, 'model_a_buy') and strat.model_a_buy is not None:
                    self.lbl_model_A.config(text=f"[利潤引擎A] 多: {strat.model_a_buy*100:4.1f}% | 空: {strat.model_a_sell*100:4.1f}%")
                    self.lbl_model_B.config(text=f"[勝率引擎B] 多: {strat.model_b_buy*100:4.1f}% | 空: {strat.model_b_sell*100:4.1f}%")

                # (原來的 status 更新已移至下方統一處理)

        except Exception as e:
            logger.error(f"UI更新發生錯誤: {e}")
            
        if self.running:
            self.anim_idx = (self.anim_idx + 1) % len(self.anim_frames)
            frame = self.anim_frames[self.anim_idx]
            
            self.lbl_logic_status.config(text=f"狀態: 正在掃描 {frame}", foreground=self.YELLOW_ACC)
        else: 
            self.lbl_logic_status.config(text="狀態: 已停止", foreground=self.YELLOW_ACC)
            

    def bot_loop(self):
        while self.running:
            try:
                self.engine.on_tick()
                if self.winfo_exists(): self.after(0, self.update_account_info)
            except Exception as e: logging.error(f"循環錯誤: {e}")
            time.sleep(1)
