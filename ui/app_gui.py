import tkinter as tk
from tkinter import ttk, scrolledtext
import logging
import threading
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Log handler
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────
class TradingApp(tk.Tk):
    # ── Catppuccin Mocha palette ──────────────────────────────────────────────
    BG_DARK   = "#1E1E2E"
    BG_PANEL  = "#181825"
    BG_CARD   = "#313244"
    FG_MAIN   = "#CDD6F4"
    FG_SUB    = "#A6ADC8"
    ACCENT    = "#89B4FA"
    GREEN     = "#A6E3A1"
    RED       = "#F38BA8"
    YELLOW    = "#F9E2AF"
    PEACH     = "#FAB387"
    TEAL      = "#94E2D5"
    MAUVE     = "#CBA6F7"
    SEPARATOR = "#45475A"

    def __init__(self, engine):
        super().__init__()
        self.engine  = engine
        self.running = False

        self.title("⚔️  Iron Wall  |  AI 黃金策略中心")
        self.geometry("1100x720")
        self.minsize(900, 620)
        self.configure(bg=self.BG_DARK)

        # runtime counters (updated from strategy)
        self._wins   = 0
        self._losses = 0

        self._setup_styles()
        self._build_ui()
        self._setup_logging()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ─── Styles ───────────────────────────────────────────────────────────────
    def _setup_styles(self):
        s = ttk.Style()
        try: s.theme_use('clam')
        except Exception: pass

        s.configure(".",
            background=self.BG_DARK, foreground=self.FG_MAIN,
            font=('Segoe UI', 10))
        s.configure("TFrame",      background=self.BG_DARK)
        s.configure("Card.TFrame", background=self.BG_CARD)
        s.configure("TLabel",      background=self.BG_DARK, foreground=self.FG_MAIN)
        s.configure("Card.TLabel", background=self.BG_CARD, foreground=self.FG_MAIN)

        s.configure("TButton",
            font=('Segoe UI', 10, 'bold'), borderwidth=0, padding=(10, 7),
            background=self.BG_CARD, foreground=self.FG_MAIN)
        s.map("TButton",
            background=[("active", self.ACCENT), ("disabled", self.SEPARATOR)],
            foreground=[("active", self.BG_DARK), ("disabled", "#6C7086")])

        s.configure("TLabelframe",
            background=self.BG_DARK, bordercolor=self.SEPARATOR, borderwidth=1)
        s.configure("TLabelframe.Label",
            background=self.BG_DARK, foreground=self.ACCENT,
            font=('Segoe UI', 11, 'bold'))

        s.configure("TCheckbutton",
            background=self.BG_DARK, foreground=self.FG_MAIN,
            font=('Segoe UI', 10, 'bold'))
        s.map("TCheckbutton", background=[("active", self.BG_DARK)])

        # Thin horizontal separator
        s.configure("Sep.TFrame", background=self.SEPARATOR)

    # ─── Build UI ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top toolbar ───────────────────────────────────────────────────────
        toolbar = ttk.Frame(self, padding=(16, 10, 16, 10))
        toolbar.pack(fill=tk.X)

        # Title label
        ttk.Label(toolbar, text="⚔️  IRON WALL",
            font=('Segoe UI', 14, 'bold'), foreground=self.ACCENT
        ).pack(side=tk.LEFT, padx=(0, 24))

        # Buttons
        self.btn_connect = ttk.Button(toolbar, text="🔗 連線 MT5", command=self.on_connect)
        self.btn_connect.pack(side=tk.LEFT, padx=4)

        self.btn_start = ttk.Button(toolbar, text="▶ 啟動 EA", command=self.on_start, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=4)

        self.btn_stop = ttk.Button(toolbar, text="⏹ 停止", command=self.on_stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        self.btn_retrain = ttk.Button(toolbar, text="🔄 重訓模型", command=self.on_retrain)
        self.btn_retrain.pack(side=tk.LEFT, padx=4)

        # Lot size
        ttk.Label(toolbar, text="手數:", foreground=self.FG_SUB).pack(side=tk.LEFT, padx=(20, 4))
        self.ent_lots = self._entry(toolbar, width=6, color=self.GREEN)
        self.ent_lots.insert(0, str(getattr(self.engine.strategies[0], 'lot_size', 0.1)) if self.engine.strategies else "0.1")
        self.ent_lots.pack(side=tk.LEFT, padx=(0, 4))
        self.ent_lots.bind('<Return>',   lambda e: self.update_strategy_settings(silent=True))
        self.ent_lots.bind('<FocusOut>', lambda e: self.update_strategy_settings(silent=True))

        # BB slope limit
        ttk.Label(toolbar, text="BB斜率限:", foreground=self.FG_SUB).pack(side=tk.LEFT, padx=(16, 4))
        self.ent_bb_slope = self._entry(toolbar, width=5, color=self.YELLOW)
        self.ent_bb_slope.insert(0, "3.0")
        self.ent_bb_slope.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(toolbar, text="pts", foreground=self.FG_SUB,
            font=('Segoe UI', 8)).pack(side=tk.LEFT)
        self.ent_bb_slope.bind('<Return>',   lambda e: self.update_strategy_settings(silent=True))
        self.ent_bb_slope.bind('<FocusOut>', lambda e: self.update_strategy_settings(silent=True))

        # Status
        self.lbl_status = tk.Label(toolbar,
            text="● 未連線", font=('Segoe UI', 10, 'bold'),
            fg=self.RED, bg=self.BG_DARK)
        self.lbl_status.pack(side=tk.RIGHT, padx=8)

        # ── Thin separator ────────────────────────────────────────────────────
        ttk.Frame(self, style="Sep.TFrame", height=1).pack(fill=tk.X)

        # ── Dashboard row ─────────────────────────────────────────────────────
        dash = ttk.Frame(self, padding=(16, 10))
        dash.pack(fill=tk.X)
        dash.columnconfigure((0, 1, 2, 3), weight=1, uniform="col")

        # Card 1 — Account
        c1 = self._card(dash, "💰 帳戶")
        c1.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.lbl_balance    = self._card_val(c1, "餘額", "$0.00",    self.GREEN)
        self.lbl_equity     = self._card_val(c1, "淨值", "$0.00",    self.GREEN)
        self.lbl_daily_pnl  = self._card_val(c1, "今日損益", "$0.00", self.FG_MAIN)

        # Card 2 — Position
        c2 = self._card(dash, "📊 持倉")
        c2.grid(row=0, column=1, sticky="nsew", padx=3)
        self.lbl_positions  = self._card_val(c2, "持倉筆數", "0 筆",  self.FG_MAIN)
        self.lbl_profit     = self._card_val(c2, "未實現",   "$0.00", self.FG_MAIN)
        self.lbl_win_loss   = self._card_val(c2, "今日 W/L", "0W / 0L", self.FG_MAIN)

        # Card 3 — AI Signal
        c3 = self._card(dash, "🧠 AI 信心率")
        c3.grid(row=0, column=2, sticky="nsew", padx=3)
        self.lbl_conf_buy   = self._card_val(c3, "多單信心", "--.-%",  self.FG_MAIN)
        self.lbl_conf_sell  = self._card_val(c3, "空單信心", "--.-%",  self.FG_MAIN)
        self.lbl_bb_slope   = self._card_val(c3, "BB斜率",   "-- pts/bar", self.FG_MAIN)

        # Card 4 — PPT Status
        c4 = self._card(dash, "🔍 PPT 階段")
        c4.grid(row=0, column=3, sticky="nsew", padx=(6, 0))
        self.lbl_ppt_stage  = self._card_val(c4, "當前階段",   "無訊號",   self.YELLOW)
        self.lbl_ppt_dir    = self._card_val(c4, "訊號方向",   "--",       self.FG_MAIN)
        self.lbl_filter     = self._card_val(c4, "過濾狀態",   "✅ 正常",   self.GREEN)

        # ── Dashboard row 2 (Indicators) ──────────────────────────────────────
        dash2 = ttk.Frame(self, padding=(16, 0, 16, 10))
        dash2.pack(fill=tk.X)
        dash2.columnconfigure((0, 1, 2, 3), weight=1, uniform="col")

        # Card 5 — Live Indicators
        c5 = self._card(dash2, "📈 盤中即時指標 (與 MT5 圖表比對)")
        c5.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=0)
        
        # 建立內部 frame 來橫向排列指標
        ind_frame = tk.Frame(c5, bg=self.BG_CARD)
        ind_frame.pack(fill=tk.X, padx=10, pady=8)
        
        self.lbl_ind_ema12  = tk.Label(ind_frame, text="EMA12: 0.00", font=('Segoe UI', 10, 'bold'), fg=self.FG_MAIN, bg=self.BG_CARD)
        self.lbl_ind_ema12.pack(side=tk.LEFT, padx=(0, 30))
        
        self.lbl_ind_bb     = tk.Label(ind_frame, text="BB(21, 2.1): 0.00 / 0.00", font=('Segoe UI', 10, 'bold'), fg=self.FG_MAIN, bg=self.BG_CARD)
        self.lbl_ind_bb.pack(side=tk.LEFT, padx=30)
        
        self.lbl_ind_cs     = tk.Label(ind_frame, text="盤整分數: 0.00", font=('Segoe UI', 10, 'bold'), fg=self.FG_MAIN, bg=self.BG_CARD)
        self.lbl_ind_cs.pack(side=tk.LEFT, padx=30)

        # ── Separator ─────────────────────────────────────────────────────────
        ttk.Frame(self, style="Sep.TFrame", height=1).pack(fill=tk.X, padx=16)

        # ── Status bar ────────────────────────────────────────────────────────
        status_bar = ttk.Frame(self, padding=(16, 6))
        status_bar.pack(fill=tk.X)

        self.lbl_ea_status = tk.Label(status_bar,
            text="EA 狀態: ⏸ 已停止",
            font=('Consolas', 10), fg=self.YELLOW, bg=self.BG_DARK, anchor='w')
        self.lbl_ea_status.pack(side=tk.LEFT)

        self.lbl_last_signal = tk.Label(status_bar,
            text="上次訊號: --",
            font=('Consolas', 10), fg=self.FG_SUB, bg=self.BG_DARK, anchor='w')
        self.lbl_last_signal.pack(side=tk.LEFT, padx=32)

        self.lbl_tp_sl = tk.Label(status_bar,
            text="TP: 5pt  SL: 5pt  門檻: 65%",
            font=('Consolas', 10), fg=self.TEAL, bg=self.BG_DARK)
        self.lbl_tp_sl.pack(side=tk.RIGHT)

        # ── Log area ──────────────────────────────────────────────────────────
        ttk.Frame(self, style="Sep.TFrame", height=1).pack(fill=tk.X, padx=16)

        log_frame = ttk.LabelFrame(self, text=" 📋 系統日誌 ", padding=(12, 8))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(8, 12))

        self.log_area = scrolledtext.ScrolledText(
            log_frame, state='disabled', height=12,
            font=('Consolas', 10),
            bg=self.BG_PANEL, fg=self.FG_SUB,
            insertbackground="white",
            borderwidth=0, highlightthickness=0,
            selectbackground=self.ACCENT, selectforeground=self.BG_DARK
        )
        self.log_area.pack(fill=tk.BOTH, expand=True)

    # ─── Helper: create a card frame ──────────────────────────────────────────
    def _card(self, parent, title: str) -> ttk.Frame:
        outer = tk.Frame(parent, bg=self.BG_CARD, bd=0)
        tk.Label(outer, text=title,
            font=('Segoe UI', 9, 'bold'),
            fg=self.ACCENT, bg=self.BG_CARD
        ).pack(anchor='w', padx=10, pady=(8, 2))
        # thin inner separator
        tk.Frame(outer, bg=self.SEPARATOR, height=1).pack(fill=tk.X, padx=8)
        return outer

    def _card_val(self, card: tk.Frame, label: str, value: str, color: str):
        """Adds a label-value row inside a card and returns the value Label."""
        row = tk.Frame(card, bg=self.BG_CARD)
        row.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(row, text=label + ":",
            font=('Segoe UI', 9), fg=self.FG_SUB, bg=self.BG_CARD, width=8, anchor='w'
        ).pack(side=tk.LEFT)
        val_lbl = tk.Label(row, text=value,
            font=('Consolas', 11, 'bold'), fg=color, bg=self.BG_CARD, anchor='w')
        val_lbl.pack(side=tk.LEFT)
        return val_lbl

    def _entry(self, parent, width: int = 7, color: str = None) -> tk.Entry:
        return tk.Entry(parent, width=width,
            font=('Consolas', 11, 'bold'),
            bg=self.BG_PANEL, fg=color or self.FG_MAIN,
            insertbackground="white",
            highlightthickness=1, highlightbackground=self.SEPARATOR,
            highlightcolor=self.ACCENT, relief=tk.FLAT)

    # ─── Logging ──────────────────────────────────────────────────────────────
    def _setup_logging(self):
        handler = TextHandler(self.log_area)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
        self._anim = ['|', '/', '-', '\\']
        self._anim_i = 0

    # ─── Settings sync ────────────────────────────────────────────────────────
    def update_strategy_settings(self, silent=False):
        try:
            lots = float(self.ent_lots.get())
            if lots <= 0: raise ValueError
            for s in self.engine.strategies:
                s.lot_size = lots
            # Update TP/SL display
            tp = getattr(self.engine.strategies[0], 'FIXED_TP', 5.0) if self.engine.strategies else 5.0
            sl = getattr(self.engine.strategies[0], 'FIXED_SL', 5.0) if self.engine.strategies else 5.0
            thr = getattr(self.engine.strategies[0], 'THRESHOLD', 0.65) if self.engine.strategies else 0.65
            self.lbl_tp_sl.config(text=f"TP: {tp:.0f}pt  SL: {sl:.0f}pt  門檻: {thr*100:.0f}%")
            if not silent: logging.info(f"✅ 手數已更新: {lots}")
        except ValueError:
            if not silent: logging.error("❌ 手數格式錯誤")

        try:
            bb_limit = float(self.ent_bb_slope.get())
            if bb_limit <= 0: raise ValueError
            for s in self.engine.strategies:
                if hasattr(s, 'BB_SLOPE_LIMIT'):
                    s.BB_SLOPE_LIMIT = bb_limit
            if not silent: logging.info(f"✅ BB斜率限制: ±{bb_limit} pts/bar")
        except (ValueError, AttributeError):
            pass

    # ─── Connection ───────────────────────────────────────────────────────────
    def on_connect(self):
        self.lbl_status.config(text="● 連線中...", fg=self.YELLOW)
        threading.Thread(target=self._connect_task, daemon=True).start()

    def _connect_task(self):
        success = self.engine.connector.connect()
        self.after(0, self._on_connect_result, success)

    def _on_connect_result(self, success):
        if success:
            self.lbl_status.config(text="● 已連線", fg=self.GREEN)
            self.btn_start.config(state=tk.NORMAL)
            self.update_account_info()
        else:
            self.lbl_status.config(text="● 連線失敗", fg=self.RED)

    # ─── EA control ───────────────────────────────────────────────────────────
    def on_start(self):
        if self.engine.start():
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.running = True
            self.update_strategy_settings(silent=True)
            threading.Thread(target=self._bot_loop, daemon=True).start()

    def on_stop(self):
        if not self.running: return
        self.engine.stop()
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_ea_status.config(text="EA 狀態: ⏸ 已停止", fg=self.YELLOW)

    def on_retrain(self):
        self.btn_retrain.config(state=tk.DISABLED, text="⏳ 重訓中...")
        was_running = self.running
        if was_running:
            self.on_stop()
            logging.info("EA 已暫停，開始 AI 模型重訓流程...")
        else:
            logging.info("開始 AI 模型重訓流程，這可能需要數分鐘...")
        threading.Thread(target=self._retrain_task, args=(was_running,), daemon=True).start()

    def _retrain_task(self, restart_after=False):
        try:
            from ml_engine.evolve_brain import evolve
            evolve()
            for strat in self.engine.strategies:
                if hasattr(strat, 'reload_models'):
                    strat.reload_models()
            self.after(0, lambda: logging.info("✅ 模型重訓並重載完成！最新大腦已上線。"))
            if restart_after:
                self.after(500,  self.on_connect)
                self.after(3000, self.on_start)
        except Exception as e:
            err = str(e)
            self.after(0, lambda m=err: logging.error(f"重訓失敗: {m}"))
        finally:
            self.after(0, lambda: self.btn_retrain.config(state=tk.NORMAL, text="🔄 重訓模型"))

    def on_closing(self):
        self.running = False
        if self.engine.is_running:         self.engine.stop()
        if self.engine.connector.connected: self.engine.connector.disconnect()
        self.destroy()

    # ─── Dashboard update ─────────────────────────────────────────────────────
    def update_account_info(self):
        if not self.engine.connector.connected:
            return
        try:
            # ── Account ──────────────────────────────────────────────────────
            acc = self.engine.connector.get_account_info()
            if acc:
                bal = acc.get('balance', 0)
                equ = acc.get('equity', 0)
                self.lbl_balance.config(text=f"${bal:,.2f}", fg=self.GREEN)
                self.lbl_equity.config(
                    text=f"${equ:,.2f}",
                    fg=self.GREEN if equ >= bal else self.RED)

                realized, floating, total = self.engine.daily_stats
                pnl_color = self.GREEN if total > 0 else (self.RED if total < 0 else self.FG_MAIN)
                self.lbl_daily_pnl.config(text=f"${total:+,.2f}", fg=pnl_color)

            # ── Position ─────────────────────────────────────────────────────
            count, profit = self.engine.connector.get_positions_summary()
            self.lbl_positions.config(text=f"{count} 筆")
            p_color = self.GREEN if profit > 0 else (self.RED if profit < 0 else self.FG_MAIN)
            self.lbl_profit.config(text=f"${profit:+,.2f}", fg=p_color)

            # Win/Loss counter from strategy
            if self.engine.strategies:
                strat = self.engine.strategies[0]
                w = getattr(strat, 'today_wins',   self._wins)
                l = getattr(strat, 'today_losses', self._losses)
                wl_color = self.GREEN if w > l else (self.RED if l > w else self.FG_MAIN)
                self.lbl_win_loss.config(text=f"{w}W / {l}L", fg=wl_color)

            # ── AI + PPT ─────────────────────────────────────────────────────
            if self.engine.strategies:
                strat = self.engine.strategies[0]

                # Confidence
                cb    = getattr(strat, 'conf_buy',  0.0) * 100
                cs    = getattr(strat, 'conf_sell', 0.0) * 100
                slope = getattr(strat, 'bb_slope',  0.0)
                limit = getattr(strat, 'BB_SLOPE_LIMIT', 3.0)

                buy_color  = self.GREEN  if cb >= 65 else (self.YELLOW if cb >= 50 else self.FG_MAIN)
                sell_color = self.GREEN  if cs >= 65 else (self.YELLOW if cs >= 50 else self.FG_MAIN)
                self.lbl_conf_buy.config(text=f"{cb:5.1f}%",  fg=buy_color)
                self.lbl_conf_sell.config(text=f"{cs:5.1f}%", fg=sell_color)

                pct = min(abs(slope) / limit * 100, 100) if limit else 0
                slope_color = self.RED if pct >= 100 else (self.YELLOW if pct >= 70 else self.TEAL)
                self.lbl_bb_slope.config(text=f"{slope:+.2f} pts", fg=slope_color)

                # PPT Stage
                stage     = getattr(strat, 'current_stage', 0)
                sig_dir   = getattr(strat, 'signal_dir',    0)
                blocked   = getattr(strat, 'filter_blocked', False)

                stage_names = {42: "4-2 階", 41: "4-1 階", 3: "3 階", 2: "2 階", 0: "無訊號"}
                stage_str = stage_names.get(stage, f"Stage {stage}")
                stage_color = {42: self.MAUVE, 41: self.PEACH,
                               3:  self.TEAL,   2: self.ACCENT}.get(stage, self.YELLOW)

                self.lbl_ppt_stage.config(
                    text=stage_str,
                    fg=stage_color if stage != 0 else self.YELLOW)

                dir_text  = "▲ 多單" if sig_dir == 1 else ("▼ 空單" if sig_dir == -1 else "--")
                dir_color = self.GREEN if sig_dir == 1 else (self.RED if sig_dir == -1 else self.FG_MAIN)
                self.lbl_ppt_dir.config(text=dir_text, fg=dir_color)

                if blocked:
                    if stage == 0:
                        self.lbl_filter.config(text="⚡ 無 PPT 訊號", fg=self.RED)
                    else:
                        self.lbl_filter.config(
                            text=f"⚡ 波動封鎖 ({abs(slope):.1f}>{limit:.1f})",
                            fg=self.RED)
                elif pct >= 70:
                    self.lbl_filter.config(
                        text=f"⚠️ 接近限制 ({pct:.0f}%)",
                        fg=self.YELLOW)
                else:
                    self.lbl_filter.config(text="✅ 正常運行", fg=self.GREEN)

                # Last signal from strategy log
                last_sig = getattr(strat, 'last_signal_time', None)
                if last_sig:
                    self.lbl_last_signal.config(text=f"上次訊號: {last_sig}")

                # Live Indicators
                inds = getattr(strat, 'live_indicators', None)
                if inds:
                    self.lbl_ind_ema12.config(text=f"EMA(12): {inds['ema12']:.2f}")
                    self.lbl_ind_bb.config(text=f"BB(21): {inds['bb_h']:.2f} / {inds['bb_l']:.2f}")
                    cs_val = inds['consolidation_score']
                    cs_color = self.RED if cs_val > 0.4 else self.GREEN
                    self.lbl_ind_cs.config(text=f"盤整分數: {cs_val:.2f}", fg=cs_color)

        except Exception as e:
            logger.error(f"UI更新錯誤: {e}")

        # ── EA animated status ────────────────────────────────────────────────
        if self.running:
            self._anim_i = (self._anim_i + 1) % len(self._anim)
            f = self._anim[self._anim_i]
            self.lbl_ea_status.config(
                text=f"EA 狀態: ▶ 掃描中 {f}",
                fg=self.GREEN)

    # ─── Bot loop ─────────────────────────────────────────────────────────────
    def _bot_loop(self):
        while self.running:
            try:
                self.engine.on_tick()
                if self.winfo_exists():
                    self.after(0, self.update_account_info)
            except Exception as e:
                logging.error(f"循環錯誤: {e}")
            time.sleep(1)

    # backward compat alias
    def bot_loop(self):
        self._bot_loop()
