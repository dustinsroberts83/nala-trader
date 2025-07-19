import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import required packages with error handling
try:
    import yfinance as yf
except ImportError as e:
    st.error(f"Error importing yfinance: {e}")
    st.error("Please make sure yfinance is installed: pip install yfinance")
    st.stop()

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing plotly: {e}")
    st.error("Charts will not be available. Please install plotly: pip install plotly")
    PLOTLY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    
import io
import base64

# Page configuration
st.set_page_config(
    page_title="NALA Trader",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force dark theme and custom styling
st.markdown("""
<style>
    /* Force dark theme */
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    
    /* Main container styling */
    .main .block-container {
        background-color: #0E1117;
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #262730;
    }
    
    /* Header styling */
    .main-header {
        display: flex;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 2px solid #262730;
        margin-bottom: 2rem;
        background-color: #0E1117;
    }
    
    .logo {
        margin-right: 20px;
    }
    
    .title-section h1 {
        margin: 0;
        color: white !important;
        font-size: 2.5rem;
    }
    
    .subtitle {
        color: #A0A0A0 !important;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-metric {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
    }
    
    .danger-metric {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }
    
    /* Force white text on all elements */
    .stMarkdown, .stText, .stMetric, .stDataFrame {
        color: white !important;
    }
    
    /* Input fields dark theme */
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input,
    .stSlider > div > div > div {
        background-color: #262730 !important;
        color: white !important;
        border: 1px solid #404040 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4a9eff 0%, #0066cc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #6bb3ff 0%, #0080ff 100%);
        transform: translateY(-2px);
    }
    
    /* Success/Error message styling */
    .stAlert {
        background-color: #262730 !important;
        border: 1px solid #404040 !important;
        color: white !important;
    }
    
    /* Data table styling */
    .stDataFrame {
        background-color: #1E1E1E !important;
    }
    
    .stDataFrame table {
        background-color: #1E1E1E !important;
        color: white !important;
    }
    
    .stDataFrame th {
        background-color: #262730 !important;
        color: white !important;
    }
    
    .stDataFrame td {
        background-color: #1E1E1E !important;
        color: white !important;
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background-color: #4a9eff !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: #262730;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Hide Streamlit menu and footer for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar for dark theme */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #262730;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4a9eff;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6bb3ff;
    }
</style>
""", unsafe_allow_html=True)

class NalaTrader:
    def __init__(self):
        self.stock_universe = [
            'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NFLX',
            'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'CSCO',
            'PYPL', 'UBER', 'ZM', 'ROKU', 'SHOP', 'SNAP',
            'COST', 'HD', 'WMT', 'DIS', 'NKE', 'SBUX', 'MCD', 'LOW',
            'TGT', 'LULU', 'AMGN', 'GILD', 'MRNA', 'PFE', 'JNJ', 'UNH',
            'MA', 'V', 'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C',
            'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU',
            'XLB', 'XLRE', 'QQQ', 'SPY', 'IWM', 'VTI'
        ]
        
    def safe_float(self, value):
        """Safely convert pandas values to float"""
        try:
            if hasattr(value, 'iloc'):
                return float(value.iloc[-1])
            elif hasattr(value, 'item'):
                return float(value.item())
            else:
                return float(value)
        except:
            return 0.0
    
    def calculate_momentum_score(self, data):
        """Calculate momentum score"""
        if len(data) < 20:
            return 0
            
        try:
            close = data['Close']
            returns_5d = 0
            returns_10d = 0
            returns_20d = 0
            
            if len(close) > 5:
                returns_5d = (self.safe_float(close.iloc[-1]) / self.safe_float(close.iloc[-6]) - 1) * 100
            if len(close) > 10:
                returns_10d = (self.safe_float(close.iloc[-1]) / self.safe_float(close.iloc[-11]) - 1) * 100
            if len(close) > 20:
                returns_20d = (self.safe_float(close.iloc[-1]) / self.safe_float(close.iloc[-21]) - 1) * 100
            
            # Volume momentum
            volume = data['Volume']
            avg_volume_20 = self.safe_float(volume.rolling(20).mean().iloc[-1])
            recent_volume = self.safe_float(volume.iloc[-3:].mean())
            volume_ratio = recent_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            # Volatility filter
            volatility = self.safe_float(close.pct_change().rolling(20).std().iloc[-1]) * np.sqrt(252)
            vol_penalty = max(0, (volatility - 0.3) * 10)
            
            # Price action quality
            price_strength = 0
            current_price = self.safe_float(close.iloc[-1])
            ma_20 = self.safe_float(close.rolling(20).mean().iloc[-1])
            ma_50 = self.safe_float(close.rolling(50).mean().iloc[-1])
            
            if returns_5d > 0 and returns_10d > 0:
                price_strength += 10
            if current_price > ma_20:
                price_strength += 5
            if current_price > ma_50:
                price_strength += 5
                
            # Final momentum score
            momentum_score = (
                returns_5d * 0.4 +
                returns_10d * 0.3 +
                returns_20d * 0.2 +
                min(volume_ratio * 5, 10) +
                price_strength - vol_penalty
            )
            
            return max(0, momentum_score)
            
        except Exception as e:
            return 0
    
    def download_data(self, symbols, start_date, end_date):
        """Download price data for stocks"""
        data_dict = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            try:
                status_text.text(f'Downloading {symbol}... ({i+1}/{len(symbols)})')
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if len(data) > 20:
                    data_dict[symbol] = data
                progress_bar.progress((i + 1) / len(symbols))
            except:
                continue
        
        status_text.text(f'✅ Successfully loaded {len(data_dict)} stocks')
        progress_bar.empty()
        
        return data_dict
    
    def run_backtest(self, initial_capital, weekly_addition, max_positions, 
                     momentum_threshold, profit_target, stop_loss, portfolio_heat_limit,
                     trailing_stop_distance, volume_spike_required, min_hold_days, start_date, end_date):
        """Run the backtest simulation with advanced controls"""
        
        # Initialize variables
        positions = {}
        cash = initial_capital
        trades = []
        daily_values = []
        total_contributed = initial_capital
        max_position_size = 0.25
        max_portfolio_heat = portfolio_heat_limit / 100  # Convert to decimal
        trailing_stop = trailing_stop_distance / 100  # Convert to decimal
        volume_threshold = volume_spike_required / 100  # Convert to decimal
        max_trades_per_day = 2
        
        # Download data
        st.info("🐕 NALA is fetching market data...")
        data_dict = self.download_data(self.stock_universe, start_date, end_date)
        
        if not data_dict:
            st.error("No data available for backtesting")
            return None
        
        # Get trading dates
        sample_data = list(data_dict.values())[0]
        start_date_pd = pd.to_datetime(start_date)
        trading_dates = sample_data.index[sample_data.index >= start_date_pd]
        
        # Run simulation
        st.info("🐕 NALA is analyzing opportunities...")
        simulation_progress = st.progress(0)
        
        trades_today = 0
        week_number = 0
        last_week = None
        
        for idx, current_date in enumerate(trading_dates):
            try:
                # Weekly capital addition
                current_week = current_date.isocalendar()[1]
                if current_week != last_week:
                    week_number += 1
                    if week_number > 1:
                        cash += weekly_addition
                        total_contributed += weekly_addition
                    last_week = current_week
                    trades_today = 0
                
                # Manage existing positions
                to_sell = []
                for symbol, position in positions.items():
                    if symbol in data_dict and current_date in data_dict[symbol].index:
                        current_price = self.safe_float(data_dict[symbol].loc[current_date, 'Close'])
                        entry_price = position['entry_price']
                        
                        # Update highest price for trailing stop
                        if current_price > position['highest_price']:
                            position['highest_price'] = current_price
                        
                        position['days_held'] += 1
                        
                        # Check exit conditions
                        pnl_pct = (current_price / entry_price - 1)
                        trailing_stop_price = position['highest_price'] * (1 - trailing_stop)
                        
                        # Exit conditions
                        if pnl_pct <= -stop_loss:
                            reason = "Stop Loss"
                        elif pnl_pct >= profit_target:
                            reason = "Profit Target"
                        elif current_price <= trailing_stop_price and pnl_pct > 0.05:
                            reason = "Trailing Stop"
                        elif position['days_held'] >= 3:
                            current_data = data_dict[symbol].loc[:current_date]
                            momentum_score = self.calculate_momentum_score(current_data)
                            if momentum_score < momentum_threshold * 0.6:
                                reason = "Momentum Fade"
                            else:
                                reason = None
                        else:
                            reason = None
                        
                        if reason:
                            to_sell.append((symbol, current_price, reason))
                
                # Execute sells
                for symbol, price, reason in to_sell:
                    if symbol in positions:
                        shares = positions[symbol]['shares']
                        proceeds = shares * price
                        entry_price = positions[symbol]['entry_price']
                        pnl_pct = (price / entry_price - 1) * 100
                        
                        cash += proceeds
                        del positions[symbol]
                        
                        trades.append({
                            'date': current_date,
                            'symbol': symbol,
                            'action': 'SELL',
                            'shares': shares,
                            'price': price,
                            'value': proceeds,
                            'reason': reason,
                            'pnl_pct': pnl_pct
                        })
                
                # Look for new opportunities
                portfolio_value = cash
                for symbol, position in positions.items():
                    if symbol in data_dict and current_date in data_dict[symbol].index:
                        current_price = self.safe_float(data_dict[symbol].loc[current_date, 'Close'])
                        portfolio_value += position['shares'] * current_price
                
                portfolio_heat = (portfolio_value - cash) / portfolio_value if portfolio_value > 0 else 0
                
                if (len(positions) < max_positions and 
                    trades_today < max_trades_per_day and 
                    portfolio_heat < max_portfolio_heat):
                    
                    # Find opportunities
                    opportunities = []
                    for symbol, data in data_dict.items():
                        if current_date in data.index:
                            current_data = data.loc[:current_date]
                            if len(current_data) >= 20:
                                current_price = self.safe_float(current_data['Close'].iloc[-1])
                                if current_price >= 10:  # Quality filter
                                    momentum_score = self.calculate_momentum_score(current_data)
                                    if momentum_score >= momentum_threshold:
                                        # Check volume requirement
                                        volume = current_data['Volume']
                                        avg_volume_20 = self.safe_float(volume.rolling(20).mean().iloc[-1])
                                        recent_volume = self.safe_float(volume.iloc[-1])
                                        volume_ratio = recent_volume / avg_volume_20 if avg_volume_20 > 0 else 0
                                        
                                        if volume_ratio >= volume_threshold:
                                            opportunities.append({
                                                'symbol': symbol,
                                                'score': momentum_score,
                                                'price': current_price
                                            })
                    
                    # Sort and execute trades
                    opportunities.sort(key=lambda x: x['score'], reverse=True)
                    
                    for opp in opportunities:
                        if (len(positions) >= max_positions or 
                            trades_today >= max_trades_per_day):
                            break
                            
                        symbol = opp['symbol']
                        price = opp['price']
                        
                        if symbol not in positions:
                            # Calculate position size
                            max_position_value = portfolio_value * max_position_size
                            shares = int(max_position_value / price)
                            cost = shares * price
                            
                            if cost <= cash and shares > 0:
                                cash -= cost
                                positions[symbol] = {
                                    'shares': shares,
                                    'entry_price': price,
                                    'entry_date': current_date,
                                    'highest_price': price,
                                    'days_held': 0
                                }
                                
                                trades.append({
                                    'date': current_date,
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'shares': shares,
                                    'price': price,
                                    'value': cost,
                                    'reason': f"NALA MOMENTUM: {opp['score']:.1f}"
                                })
                                
                                trades_today += 1
                
                # Calculate daily portfolio value
                portfolio_value = cash
                for symbol, position in positions.items():
                    if symbol in data_dict and current_date in data_dict[symbol].index:
                        current_price = self.safe_float(data_dict[symbol].loc[current_date, 'Close'])
                        portfolio_value += position['shares'] * current_price
                
                daily_values.append({
                    'date': current_date,
                    'value': portfolio_value,
                    'cash': cash,
                    'positions': len(positions)
                })
                
                # Update progress
                simulation_progress.progress((idx + 1) / len(trading_dates))
                
            except Exception as e:
                continue
        
        simulation_progress.empty()
        
        return {
            'trades': trades,
            'daily_values': daily_values,
            'final_value': daily_values[-1]['value'] if daily_values else total_contributed,
            'total_contributed': total_contributed,
            'positions': positions,
            'cash': cash
        }

def show_strategy_explanation(max_positions, momentum_threshold, profit_target, stop_loss,
                            portfolio_heat_limit, trailing_stop_distance, volume_spike_required, min_hold_days):
    """Show detailed explanation of NALA's trading logic"""
    
    with st.expander("🧠 NALA's Trading Logic Explained", expanded=True):
        st.markdown("### 🐕 How NALA Will Trade With Your Settings")
        
        # Entry Logic
        st.markdown("#### 🎯 **Entry Logic** - When NALA Buys:")
        st.markdown(f"""
        1. **Scan 60+ Quality Stocks** daily for opportunities
        2. **Momentum Score ≥ {momentum_threshold}** - Combines 5-day, 10-day, and 20-day price momentum
        3. **Volume Confirmation** - Requires {volume_spike_required}% of average volume (filters out weak moves)
        4. **Price Action** - Stock must be above both 20-day and 50-day moving averages
        5. **Portfolio Limits** - Only if we have fewer than {max_positions} positions and less than {portfolio_heat_limit}% invested
        6. **Position Size** - Each position will be ~{100/max_positions:.0f}% of portfolio (max 25%)
        """)
        
        # Exit Logic
        st.markdown("#### 🚪 **Exit Logic** - When NALA Sells:")
        st.markdown(f"""
        1. **Profit Target**: Sell at +{profit_target}% gain 🎉
        2. **Stop Loss**: Sell at -{stop_loss}% loss 🛑  
        3. **Trailing Stop**: Sell if price drops {trailing_stop_distance}% from highest point (locks in profits)
        4. **Momentum Fade**: After {min_hold_days} days, sell if momentum score drops below {momentum_threshold * 0.6:.1f}
        5. **Time Limit**: Sell after 30 days regardless (prevents dead money)
        """)
        
        # Risk Management
        st.markdown("#### 🛡️ **Risk Management**:")
        st.markdown(f"""
        - **Cash Reserve**: Keep {100-portfolio_heat_limit}% in cash as safety buffer
        - **Position Sizing**: Max {100/max_positions:.0f}% per stock prevents concentration risk
        - **Quality Filter**: Only trades stocks above $10 with good liquidity
        - **No Correlation**: Avoids buying too many similar stocks
        """)
        
        # Example Scenario
        st.markdown("#### 📈 **Example Trade Scenario**:")
        example_investment = 1000
        position_size = example_investment * 0.25
        profit_amount = position_size * (profit_target/100)
        loss_amount = position_size * (stop_loss/100)
        
        st.markdown(f"""
        **With ${example_investment:,} portfolio:**
        - NALA finds AAPL with momentum score {momentum_threshold + 5}
        - Buys ${position_size:.0f} worth (~{100/max_positions:.0f}% of portfolio)
        - Sets stop loss at -{stop_loss}% (max loss: ${loss_amount:.0f})
        - Takes profit at +{profit_target}% (profit: ${profit_amount:.0f})
        - Trails stop {trailing_stop_distance}% below highest price
        """)
        
        # Strategy Summary
        st.markdown("#### 🎯 **Strategy Summary**:")
        conservatism = "Conservative" if momentum_threshold > 20 else "Moderate" if momentum_threshold > 10 else "Aggressive"
        risk_level = "Low" if stop_loss < 5 else "Medium" if stop_loss < 10 else "High"
        
        st.info(f"""
        **Trading Style**: {conservatism} momentum following
        **Risk Level**: {risk_level} risk tolerance
        **Expected Trades**: ~{max_positions * 2}-{max_positions * 4} per month
        **Cash Management**: Keep {100-portfolio_heat_limit}% cash buffer
        **Win Rate Target**: 45-65% (higher momentum threshold = higher win rate)
        """)

def create_header():
    """Create the header with NALA's photo"""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        # Try to load NALA's photo
        try:
            # Look for NALA's photo
            import os
            photo_files = ['nala.jpg', 'nala.jpeg', 'nala.png', 'nala.bmp']
            nala_image = None
            
            for filename in photo_files:
                if os.path.exists(filename):
                    nala_image = Image.open(filename)
                    break
            
            if nala_image:
                # Resize and make circular
                size = 120
                nala_image = nala_image.resize((size, size))
                st.image(nala_image, width=size)
            else:
                # Placeholder
                st.markdown("### 🐕")
        except:
            st.markdown("### 🐕")
    
    with col2:
        st.markdown("# NALA Trader")
        st.markdown("### *Normalized Asset Leveraging Algorithm*")
        st.markdown("---")

def main():
    create_header()
    
    # Initialize trader
    trader = NalaTrader()
    
    # Sidebar configuration
    st.sidebar.header("🐕 NALA Configuration")
    
    # Trading mode
    mode = st.sidebar.radio(
        "Trading Mode",
        ["Simulation Mode", "Live Trading Mode"],
        help="Choose between backtesting simulation or live trading"
    )
    
    if mode == "Live Trading Mode":
        st.sidebar.warning("🚧 Live trading coming soon!")
        st.sidebar.info("Please use Simulation Mode for now.")
        return
    
    # Investment settings
    st.sidebar.subheader("Investment Settings")
    
    initial_investment = st.sidebar.number_input(
        "Initial Investment ($)",
        min_value=10,
        max_value=1000000,
        value=100,
        help="Starting capital amount"
    )
    
    weekly_contribution = st.sidebar.number_input(
        "Weekly Contribution ($)",
        min_value=0,
        max_value=10000,
        value=50,
        help="Amount to add each week"
    )
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime(2025, 2, 28),
        help="Backtest start date"
    )
    
    # Strategy settings
    st.sidebar.subheader("NALA Strategy Settings")
    
    max_positions = st.sidebar.slider(
        "Max Positions",
        min_value=1,
        max_value=10,
        value=6,
        help="Maximum number of stocks to hold simultaneously"
    )
    
    momentum_threshold = st.sidebar.slider(
        "Momentum Threshold",
        min_value=5.0,
        max_value=50.0,
        value=15.0,
        step=0.5,
        help="Minimum momentum score required to trigger buy signals"
    )
    
    profit_target = st.sidebar.slider(
        "Profit Target (%)",
        min_value=5,
        max_value=100,
        value=25,
        help="Automatically sell when position reaches this profit percentage"
    )
    
    stop_loss = st.sidebar.slider(
        "Stop Loss (%)",
        min_value=1,
        max_value=20,
        value=7,
        help="Automatically sell when position loses this percentage"
    )
    
    # Advanced strategy controls
    st.sidebar.subheader("Advanced Controls")
    
    portfolio_heat_limit = st.sidebar.slider(
        "Portfolio Heat Limit (%)",
        min_value=25,
        max_value=100,
        value=75,
        step=5,
        help="Maximum percentage of portfolio invested at once (rest stays in cash)"
    )
    
    trailing_stop_distance = st.sidebar.slider(
        "Trailing Stop Distance (%)",
        min_value=5,
        max_value=30,
        value=15,
        step=1,
        help="How far below peak price to set trailing stop loss"
    )
    
    volume_spike_required = st.sidebar.slider(
        "Volume Spike Required (%)",
        min_value=100,
        max_value=300,
        value=120,
        step=10,
        help="Require this % of average volume before buying (120% = 20% above normal)"
    )
    
    min_hold_days = st.sidebar.slider(
        "Minimum Hold Days",
        min_value=1,
        max_value=14,
        value=3,
        help="Minimum days to hold a position before considering exit"
    )
    
    # Strategy explanation button
    if st.sidebar.button("🧠 Explain NALA's Logic", help="See exactly how NALA will trade with your settings"):
        show_strategy_explanation(
            max_positions, momentum_threshold, profit_target, stop_loss,
            portfolio_heat_limit, trailing_stop_distance, volume_spike_required, min_hold_days
        )
    
    # Main content area
    if st.sidebar.button("🐕 Nala Fetch", type="primary", use_container_width=True):
        end_date = datetime.now().date()  # Convert to date object
        
        try:
            with st.spinner("🐕 NALA is working..."):
                results = trader.run_backtest(
                    initial_capital=initial_investment,
                    weekly_addition=weekly_contribution,
                    max_positions=max_positions,
                    momentum_threshold=momentum_threshold,
                    profit_target=profit_target/100,
                    stop_loss=stop_loss/100,
                    portfolio_heat_limit=portfolio_heat_limit,
                    trailing_stop_distance=trailing_stop_distance,
                    volume_spike_required=volume_spike_required,
                    min_hold_days=min_hold_days,
                    start_date=start_date,
                    end_date=end_date
                )
            
            if results:
                # Store results in session state
                st.session_state['results'] = results
                st.success("🎾 Good dog, NALA! Trading analysis complete!")
            else:
                st.error("❌ NALA couldn't complete the analysis. Please try again.")
                
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.error("Please check your settings and try again.")
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Calculate metrics
        final_value = results['final_value']
        total_contributed = results['total_contributed']
        profit = final_value - total_contributed
        roi = (profit / total_contributed) * 100
        
        trades = results['trades']
        winning_trades = [t for t in trades if t.get('pnl_pct', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl_pct', 0) < 0]
        completed_trades = [t for t in trades if 'pnl_pct' in t]
        
        win_rate = len(winning_trades) / len(completed_trades) * 100 if completed_trades else 0
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
        best_trade = max([t['pnl_pct'] for t in completed_trades], default=0)
        worst_trade = min([t['pnl_pct'] for t in completed_trades], default=0)
        
        # Results dashboard
        st.header("🐕 NALA Trading Results")
        
        # Key metrics with total investment
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Value", f"${final_value:,.2f}", f"${profit:+,.2f}")
        
        with col2:
            st.metric("Total Invested", f"${total_contributed:,.2f}", f"Contributed")
        
        with col3:
            st.metric("ROI", f"{roi:.1f}%", f"{roi:.1f}%")
        
        with col4:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Performance chart
        if results['daily_values'] and PLOTLY_AVAILABLE:
            df_performance = pd.DataFrame(results['daily_values'])
            df_performance['date'] = pd.to_datetime(df_performance['date'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_performance['date'],
                y=df_performance['value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#4a9eff', width=3)
            ))
            
            fig.update_layout(
                title="NALA Portfolio Performance",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        elif results['daily_values']:
            # Fallback to Streamlit's built-in chart if plotly not available
            df_performance = pd.DataFrame(results['daily_values'])
            df_performance['date'] = pd.to_datetime(df_performance['date'])
            df_performance = df_performance.set_index('date')
            st.line_chart(df_performance['value'])
        
        # Detailed metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Metrics")
            
            metrics_data = {
                "Metric": ["Best Trade", "Worst Trade", "Avg Win", "Avg Loss", "Available Cash"],
                "Value": [f"+{best_trade:.2f}%", f"{worst_trade:.2f}%", f"+{avg_win:.2f}%", 
                         f"{avg_loss:.2f}%", f"${results['cash']:,.2f}"]
            }
            
            st.dataframe(pd.DataFrame(metrics_data), hide_index=True)
        
        with col2:
            st.subheader("Current Positions")
            
            if results['positions']:
                positions_data = []
                for symbol, pos in results['positions'].items():
                    positions_data.append({
                        "Symbol": symbol,
                        "Shares": pos['shares'],
                        "Entry Price": f"${pos['entry_price']:.2f}",
                        "Days Held": pos['days_held']
                    })
                
                st.dataframe(pd.DataFrame(positions_data), hide_index=True)
            else:
                st.info("No open positions")
        
        # Recent trades
        st.subheader("Recent Trades")
        
        if trades:
            recent_trades = trades[-10:]  # Last 10 trades
            trades_data = []
            
            for trade in reversed(recent_trades):
                trades_data.append({
                    "Date": trade['date'].strftime('%Y-%m-%d'),
                    "Action": trade['action'],
                    "Symbol": trade['symbol'],
                    "Shares": trade['shares'],
                    "Price": f"${trade['price']:.2f}",
                    "P&L": f"{trade.get('pnl_pct', 0):.2f}%" if trade['action'] == 'SELL' else "-",
                    "Reason": trade.get('reason', '')
                })
            
            st.dataframe(pd.DataFrame(trades_data), hide_index=True)
        else:
            st.info("No trades executed yet")
        
        # Top winners
        if winning_trades:
            st.subheader("🏆 Top Winners")
            
            top_winners = sorted(winning_trades, key=lambda x: x['pnl_pct'], reverse=True)[:5]
            winners_data = []
            
            for trade in top_winners:
                winners_data.append({
                    "Symbol": trade['symbol'],
                    "Profit": f"+{trade['pnl_pct']:.2f}%",
                    "Date": trade['date'].strftime('%Y-%m-%d')
                })
            
            st.dataframe(pd.DataFrame(winners_data), hide_index=True)

if __name__ == "__main__":
    main()
