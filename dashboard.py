# =============================================================
# dashboard.py — Streamlit Dashboard for Trader Sentiment Analysis
# =============================================================
# Run with: streamlit run dashboard.py
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Trader Sentiment Analysis ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────
# DATA LOADING (cached for performance)
# ─────────────────────────────────────────────────
@st.cache_data
def load_and_process_data():
    """Load and process all datasets. Cached to avoid reloading."""
    
    # Load raw data
    fg_df = pd.read_csv("fear_greed_index.csv")
    trade_df = pd.read_csv("historical_data.csv")
    
    # Process Fear & Greed
    fg_df['date'] = pd.to_datetime(fg_df['date'])
    fg_df['value'] = fg_df['value'].astype(int)
    
    # Process trades
    if 'Timestamp IST' in trade_df.columns:
        trade_df['date'] = pd.to_datetime(
            trade_df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce'
        ).dt.normalize()
    elif 'Timestamp' in trade_df.columns:
        sample_val = trade_df['Timestamp'].iloc[0]
        unit = 'ms' if sample_val > 1e12 else 's'
        trade_df['date'] = pd.to_datetime(trade_df['Timestamp'], unit=unit).dt.normalize()
    
    trade_df = trade_df.dropna(subset=['date']).drop_duplicates().reset_index(drop=True)
    
    # Align dates
    date_min = max(fg_df['date'].min(), trade_df['date'].min())
    date_max = min(fg_df['date'].max(), trade_df['date'].max())
    fg_aligned = fg_df[(fg_df['date'] >= date_min) & (fg_df['date'] <= date_max)].copy()
    trade_aligned = trade_df[(trade_df['date'] >= date_min) & (trade_df['date'] <= date_max)].copy()
    
    # Column mappings
    PNL_COL = 'Closed PnL'
    SIZE_USD_COL = 'Size USD'
    SIDE_COL = 'Side'
    ACCOUNT_COL = 'Account'
    COIN_COL = 'Coin'
    
    # Numeric conversions
    for col in [PNL_COL, SIZE_USD_COL, 'Execution Price', 'Size Tokens']:
        if col in trade_aligned.columns:
            trade_aligned[col] = pd.to_numeric(trade_aligned[col], errors='coerce')
    
    # Derived columns
    if SIZE_USD_COL in trade_aligned.columns:
        trade_aligned['notional_usd'] = trade_aligned[SIZE_USD_COL].abs()
    
    trade_aligned['is_win'] = (trade_aligned[PNL_COL] > 0).astype(int)
    
    side_map = {'BUY': 'Long', 'SELL': 'Short', 'Buy': 'Long', 'Sell': 'Short'}
    trade_aligned['side_label'] = trade_aligned[SIDE_COL].map(side_map).fillna(trade_aligned[SIDE_COL])
    
    trade_aligned = trade_aligned.dropna(subset=['notional_usd', PNL_COL]).reset_index(drop=True)
    
    # Daily metrics
    long_count = trade_aligned[trade_aligned['side_label'] == 'Long'].groupby('date').size()
    short_count = trade_aligned[trade_aligned['side_label'] == 'Short'].groupby('date').size()
    
    daily_metrics = trade_aligned.groupby('date').agg(
        total_pnl=('Closed PnL', 'sum'),
        mean_pnl=('Closed PnL', 'mean'),
        num_trades=('Closed PnL', 'count'),
        num_winners=('is_win', 'sum'),
        avg_trade_size=('notional_usd', 'mean'),
        total_volume=('notional_usd', 'sum'),
        unique_traders=('Account', 'nunique'),
    ).reset_index()
    
    daily_metrics['win_rate'] = (daily_metrics['num_winners'] / daily_metrics['num_trades'] * 100).round(2)
    daily_metrics['long_count'] = daily_metrics['date'].map(long_count).fillna(0).astype(int)
    daily_metrics['short_count'] = daily_metrics['date'].map(short_count).fillna(0).astype(int)
    daily_metrics['long_short_ratio'] = (
        daily_metrics['long_count'] / daily_metrics['short_count'].replace(0, np.nan)
    ).round(4)
    
    # Merge with FG
    fg_clean = fg_aligned[['date', 'value', 'classification']].rename(
        columns={'value': 'fg_value', 'classification': 'fg_class'}
    )
    merged = daily_metrics.merge(fg_clean, on='date', how='inner')
    
    # Sentiment classification
    def classify_sentiment(value):
        if value <= 25: return 'Extreme Fear'
        elif value <= 45: return 'Fear'
        elif value <= 55: return 'Neutral'
        elif value <= 75: return 'Greed'
        else: return 'Extreme Greed'
    
    merged['sentiment'] = merged['fg_value'].apply(classify_sentiment)
    
    # Trader profiles
    trade_with_sent = trade_aligned.merge(merged[['date', 'fg_value', 'sentiment']], on='date', how='inner')
    
    trader_profiles = trade_with_sent.groupby('Account').agg(
        total_trades=('Closed PnL', 'count'),
        total_pnl=('Closed PnL', 'sum'),
        win_rate=('is_win', 'mean'),
        avg_trade_size=('notional_usd', 'mean'),
        active_days=('date', 'nunique'),
        num_coins=('Coin', 'nunique'),
    ).reset_index()
    trader_profiles['win_rate'] = (trader_profiles['win_rate'] * 100).round(2)
    trader_profiles['trades_per_day'] = (trader_profiles['total_trades'] / trader_profiles['active_days']).round(1)
    
    return merged, trade_aligned, trader_profiles, trade_with_sent


# Load data
merged_df, trade_aligned, trader_profiles, trade_with_sent = load_and_process_data()

# ─────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────
st.sidebar.title("🎛️ Controls")

# Date range filter
date_range = st.sidebar.date_input(
    "Date Range",
    value=(merged_df['date'].min().date(), merged_df['date'].max().date()),
    min_value=merged_df['date'].min().date(),
    max_value=merged_df['date'].max().date()
)

if len(date_range) == 2:
    mask = (merged_df['date'].dt.date >= date_range[0]) & (merged_df['date'].dt.date <= date_range[1])
    filtered_df = merged_df[mask].copy()
else:
    filtered_df = merged_df.copy()

# Sentiment filter
sentiment_options = ['All'] + ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
selected_sentiment = st.sidebar.multiselect(
    "Sentiment Filter",
    options=sentiment_options,
    default=['All']
)

if 'All' not in selected_sentiment and selected_sentiment:
    filtered_df = filtered_df[filtered_df['sentiment'].isin(selected_sentiment)]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing:** {len(filtered_df)} days")
st.sidebar.markdown(f"**Total trades:** {trade_aligned.shape[0]:,}")
st.sidebar.markdown(f"**Unique traders:** {trader_profiles.shape[0]:,}")

# ─────────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────────
st.title("Trader Sentiment Analysis Dashboard 📈")
st.markdown("*Analyzing how Fear & Greed affects trader behavior on HyperLiquid*")

tab1, tab2, tab3, tab4 = st.tabs([
    " Overview", " Sentiment Impact", " Trader Segments", " Predictions"
])

# ─────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────
with tab1:
    st.header("Market Overview")
    
    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Days", f"{len(filtered_df)}")
    with col2:
        avg_fg = filtered_df['fg_value'].mean()
        st.metric("Avg FG Index", f"{avg_fg:.0f}")
    with col3:
        total_pnl = filtered_df['total_pnl'].sum()
        st.metric("Total PnL", f"${total_pnl:,.0f}")
    with col4:
        avg_wr = filtered_df['win_rate'].mean()
        st.metric("Avg Win Rate", f"{avg_wr:.1f}%")
    with col5:
        avg_trades = filtered_df['num_trades'].mean()
        st.metric("Avg Trades/Day", f"{avg_trades:,.0f}")
    
    st.markdown("---")
    
    # Time series: FG Index + PnL
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=filtered_df['date'], y=filtered_df['fg_value'],
                   name='Fear & Greed Index', fill='tozeroy',
                   line=dict(color='#1976d2', width=1), fillcolor='rgba(25,118,210,0.15)'),
        secondary_y=False
    )
    
    pnl_roll = filtered_df.set_index('date')['total_pnl'].rolling(7, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(x=pnl_roll.index, y=pnl_roll.values,
                   name='Daily PnL (7d avg)', line=dict(color='#d32f2f', width=1.5)),
        secondary_y=True
    )
    
    # Sentiment zones
    fig.add_hrect(y0=0, y1=25, fillcolor="red", opacity=0.05, line_width=0, secondary_y=False)
    fig.add_hrect(y0=75, y1=100, fillcolor="green", opacity=0.05, line_width=0, secondary_y=False)
    
    fig.update_layout(
        title="Fear & Greed Index vs Trading Performance",
        xaxis_title="Date", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    fig.update_yaxes(title_text="FG Index", secondary_y=False)
    fig.update_yaxes(title_text="PnL ($)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment distribution
    col1, col2 = st.columns(2)
    
    with col1:
        sent_counts = filtered_df['sentiment'].value_counts()
        colors_map = {
            'Extreme Fear': '#d32f2f', 'Fear': '#ff7043', 'Neutral': '#ffd54f',
            'Greed': '#66bb6a', 'Extreme Greed': '#2e7d32'
        }
        fig_pie = px.pie(
            values=sent_counts.values, names=sent_counts.index,
            title="Sentiment Distribution",
            color=sent_counts.index,
            color_discrete_map=colors_map
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_vol = px.bar(
            filtered_df, x='date', y='total_volume',
            color='sentiment', title="Daily Volume by Sentiment",
            color_discrete_map=colors_map
        )
        fig_vol.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig_vol, use_container_width=True)


# ─────────────────────────────────────────────────
# TAB 2: SENTIMENT IMPACT
# ─────────────────────────────────────────────────
with tab2:
    st.header(" Sentiment Impact on Performance")
    
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    
    perf = filtered_df.groupby('sentiment').agg(
        days=('date', 'count'),
        avg_pnl=('total_pnl', 'mean'),
        avg_win_rate=('win_rate', 'mean'),
        avg_volume=('total_volume', 'mean'),
        avg_ls_ratio=('long_short_ratio', 'mean'),
        avg_trades=('num_trades', 'mean'),
        avg_size=('avg_trade_size', 'mean'),
    ).reindex(sentiment_order).dropna(how='all')
    
    # Multi-metric comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            perf.reset_index(), x='sentiment', y='avg_pnl',
            color='sentiment', color_discrete_map=colors_map,
            title="Average Daily PnL by Sentiment"
        )
        fig.add_hline(y=0, line_dash='dash', line_color='black')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            perf.reset_index(), x='sentiment', y='avg_win_rate',
            color='sentiment', color_discrete_map=colors_map,
            title="Average Win Rate by Sentiment"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig = px.bar(
            perf.reset_index(), x='sentiment', y='avg_ls_ratio',
            color='sentiment', color_discrete_map=colors_map,
            title="Long/Short Ratio by Sentiment"
        )
        fig.add_hline(y=1.0, line_dash='dash', line_color='black', annotation_text="Neutral")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        fig = px.bar(
            perf.reset_index(), x='sentiment', y='avg_size',
            color='sentiment', color_discrete_map=colors_map,
            title="Avg Trade Size by Sentiment"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot
    st.subheader("Continuous Relationship")
    scatter_metric = st.selectbox(
        "Select metric to compare with FG Index:",
        ['total_pnl', 'win_rate', 'num_trades', 'long_short_ratio', 'avg_trade_size']
    )
    
    fig = px.scatter(
        filtered_df, x='fg_value', y=scatter_metric,
        color='sentiment', color_discrete_map=colors_map,
        trendline='ols', title=f"FG Index vs {scatter_metric}",
        opacity=0.5
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    st.subheader(" Detailed Performance Table")
    st.dataframe(perf.round(2), use_container_width=True)


# ─────────────────────────────────────────────────
# TAB 3: TRADER SEGMENTS
# ─────────────────────────────────────────────────
with tab3:
    st.header("Trader Segments & Archetypes")
    
    # Segmentation explorer
    seg_type = st.radio(
        "Choose segmentation method:",
        ["By Win Rate", "By Trade Frequency", "By Position Size"],
        horizontal=True
    )
    
    tp = trader_profiles.copy()
    
    if seg_type == "By Win Rate":
        tp['segment'] = np.where(tp['win_rate'] >= 50, 'Winners (≥50%)', 'Struggling (<50%)')
    elif seg_type == "By Trade Frequency":
        median_trades = tp['total_trades'].median()
        tp['segment'] = np.where(tp['total_trades'] >= median_trades, 'Frequent', 'Infrequent')
    else:
        median_size = tp['avg_trade_size'].median()
        tp['segment'] = np.where(tp['avg_trade_size'] >= median_size, 'Large', 'Small')
    
    # Segment summary
    seg_summary = tp.groupby('segment').agg(
        traders=('Account', 'count'),
        avg_pnl=('total_pnl', 'mean'),
        avg_wr=('win_rate', 'mean'),
        avg_size=('avg_trade_size', 'mean'),
        avg_trades=('total_trades', 'mean'),
    ).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Segment Summary")
        st.dataframe(seg_summary, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            tp, x='win_rate', y='total_pnl', color='segment',
            size='total_trades', size_max=15,
            title="Trader Scatter: Win Rate vs Total PnL",
            hover_data=['Account', 'avg_trade_size', 'active_days'],
            opacity=0.6
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            tp, x='win_rate', color='segment', nbins=30,
            title="Win Rate Distribution by Segment",
            barmode='overlay', opacity=0.7
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            tp, x='segment', y='total_pnl', color='segment',
            title="PnL Distribution by Segment",
            points='outliers'
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top/Bottom traders
    st.subheader(" Top & Bottom Traders")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top 10 by Total PnL:**")
        top10 = tp.nlargest(10, 'total_pnl')[
            ['Account', 'total_pnl', 'win_rate', 'total_trades', 'avg_trade_size']
        ]
        st.dataframe(top10.round(2), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Bottom 10 by Total PnL:**")
        bottom10 = tp.nsmallest(10, 'total_pnl')[
            ['Account', 'total_pnl', 'win_rate', 'total_trades', 'avg_trade_size']
        ]
        st.dataframe(bottom10.round(2), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────
# TAB 4: PREDICTIONS & STRATEGY
# ─────────────────────────────────────────────────
with tab4:
    st.header("Predictions & Strategy Rules")
    
    # Current sentiment indicator
    latest = merged_df.sort_values('date').iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fg_val = latest['fg_value']
        fg_color = (
            "🔴" if fg_val <= 25 else
            "🟠" if fg_val <= 45 else
            "🟡" if fg_val <= 55 else
            "🟢" if fg_val <= 75 else
            "💚"
        )
        st.metric(
            f"{fg_color} Latest FG Index",
            f"{fg_val:.0f} ({latest['sentiment']})",
            delta=f"{latest['fg_value'] - merged_df.sort_values('date').iloc[-2]['fg_value']:+.0f} from yesterday"
        )
    with col2:
        st.metric("Latest Daily PnL", f"${latest['total_pnl']:,.0f}")
    with col3:
        st.metric("Latest Win Rate", f"{latest['win_rate']:.1f}%")
    
    st.markdown("---")
    
    # Strategy decision matrix
    st.subheader("Strategy Decision Matrix")
    
    strategy_data = {
        'Sentiment': ['Extreme Fear (≤25)', 'Fear (26-45)', 'Neutral (46-55)', 
                      'Greed (56-75)', 'Extreme Greed (>75)'],
        'Position Sizing': [
            'Cut 50% — Capital preservation',
            'Cut 30% — Selective entries only',
            'Normal sizing — Standard rules',
            'Normal — Trail stops tighter',
            'Cut 30-40% — Take profits'
        ],
        'Trade Frequency': [
            'Reduce 50% — A+ setups only',
            'Reduce 30% — Higher timeframes',
            'Normal frequency',
            'Normal — Avoid FOMO',
            'Reduce 30% — No new FOMO'
        ],
        'Long/Short Bias': [
            ' Reduce longs — Hedging priority',
            ' Cautious long — Watch reversals',
            ' Balanced approach',
            ' Moderate long bias',
            ' Reduce longs — Crowded trade'
        ]
    }
    
    strategy_df = pd.DataFrame(strategy_data)
    
    # Highlight current regime
    current_sentiment = latest['sentiment']
    st.dataframe(strategy_df, use_container_width=True, hide_index=True)
    
    st.info(f"**Current regime: {current_sentiment}** — "
            f"Check the corresponding row above for recommended actions.")
    
    st.markdown("---")
    
    # Model performance summary
    st.subheader(" Predictive Model Summary")
    
    st.markdown("""
    **Model:** Gradient Boosting / Random Forest classifier  
    **Target:** Next-day profitability bucket (Loss / Breakeven / Profit)  
    **Validation:** TimeSeriesSplit (5-fold, temporal order preserved)  
    
    **Key Predictive Features (ranked by importance):**
    1. 🔵 Fear & Greed Index (current + 7-day rolling)
    2. 🔵 Previous-day PnL and win rate
    3. 🔵 FG momentum (rate of sentiment change)
    4. 🔵 PnL volatility (7-day rolling std)
    5. 🔵 Trade volume and frequency metrics
    
    **Practical Use:**
    - Check the FG Index each morning
    - If FG is dropping rapidly (negative momentum), expect lower next-day PnL
    - Combine with your trader segment for personalized sizing rules
    """)
    
    # Quick backtesting visualization
    st.subheader("Sentiment Regime Returns")
    
    regime_returns = filtered_df.groupby('sentiment')['total_pnl'].agg(['mean', 'std', 'count'])
    regime_returns.columns = ['Mean PnL', 'Std PnL', 'Days']
    regime_returns['Sharpe-like'] = (regime_returns['Mean PnL'] / regime_returns['Std PnL']).round(3)
    regime_returns = regime_returns.reindex(
        ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    ).dropna()
    
    st.dataframe(regime_returns.round(2), use_container_width=True)

# ─────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 12px;'>"
    "Trader Sentiment Analysis Dashboard | "
    "Data: HyperLiquid Historical Trades + Crypto Fear & Greed Index | "
    "Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True
)