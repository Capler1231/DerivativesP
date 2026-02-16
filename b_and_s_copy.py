import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Derivatives Analytics Dashboard",
    page_icon="",
    layout="wide"
)

st.title("Derivatives Analytics")
st.caption("Black-Scholes Pricing - Monte Carlo simulation - Volatility modeling")

# Black-Scholes calculation functions
def calculate_d1_d2(S, K, T, r, sigma):
    """Calculate d1 and d2 for Black-Scholes formula"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def black_scholes_call(S, K, T, r, sigma):
    """Calculate call option price"""
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate put option price"""
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def risk_neutral_density(S, T, r, sigma):

    mu = np.log(S) + (r - 0.5 * sigma**2) * T
    variance = sigma**2 * T

    ST_range = np.linspace(S*0.2, S*2, 500)

    density = (1 / (ST_range * sigma * np.sqrt(2*np.pi*T))) * \
              np.exp(-(np.log(ST_range) - mu)**2 / (2 * variance))

    return ST_range, density

def calculate_greeks(S, K, T, r, sigma):
    """Calculate the Greeks (sensitivities)"""
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    
    # Delta
    call_delta = norm.cdf(d1)
    put_delta = norm.cdf(d1) - 1
    
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta (per day)
    call_theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                  r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    put_theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Vega (per 1% change)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    # Rho (per 1% change)
    call_rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        'call_delta': call_delta,
        'put_delta': put_delta,
        'gamma': gamma,
        'call_theta': call_theta,
        'put_theta': put_theta,
        'vega': vega,
        'call_rho': call_rho,
        'put_rho': put_rho
    }

# Create input columns
st.sidebar.header("Model Inputs")


S_input = st.sidebar.number_input(
    "Stock Price (Manual Override)",
    min_value=0.01,
    max_value=5000.0,
    value=42.0,
    help="Manual stock price"
    )
    
r = st.sidebar.number_input(
        "Risk-Free Rate (r)",
        min_value=0.001,
        max_value=0.3,
        value=0.025,
        step=0.001,
        format="%.3f",
        help="Annual risk-free interest rate (e.g., 0.025 for 2.5%)"
    )

ticker = st.sidebar.text_input("Stock Ticker (optional)", "AAPL")

use_live_price = st.sidebar.checkbox("Use Live Market Price", value=False)

S_market = None

if use_live_price:
    try:
        data = yf.Ticker(ticker)
        S_market = data.history(period="1d")["Close"].iloc[-1]
        st.write(f"Live Price: ${S_market:.2f}")
    except:
        st.warning("Could not fetch live price. Using manual input.")

S = S_market if use_live_price and S_market else S_input

K = st.sidebar.number_input(
        "Strike Price (K)",
        min_value=float(S * 0.5),
        max_value=float(S * 1.5),
        value=float(S),
        step=1.0,
        help="Strike price of the option"
    )
    
sigma = st.sidebar.number_input(
        "Volatility (σ)",
        min_value=0.001,
        max_value=1.0,
        value=0.20,
        step=0.001,
        format="%.3f",
        help="Annual volatility (e.g., 0.20 for 20%)"
    )

Days = st.sidebar.number_input(
        "Days to Expiration",
        min_value=1,
        max_value=3650,
        value=365,
        step=1,
        help="Days to expiration (Gets converted to years (T) for calculations)"
    )
T = Days / 365.0

# --- Moneyness ---
moneyness = S / K

st.markdown("### Market Diagnostics")

d1_col, d2_col, d3_col, d4_col = st.columns(4)

d1_col.metric("Stock Price", f"${S:.2f}")
d2_col.metric("Strike", f"${K:.2f}")
d3_col.metric("Moneyness", f"{S/K:.3f}")
d4_col.metric("Days to Expiry", f"{Days}")

st.divider()

# Calculate prices
call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

#Calculate intrinsic values
call_intrinsic = max(S - K, 0)
put_intrinsic = max(K - S, 0)

call_time_value = call_price - call_intrinsic
put_time_value = put_price - put_intrinsic

# Display main results
st.subheader("Option Prices")
col1, col2 = st.columns(2)

with col1:
    st.metric("Call Option Price", f"${call_price:.4f}")
    st.caption(f"Intrinsic Value: ${call_intrinsic:.4f}")
    st.caption(f"Time Value: ${call_time_value:.4f}")

with col2:
    st.metric("Put Option Price", f"${put_price:.4f}")
    st.caption(f"Intrinsic Value: ${put_intrinsic:.4f}")
    st.caption(f"Time Value: ${put_time_value:.4f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Pricing & Simulation",
    "Volatility Smile",
    "Greeks Heatmap",
    "Risk-Neutral Density",
    "Volatility Surface"
])

st.divider()

# Calculate and display Greeks
greeks = calculate_greeks(S, K, T, r, sigma)

st.subheader("Greeks")

g1, g2, g3, g4, g5 = st.columns(5)

g1.metric("Delta (Call)", f"{greeks['call_delta']:.4f}")
g2.metric("Gamma", f"{greeks['gamma']:.6f}")
g3.metric("Theta (Call)", f"{greeks['call_theta']:.4f}")
g4.metric("Vega", f"{greeks['vega']:.4f}")
g5.metric("Rho (Call)", f"{greeks['call_rho']:.4f}")

#monte carlo simulation for call option
def monte_carlo_call(
    S, K, T, r, sigma,
    simulations=50000,
    seed=None,
    antithetic=True,
    return_path=False
):

    if seed is not None:
        np.random.seed(seed)

    if antithetic:
        Z = np.random.standard_normal(simulations // 2)
        Z = np.concatenate([Z, -Z])
    else:
        Z = np.random.standard_normal(simulations)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z

    ST = S * np.exp(drift + diffusion)

    payoff = np.maximum(ST - K, 0)
    discounted = np.exp(-r * T) * payoff

    if return_path:
        cumulative = np.cumsum(discounted) / np.arange(1, len(discounted) + 1)
        return cumulative

    return np.mean(discounted)


def implied_volatility(option_price, S, K, T, r, 
                       option_type="call", tol=1e-6):

    sigma = 0.2

    for _ in range(100):
        if option_type == "call":
            price = black_scholes_call(S, K, T, r, sigma)
        else:
            price = black_scholes_put(S, K, T, r, sigma)

        vega = calculate_greeks(S, K, T, r, sigma)['vega']
        diff = price - option_price

        if abs(diff) < tol:
            return sigma

        sigma -= diff / vega

    return sigma

def get_option_chain_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options
    
    if not expirations:
        return None, None, None
    
    expiry = expirations[0]  # nearest expiration
    chain = ticker.option_chain(expiry)
    
    return expiry, chain.calls, chain.puts

st.divider()

with tab1:
    st.subheader("Monte Carlo Pricing")

    simulations = st.slider("Simulations", 1000, 200000, 50000)
    use_fixed_seed = st.checkbox("Use Fixed Seed for Reproducibility", value=True)
    seed_value = 42 if use_fixed_seed else None

    use_antithetic = st.checkbox(
        "Use Antithetic Variates (Variance Reduction)", 
        value=True
        )
    
    raw_Z = np.random.standard_normal(simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * raw_Z)
    payoffs = np.exp(-r * T) * np.maximum(ST - K, 0)

    std_error = np.std(payoffs) / np.sqrt(simulations)
    st.write(f"Estimated Standard Error: ${std_error:.6f}"
    )
    

    mc_price = monte_carlo_call(
        S, K, T, r, sigma, 
        simulations, 
        seed=seed_value,
        antithetic=use_antithetic
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Black-Scholes Call", f"${call_price:.4f}")

    with col2:
        st.metric("Monte Carlo Call", f"${mc_price:.4f}")

    st.write("Absolute Difference:", abs(call_price - mc_price))

    st.markdown("### Convergence to Black-Scholes Price")

    convergence_path = monte_carlo_call(
        S, K, T, r, sigma, 
        simulations = simulations, 
        seed=seed_value,
        antithetic=use_antithetic, 
        return_path=True
    )
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=convergence_path,
        mode='lines',
        name='Monte Carlo Estimate'
    ))

    fig.add_hline(
        y=call_price, 
        line_dash="dash", 
        line_color="red", 
        annotation_text="Black-Scholes Price", 
        annotation_position="top right"
    )

    fig.update_layout(
        xaxis_title="Number of Simulations",
        yaxis_title="Estimated Call Price",
    )

    st.plotly_chart(fig, width = "stretch")

with tab2:
    st.subheader("Implied Volatility Smile")

    ticker_obj = yf.Ticker(ticker)
    expirations = ticker_obj.options

    if not expirations:
        st.warning("No option data available.")
    else:
        # Let user select expiration
        expiry = st.selectbox("Select Expiration", expirations)

        chain = ticker_obj.option_chain(expiry)
        calls_df = chain.calls
        puts_df = chain.puts

        st.write(f"Selected Expiration: {expiry}")

        # Filter out zero IVs and bad data
        calls_df = calls_df[
            (calls_df["impliedVolatility"] > 0) &
            (calls_df["strike"] / S > 0.7) &
            (calls_df["strike"] / S < 1.3)
            ]
        
        puts_df = puts_df[
            (puts_df["impliedVolatility"] > 0) &
            (puts_df["strike"] / S > 0.7) &
            (puts_df["strike"] / S < 1.3)
            ]
        
        combined_df = pd.concat([calls_df, puts_df])

        combined_df["moneyness"] = combined_df["strike"] / S

        atm_window = combined_df[
            (combined_df["moneyness"] > 0.9) &
            (combined_df["moneyness"] < 1.1)
            ]
        
        if len(atm_window) > 5:
            x = atm_window["moneyness"].values
            y = atm_window["impliedVolatility"].values

            #Linear regression (1st degree polynomial fit)
            slope, intercept = np.polyfit(x, y, 1)

            st.markdown("### Skew Analytics")
            st.write(f"ATM Skew Slope: {slope: .4f}")
            st.write(f"Slope (% change in IV per 1 unit moneyness): {slope * 100:.2f}%")

            if slope < 0:
                st.write("Negative skew (Downside protection demand)")
            else:
                st.write("Positive skew (Upside speculation)")

        # Plot Smile
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=calls_df["strike"] / S,
            y=calls_df["impliedVolatility"],
            mode='lines',
            name='Calls'
        ))

        fig.add_trace(go.Scatter(
            x=puts_df["strike"] / S,
            y=puts_df["impliedVolatility"],
            mode='lines',
            name='Puts'
        ))

        fig.add_hline(
            y=sigma,
            line_dash="dot",
            line_color="blue",
            annotation_text="Model Volatility",
        )

        fig.add_vline(
            x = K / S, 
            line_dash = "dash", 
            line_color="gray", 
            annotation_text="At-the-money", 
            annotation_position="top left"
            )

        fig.update_layout(
            xaxis_title="Moneyness (K / S)",
            yaxis_title="Implied Volatility",
            yaxis = dict(tickformat=".1%")
        )

        if len(atm_window) > 5:
            x_line = np.linspace(0.9, 1.1, 50)
            y_line = slope * x_line + intercept

            fig.add_trace(go.Scatter(
                x = x_line,
                y = y_line,
                mode = 'lines',
                name = 'Skew Fit',
                line = dict(dash = "dash"))
            )

        st.plotly_chart(fig, width = "stretch")

with tab3:
    st.subheader("Delta Heatmap")

    S_range = np.linspace(S*0.5, S*1.5, 40)
    sigma_range = np.linspace(0.05, 0.6, 40)

    delta_matrix = np.zeros((40, 40))

    for i, s_val in enumerate(S_range):
        for j, vol in enumerate(sigma_range):
            delta_matrix[i, j] = calculate_greeks(
                s_val, K, T, r, vol
            )['call_delta']

    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(delta_matrix, ax=ax)
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Stock Price")
    st.pyplot(fig)

with tab4:
    st.subheader("Risk-Neutral Probability Density")

    ST_range, density = risk_neutral_density(S, T, r, sigma)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ST_range,
        y=density,
        mode='lines',
        name='Risk-Neutral Density'
    ))

    fig.add_vline(x=K, line_dash="dash", annotation_text="Strike")

    fig.update_layout(
        xaxis_title="Future Stock Price (S_T)",
        yaxis_title="Probability Density"
    )

    st.plotly_chart(fig, width = "stretch")

with tab5:
    st.subheader("Market Volatility Surface")

    ticker_obj = yf.Ticker(ticker)
    expirations = ticker_obj.options

    if not expirations:
        st.warning("No option data available.")
    else:
        selected_expirations = expirations[:5]  # first 5 expiries

        surface_data = []

        for expiry in selected_expirations:
            chain = ticker_obj.option_chain(expiry)
            calls = chain.calls
            calls = calls[calls["impliedVolatility"] > 0]

            time_to_expiry = (
                pd.to_datetime(expiry) - pd.Timestamp.today()
            ).days / 365.0

            for _, row in calls.iterrows():
                surface_data.append([
                    row["strike"],
                    time_to_expiry,
                    row["impliedVolatility"]
                ])

        surface_df = pd.DataFrame(surface_data,
                                  columns=["Strike", "Time", "IV"])

        fig = go.Figure(data=[go.Mesh3d(
            x=surface_df["Strike"],
            y=surface_df["Time"],
            z=surface_df["IV"],
            opacity=0.7
        )])

        fig.update_layout(
            scene=dict(
                xaxis_title="Strike",
                yaxis_title="Time to Expiry",
                zaxis_title="Implied Volatility"
            )
        )

        st.plotly_chart(fig, width = "stretch")

# Add information section
with st.expander("Model & Quantitative Framework Information"):

    st.markdown("""
    ## 1️ Black-Scholes Model

    **Call Price**
    $$C = S N(d_1) - K e^{-rT} N(d_2)$$

    **Put Price**
    $$P = K e^{-rT} N(-d_2) - S N(-d_1)$$

    $$d_1 = \\frac{\\ln(S/K) + (r + \\sigma^2/2)T}{\\sigma\\sqrt{T}}$$
    $$d_2 = d_1 - \\sigma\\sqrt{T}$$

    **Interpretation**
    - $N(d_1)$ ≈ risk-adjusted probability weighting
    - $N(d_2)$ ≈ risk-neutral probability of finishing ITM
    - Prices assume continuous hedging and no arbitrage

    ---

    ## 2️ Greeks (Risk Sensitivities)

    - **Delta**: Sensitivity to underlying price
    - **Gamma**: Sensitivity of delta
    - **Theta**: Time decay
    - **Vega**: Sensitivity to volatility
    - **Rho**: Sensitivity to interest rates

    These measure how option value changes with market conditions.

    ---

    ## 3️ Monte Carlo Simulation

    Stock evolution under risk-neutral measure:

    $$S_T = S \\exp((r - 0.5\\sigma^2)T + \\sigma \\sqrt{T} Z)$$

    - Uses random sampling
    - Converges at rate $O(1/\\sqrt{N})$
    - Antithetic variates reduce variance
    - Convergence chart shows numerical stability

    ---

    ## 4️ Implied Volatility

    Implied volatility is the value of σ such that:

    $$\\text{Market Price} = \\text{Black-Scholes Price}(\\sigma)$$

    The volatility smile demonstrates that real markets do **not**
    assume constant volatility.

    ---

    ## 5️ Volatility Smile & Skew

    - Plotted against moneyness (K/S)
    - Negative skew: downside protection demand
    - Positive skew: upside speculation
    - Slope estimated via local linear regression near ATM

    ---

    ## 6️ Risk-Neutral Density

    Under Black-Scholes:

    $$S_T \\sim \\text{Lognormal}$$

    The plotted density represents the market-implied future
    probability distribution under the risk-neutral measure.

    ---

    ## 7️ Volatility Surface

    The 3D surface across strikes and expirations shows:

    - Term structure of volatility
    - Skew variation over time
    - Market perception of risk

    Real markets exhibit non-flat surfaces,
    contradicting constant volatility assumption.

    ---

    ## Model Assumptions

    - European exercise
    - No dividends
    - Constant volatility
    - No transaction costs
    - Log-normal price dynamics
    
    ---
    
    This dashboard integrates analytical models, 
    numerical simulation, and live market data 
    to explore modern derivatives pricing and 
    volatility dynamics.

                
    """)

