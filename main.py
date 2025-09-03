import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

class GeneticPredictor:
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.best_individual = None
        self.best_fitness_all_time = float('-inf')  # Track best fitness across all generations
        self.fitness_history = []
        
    def create_individual(self) -> List[float]:
        """Create a random individual (weights for technical indicators)"""
        # Weights for: SMA_5, SMA_20, RSI, MACD, BB_upper, BB_lower, volume_ratio, price_change
        return [random.uniform(-1, 1) for _ in range(8)]
    
    def create_population(self) -> List[List[float]]:
        """Create initial population"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        features = pd.DataFrame()
        
        # Simple Moving Averages
        features['sma_5'] = ta.trend.sma_indicator(data['Close'], window=5)
        features['sma_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        
        # RSI
        features['rsi'] = ta.momentum.rsi(data['Close'], window=14)
        
        # MACD
        features['macd'] = ta.trend.macd_diff(data['Close'])
        
        # Bollinger Bands
        bb_high = ta.volatility.bollinger_hband(data['Close'])
        bb_low = ta.volatility.bollinger_lband(data['Close'])
        features['bb_upper'] = (data['Close'] - bb_high) / data['Close']
        features['bb_lower'] = (data['Close'] - bb_low) / data['Close']
        
        # Volume ratio
        features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
        
        # Price change
        features['price_change'] = data['Close'].pct_change()
        
        # Normalize features
        for col in features.columns:
            features[col] = (features[col] - features[col].mean()) / (features[col].std() + 1e-8)
        
        return features.fillna(0)
    
    def predict_price(self, individual: List[float], features: pd.DataFrame, current_price: float) -> float:
        """Predict next price using individual's weights"""
        if len(features) == 0:
            return current_price
        
        latest_features = features.iloc[-1].values
        prediction_factor = np.dot(individual, latest_features)
        
        # Convert to price prediction (small percentage change)
        prediction_factor = np.tanh(prediction_factor) * 0.05  # Limit to ±5%
        predicted_price = current_price * (1 + prediction_factor)
        
        return predicted_price
    
    def chain_predict(self, individual: List[float], features: pd.DataFrame, 
                     current_price: float, n_steps: int) -> List[float]:
        """Make n-step ahead predictions by chaining predictions"""
        predictions = []
        current_features = features.copy()
        last_price = current_price
        
        for _ in range(n_steps):
            # Predict next price
            predicted_price = self.predict_price(individual, current_features, last_price)
            predictions.append(predicted_price)
            
            # Update features for next prediction (simulate new data point)
            # This is a simplified approach - in reality, we'd need to update all indicators
            last_price = predicted_price
            
        return predictions
    
    def fitness(self, individual: List[float], data: pd.DataFrame, n_steps: int = 1) -> float:
        """Calculate fitness of an individual with n-step ahead prediction"""
        features = self.calculate_features(data)
        
        if len(features) < 50:  # Need enough data
            return float('-inf')
        
        predictions = []
        actuals = []
        
        # Test on last 100 points
        test_start = max(50, len(data) - 100)
        
        for i in range(test_start, len(data) - n_steps):
            current_features = features.iloc[:i+1]
            current_price = data['Close'].iloc[i]
            
            # Make n-step ahead predictions
            predicted_prices = self.chain_predict(individual, current_features, current_price, n_steps)
            actual_next_prices = [data['Close'].iloc[i+j+1] for j in range(n_steps)]
            
            predictions.extend(predicted_prices)
            actuals.extend(actual_next_prices)
        
        if len(predictions) == 0:
            return float('-inf')
        
        # Calculate fitness based on prediction accuracy
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Mean Absolute Percentage Error (lower is better)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Return negative MAPE as fitness (higher is better)
        return -mape
    
    def selection(self, population: List[List[float]], fitness_scores: List[float]) -> List[List[float]]:
        """Tournament selection"""
        selected = []
        tournament_size = 3
        
        for _ in range(self.population_size):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Single point crossover"""
        if random.random() > 0.8:  # 80% crossover rate
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def mutate(self, individual: List[float]) -> List[float]:
        """Random mutation"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] += random.gauss(0, 0.1)
                mutated[i] = max(-1, min(1, mutated[i]))  # Keep in bounds
        return mutated
    
    def evolve(self, data: pd.DataFrame, n_steps: int = 1, progress_callback=None) -> List[float]:
        """Main evolution loop"""
        population = self.create_population()
        self.fitness_history = []
        self.best_fitness_all_time = float('-inf')
        
        for generation in range(self.generations):
            # Calculate fitness
            fitness_scores = [self.fitness(individual, data, n_steps) for individual in population]
            
            # Track best fitness
            best_fitness = max(fitness_scores)
            self.fitness_history.append(best_fitness)
            
            # Update best individual and all-time best fitness
            best_idx = np.argmax(fitness_scores)
            if best_fitness > self.best_fitness_all_time:
                self.best_fitness_all_time = best_fitness
                self.best_individual = population[best_idx].copy()
            
            if progress_callback:
                progress_callback(generation + 1, self.generations, best_fitness, self.best_fitness_all_time)
            
            # Selection
            selected = self.selection(population, fitness_scores)
            
            # Create next generation
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % self.population_size]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        return self.best_individual

def fetch_stock_data(symbol: str, period: str = "100d") -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance"""
    try:
        st.info(f"Attempting to fetch data for {symbol}...")
        ticker = yf.Ticker(symbol)
        
        # Try different intervals if 1m doesn't work
        intervals = ["1m", "5m", "15m"]
        periods = ["100d", "60d", "30d"]
        
        for interval in intervals:
            for p in periods:
                try:
                    st.info(f"Trying {interval} interval with {p} period...")
                    data = ticker.history(period=p, interval=interval)
                    
                    if not data.empty and len(data) >= 100:
                        st.success(f"✅ Successfully fetched data with {interval} interval!")
                        # Get last 6000 candles or all available data
                        data = data.tail(6000)
                        return data
                except Exception as inner_e:
                    continue
        
        # If still no data, try daily data as fallback
        st.warning("Trying daily data as fallback...")
        data = ticker.history(period="1y", interval="1d")
        if not data.empty:
            st.info("Using daily data instead of minute data")
            return data.tail(6000)
        
        st.error(f"❌ No data found for symbol {symbol}. Please check if the symbol is correct.")
        st.info("Common symbols: AAPL (Apple), GOOGL (Google), MSFT (Microsoft), TSLA (Tesla)")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        st.info("This might be due to:")
        st.info("• Invalid stock symbol")
        st.info("• Market is closed (try during trading hours)")
        st.info("• Yahoo Finance API limitations")
        st.info("• Internet connectivity issues")
        return pd.DataFrame()

def main():
    st.set_page_config(page_title="Genetic Stock Predictor", layout="wide")
    
    st.title("🧬 Genetic Algorithm Stock Price Predictor")
    st.markdown("Uses evolutionary algorithms to predict the next candle's close price")
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter a valid stock symbol (e.g., AAPL, GOOGL, TSLA)")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        population_size = st.number_input("Population Size", min_value=2, max_value=50000, value=50)
        generations = st.number_input("Generations", min_value=5, max_value=5000, value=100)
        n_steps = st.number_input("Prediction Steps", min_value=1, max_value=10, value=1, 
                                 help="Number of future candles to predict")
    
    with col2:
        mutation_rate = st.slider("Mutation Rate", min_value=0.01, max_value=5.0, value=0.1)
    
    if st.sidebar.button("🚀 Start Prediction", type="primary"):
        if symbol:
            with st.spinner(f"Fetching data for {symbol}..."):
                data = fetch_stock_data(symbol)
            
            if not data.empty:
                st.success(f"Fetched {len(data)} candles for {symbol}")
                
                # Display basic info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
                with col2:
                    change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                    st.metric("Last Change", f"${change:.2f}", delta=f"{change:.2f}")
                with col3:
                    st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
                with col4:
                    st.metric("Data Points", len(data))
                
                # Initialize genetic algorithm
                predictor = GeneticPredictor(
                    population_size=population_size,
                    generations=generations,
                    mutation_rate=mutation_rate
                )
                
                # Evolution progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                fitness_chart_placeholder = st.empty()
                
                def update_progress(generation, total_generations, best_fitness, best_fitness_all_time):
                    progress = generation / total_generations
                    progress_bar.progress(progress)
                    status_text.text(f"Generation {generation}/{total_generations} - Best Fitness: {best_fitness:.2f} - All-time Best: {best_fitness_all_time:.2f}")
                    
                    # Update fitness chart
                    if len(predictor.fitness_history) > 1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=predictor.fitness_history,
                            mode='lines',
                            name='Current Generation Fitness',
                            line=dict(color='green')
                        ))
                        # Add a horizontal line for the all-time best fitness
                        fig.add_hline(y=best_fitness_all_time, line_dash="dash", 
                                     line_color="red", annotation_text="All-time Best")
                        fig.update_layout(
                            title="Evolution Progress",
                            xaxis_title="Generation",
                            yaxis_title="Fitness (Negative MAPE)",
                            height=300
                        )
                        fitness_chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Run evolution
                with st.spinner("Training genetic algorithm..."):
                    best_weights = predictor.evolve(data, n_steps, update_progress)
                
                progress_bar.progress(1.0)
                status_text.text("Evolution completed!")
                
                if best_weights:
                    # Make prediction
                    features = predictor.calculate_features(data)
                    current_price = data['Close'].iloc[-1]
                    
                    # Make chain predictions
                    predicted_prices = predictor.chain_predict(best_weights, features, current_price, n_steps)
                    
                    # Display prediction
                    st.header("🎯 Prediction Results")
                    
                    # Create columns for each prediction step
                    cols = st.columns(n_steps + 1)
                    
                    with cols[0]:
                        st.metric("Current Close", f"${current_price:.2f}")
                    
                    for i in range(n_steps):
                        with cols[i + 1]:
                            if i == 0:
                                predicted_change = predicted_prices[i] - current_price
                            else:
                                predicted_change = predicted_prices[i] - predicted_prices[i-1]
                            
                            st.metric(f"Step {i+1} Prediction", f"${predicted_prices[i]:.2f}", 
                                    delta=f"${predicted_change:.2f}")
                    
                    # Show evolved weights
                    st.subheader("🧬 Evolved Model Weights")
                    weight_names = ['SMA_5', 'SMA_20', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Volume_Ratio', 'Price_Change']
                    
                    weights_df = pd.DataFrame({
                        'Feature': weight_names,
                        'Weight': best_weights
                    })
                    
                    fig = go.Figure(data=[
                        go.Bar(x=weight_names, y=best_weights, 
                               marker_color=['red' if w < 0 else 'green' for w in best_weights])
                    ])
                    fig.update_layout(
                        title="Feature Weights (Evolved)",
                        xaxis_title="Technical Indicators",
                        yaxis_title="Weight",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Price chart with prediction
                    st.subheader("📈 Price Chart with Prediction")
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        vertical_spacing=0.1,
                        subplot_titles=('Price', 'Volume'),
                        row_width=[0.7, 0.3]
                    )
                    
                    # Price candlestick
                    last_100 = data.tail(100)
                    fig.add_trace(
                        go.Candlestick(
                            x=last_100.index,
                            open=last_100['Open'],
                            high=last_100['High'],
                            low=last_100['Low'],
                            close=last_100['Close'],
                            name="Price"
                        ),
                        row=1, col=1
                    )
                    
                    # Add prediction points
                    next_times = [last_100.index[-1] + pd.Timedelta(minutes=i+1) for i in range(n_steps)]
                    fig.add_trace(
                        go.Scatter(
                            x=next_times,
                            y=predicted_prices,
                            mode='markers+lines',
                            marker=dict(size=7, color='red', symbol='circle'),
                            line=dict(color='red', dash='dash'),
                            name='Predictions'
                        ),
                        row=1, col=1
                    )
                    
                    # Volume
                    fig.add_trace(
                        go.Bar(
                            x=last_100.index,
                            y=last_100['Volume'],
                            name="Volume",
                            marker_color='lightblue'
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        title=f"{symbol} - Last 100 Candles with {n_steps}-Step Prediction",
                        xaxis_rangeslider_visible=False,
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Model performance metrics
                    st.subheader("📊 Model Performance")
                    
                    # Calculate some validation metrics on recent data
                    test_data = data.tail(50)  # Last 50 points for testing
                    test_features = predictor.calculate_features(test_data)
                    
                    predictions = []
                    actuals = []
                    
                    for i in range(25, len(test_data) - n_steps):
                        current_features = test_features.iloc[:i+1]
                        current_price = test_data['Close'].iloc[i]
                        
                        # Make n-step ahead predictions
                        predicted_prices_test = predictor.chain_predict(best_weights, current_features, current_price, n_steps)
                        actual_next_prices = [test_data['Close'].iloc[i+j+1] for j in range(n_steps)]
                        
                        predictions.extend(predicted_prices_test)
                        actuals.extend(actual_next_prices)
                    
                    if predictions:
                        predictions = np.array(predictions)
                        actuals = np.array(actuals)
                        
                        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
                        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("MAPE (Mean Absolute Percentage Error)", f"{mape:.2f}%")
                        with col2:
                            st.metric("RMSE (Root Mean Square Error)", f"${rmse:.2f}")
                        
                        # Show prediction vs actual chart
                        comparison_fig = go.Figure()
                        comparison_fig.add_trace(go.Scatter(
                            y=actuals, mode='lines+markers', name='Actual', line=dict(color='blue')
                        ))
                        comparison_fig.add_trace(go.Scatter(
                            y=predictions, mode='lines+markers', name='Predicted', line=dict(color='red')
                        ))
                        comparison_fig.update_layout(
                            title="Actual vs Predicted (Validation Set)",
                            yaxis_title="Price ($)",
                            xaxis_title="Test Sample",
                            height=400
                        )
                        st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Random seed information
                st.subheader("🎲 Random Seed Information")
                st.info(f"""
                The genetic algorithm uses random processes for evolution. For reproducible results, 
                you can set a random seed. Current session uses dynamic seeding based on the evolution process.
                
                **Key Features:**
                - Population-based optimization
                - Tournament selection
                - Single-point crossover
                - Gaussian mutation
                - Technical indicator feature engineering
                - {n_steps}-step ahead prediction
                """)
            
        else:
            st.warning("Please enter a stock symbol")
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.markdown("""
    This app uses genetic algorithms to evolve trading strategies for stock price prediction.
    
    **Features:**
    - Fetches 1-minute candle data
    - Uses 8 technical indicators
    - Evolves optimal weights
    - Provides next candle prediction
    
    **Disclaimer:** This is for educational purposes only. Not financial advice.
    """)

if __name__ == "__main__":
    main()
