import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load Model and Data
model = joblib.load('logistic_regression_model.pkl')
X_test = joblib.load('X_test.pkl')
y_test = joblib.load('y_test.pkl')

# Betting Simulation Functions
def simulate_betting_low_risk(X_test, y_test, model, initial_balance=1000):
    balance = initial_balance
    bet_history = []
    
    for i in range(len(X_test)):
        row = pd.DataFrame([X_test.iloc[i]], columns=X_test.columns)
        prediction_prob = model.predict_proba(row)[0]
        predicted_class = model.predict(row)[0]

        if prediction_prob[1] > 0.7:
            bet_amount = balance * 0.05  # Bet 5% of current balance
            odds = 1 / prediction_prob[1]
        else:
            continue  # Skip this game

        actual_result = y_test.iloc[i]
        if predicted_class == actual_result:
            payout = bet_amount * odds
            balance += payout - bet_amount
        else:
            balance -= bet_amount

        bet_history.append({
            'Game': i,
            'Predicted_Class': predicted_class,
            'Actual_Result': actual_result,
            'Bet_Amount': bet_amount,
            'Balance': balance,
            'Odds': odds,
            'Payout': payout if predicted_class == actual_result else 0,
            'Predicted_Probability': prediction_prob[1]
        })

    return pd.DataFrame(bet_history), balance

def simulate_betting_high_risk(X_test, y_test, model, initial_balance=1000):
    balance = initial_balance
    bet_history = []
    
    for i in range(len(X_test)):
        row = pd.DataFrame([X_test.iloc[i]], columns=X_test.columns)
        prediction_prob = model.predict_proba(row)[0]
        predicted_class = model.predict(row)[0]

        # Define high-risk as predictions with probabilities between 0.3 and 0.5
        if 0.3 <= prediction_prob[1] < 0.5:
            bet_amount = balance * 0.02  # Bet 2% of current balance
            odds = 1 / (1 - prediction_prob[1])
        else:
            continue  # Skip this game

        actual_result = y_test.iloc[i]
        if predicted_class == actual_result:
            payout = bet_amount * odds
            balance += payout - bet_amount
        else:
            balance -= bet_amount

        bet_history.append({
            'Game': i,
            'Predicted_Class': predicted_class,
            'Actual_Result': actual_result,
            'Bet_Amount': bet_amount,
            'Balance': balance,
            'Odds': odds,
            'Payout': payout if predicted_class == actual_result else 0,
            'Predicted_Probability': prediction_prob[1]
        })

    return pd.DataFrame(bet_history), balance

def simulate_betting_martingale(X_test, y_test, model, initial_balance=1000, initial_bet=10):
    balance = initial_balance
    bet_amount = initial_bet
    bet_history = []
    
    for i in range(len(X_test)):
        row = pd.DataFrame([X_test.iloc[i]], columns=X_test.columns)
        prediction_prob = model.predict_proba(row)[0]
        predicted_class = model.predict(row)[0]

        if balance <= 0:
            break  # Stop if balance is zero or negative
        
        odds = 1 / prediction_prob[1] if prediction_prob[1] > 0.5 else 1 / (1 - prediction_prob[1])

        actual_result = y_test.iloc[i]
        if predicted_class == actual_result:
            payout = bet_amount * odds
            balance += payout - bet_amount
            bet_amount = initial_bet  # Reset bet amount
        else:
            balance -= bet_amount
            bet_amount *= 2  # Double the bet amount

        bet_history.append({
            'Game': i,
            'Predicted_Class': predicted_class,
            'Actual_Result': actual_result,
            'Bet_Amount': bet_amount,
            'Balance': balance,
            'Odds': odds,
            'Payout': payout if predicted_class == actual_result else 0,
            'Predicted_Probability': prediction_prob[1]
        })

    return pd.DataFrame(bet_history), balance

def simulate_betting_flat(X_test, y_test, model, initial_balance=1000, bet_fraction=0.05):
    balance = initial_balance
    bet_history = []
    
    for i in range(len(X_test)):
        row = pd.DataFrame([X_test.iloc[i]], columns=X_test.columns)
        prediction_prob = model.predict_proba(row)[0]
        predicted_class = model.predict(row)[0]

        bet_amount = balance * bet_fraction
        odds = 1 / prediction_prob[1] if prediction_prob[1] > 0.5 else 1 / (1 - prediction_prob[1])

        actual_result = y_test.iloc[i]
        if predicted_class == actual_result:
            payout = bet_amount * odds
            balance += payout - bet_amount
        else:
            balance -= bet_amount

        bet_history.append({
            'Game': i,
            'Predicted_Class': predicted_class,
            'Actual_Result': actual_result,
            'Bet_Amount': bet_amount,
            'Balance': balance,
            'Odds': odds,
            'Payout': payout if predicted_class == actual_result else 0,
            'Predicted_Probability': prediction_prob[1]
        })

    return pd.DataFrame(bet_history), balance

# Define or import analyze_strategy function
def analyze_strategy(bet_df, initial_balance):
    total_bets = len(bet_df)
    final_balance = bet_df['Balance'].iloc[-1] if not bet_df.empty else initial_balance
    net_profit = final_balance - initial_balance
    
    return {
        'Total Bets': total_bets,
        'Final Balance': final_balance,
        'Net Profit': net_profit
    }


# Streamlit Interface
st.title("Sports Betting Dashboard")

st.sidebar.title("Betting Strategy Selector")
strategy = st.sidebar.selectbox("Select a Betting Strategy", 
                                ('Low-Risk', 'High-Risk', 'Martingale', 'Flat'))

# Add Customization for Martingale and Flat Betting Strategies
st.sidebar.title("Customization")

# Customize initial balance
initial_balance = st.sidebar.number_input("Initial Balance", value=1000, min_value=100)

# Customize initial bet for Martingale
martingale_initial_bet = st.sidebar.number_input("Martingale Initial Bet", value=10, min_value=1)

# Customize bet fraction for Flat Betting
flat_bet_fraction = st.sidebar.slider("Flat Bet Fraction", min_value=0.01, max_value=0.2, value=0.05)

# Simulate the strategies
low_risk_df, low_balance = simulate_betting_low_risk(X_test, y_test, model, initial_balance=initial_balance)
high_risk_df, high_balance = simulate_betting_high_risk(X_test, y_test, model, initial_balance=initial_balance)
martingale_df, martingale_balance = simulate_betting_martingale(X_test, y_test, model, initial_balance=initial_balance, initial_bet=martingale_initial_bet)
flat_df, flat_balance = simulate_betting_flat(X_test, y_test, model, initial_balance=initial_balance, bet_fraction=flat_bet_fraction)

low_risk_analysis = analyze_strategy(low_risk_df, initial_balance=initial_balance)
high_risk_analysis = analyze_strategy(high_risk_df, initial_balance=initial_balance)
martingale_analysis = analyze_strategy(martingale_df, initial_balance=initial_balance)
flat_analysis = analyze_strategy(flat_df, initial_balance=initial_balance)

# Display Betting Strategy Results
if strategy == 'Low-Risk':
    st.write("### Low-Risk Strategy Analysis")
    st.write(low_risk_analysis)
    st.line_chart(low_risk_df['Balance'])
    st.write("### Bet Details")
    st.write(low_risk_df[['Game', 'Predicted_Probability', 'Predicted_Class', 'Actual_Result', 'Bet_Amount', 'Odds', 'Payout']].reset_index(drop=True))

elif strategy == 'High-Risk':
    st.write("### High-Risk Strategy Analysis")
    st.write(high_risk_analysis)
    st.line_chart(high_risk_df['Balance'])
    st.write("### Bet Details")
    st.write(high_risk_df[['Game', 'Predicted_Probability', 'Predicted_Class', 'Actual_Result', 'Bet_Amount', 'Odds', 'Payout']].reset_index(drop=True))

elif strategy == 'Martingale':
    st.write("### Martingale Strategy Analysis")
    st.write(martingale_analysis)
    st.line_chart(martingale_df['Balance'])
    st.write("### Bet Details")
    st.write(martingale_df[['Game', 'Predicted_Probability', 'Predicted_Class', 'Actual_Result', 'Bet_Amount', 'Odds', 'Payout']].reset_index(drop=True))

else:
    st.write("### Flat Betting Strategy Analysis")
    st.write(flat_analysis)
    st.line_chart(flat_df['Balance'])
    st.write("### Bet Details")
    st.write(flat_df[['Game', 'Predicted_Probability', 'Predicted_Class', 'Actual_Result', 'Bet_Amount', 'Odds', 'Payout']].reset_index(drop=True))


# Additional Insights
st.sidebar.title("Insights")
st.sidebar.write("Total Bets")
if strategy == 'Low-Risk':
    st.sidebar.write(low_risk_analysis['Total Bets'])
elif strategy == 'High-Risk':
    st.sidebar.write(high_risk_analysis['Total Bets'])
elif strategy == 'Martingale':
    st.sidebar.write(martingale_analysis['Total Bets'])
else:
    st.sidebar.write(flat_analysis['Total Bets'])

st.sidebar.write("Final Balance")
if strategy == 'Low-Risk':
    st.sidebar.write(low_risk_analysis['Final Balance'])
elif strategy == 'High-Risk':
    st.sidebar.write(high_risk_analysis['Final Balance'])
elif strategy == 'Martingale':
    st.sidebar.write(martingale_analysis['Final Balance'])
else:
    st.sidebar.write(flat_analysis['Final Balance'])
