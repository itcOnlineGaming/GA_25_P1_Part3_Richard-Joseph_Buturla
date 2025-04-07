import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def print_unique_players():
    df = pd.read_csv('data/demographics.csv')
    unique_players = len(df['pid'].unique())
    print(f"Number of unique players: {unique_players}")

    # ------------- Active Users Daily ----------------------

def calculate_daily_active_users(df):
    df['Date'] = pd.to_datetime(df['Time_utc']).dt.date
    daily_active = df.groupby('Date')['pid'].nunique().reset_index()
    daily_active.columns = ['Date', 'Active_Users']
    return daily_active

def line_graph_daily_active(daily_active_df):
    plt.figure(figsize=(12, 6))
    plt.plot(daily_active_df['Date'], daily_active_df['Active_Users'], marker='o', linestyle='-')
    plt.title('Daily Active Users')
    plt.xlabel('Date')
    plt.ylabel('Number of Unique Active Users')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def bar_graph_daily_active(daily_active_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Date', y='Active_Users', data=daily_active_df)
    plt.title('Daily Active Users')
    plt.xlabel('Date')
    plt.ylabel('Number of Unique Active Users')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def analyze_daily_active_users():
    df = pd.read_csv('data/player_logged_in.csv')
    daily_active_df = calculate_daily_active_users(df)
    
    line_plot = line_graph_daily_active(daily_active_df)
    line_plot.savefig('daily_active_users_line.png')
    
    bar_plot = bar_graph_daily_active(daily_active_df)
    bar_plot.savefig('daily_active_users_bar.png')
    
    return daily_active_df


# ---------- Active Users Monthly ----------- 

def calculate_monthly_active_users(df):
    df['Time_utc'] = pd.to_datetime(df['Time_utc'])
    df['Month'] = df['Time_utc'].dt.to_period('M')
    monthly_active = df.groupby('Month')['pid'].nunique().reset_index()
    monthly_active['Month'] = monthly_active['Month'].astype(str)
    monthly_active.columns = ['Month', 'Active_Users']
    return monthly_active

def line_graph_monthly_active(monthly_active_df):
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_active_df['Month'], monthly_active_df['Active_Users'], marker='o', linestyle='-')
    plt.title('Monthly Active Users')
    plt.xlabel('Month')
    plt.ylabel('Number of Unique Active Users')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def bar_graph_monthly_active(monthly_active_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Month', y='Active_Users', data=monthly_active_df)
    plt.title('Monthly Active Users')
    plt.xlabel('Month')
    plt.ylabel('Number of Unique Active Users')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def analyze_monthly_active_users():
    df = pd.read_csv('data/player_logged_in.csv')
    monthly_active_df = calculate_monthly_active_users(df)
    
    line_plot = line_graph_monthly_active(monthly_active_df)
    line_plot.savefig('monthly_active_users_line.png')
    
    bar_plot = bar_graph_monthly_active(monthly_active_df)
    bar_plot.savefig('monthly_active_users_bar.png')
    
    return monthly_active_df