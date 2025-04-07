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

# ------------ Stickiness -------------------------

def calculate_stickiness(csv_path):
    df = pd.read_csv(csv_path)
    df['Time_utc'] = pd.to_datetime(df['Time_utc'])
    
    df['Date'] = df['Time_utc'].dt.date
    df['Year_Month'] = df['Time_utc'].dt.to_period('M')
    
    daily_active = df.groupby(['Year_Month', 'Date'])['pid'].nunique().reset_index()
    
    monthly_active = df.groupby('Year_Month')['pid'].nunique().reset_index()
    monthly_active.columns = ['Year_Month', 'MAU']
    
    avg_dau = daily_active.groupby('Year_Month')['pid'].mean().reset_index()
    avg_dau.columns = ['Year_Month', 'Average_DAU']
    
    stickiness_df = pd.merge(avg_dau, monthly_active, on='Year_Month')
    
    stickiness_df['Stickiness'] = stickiness_df['Average_DAU'] / stickiness_df['MAU']
    
    stickiness_df['Month'] = stickiness_df['Year_Month'].astype(str)
    
    return stickiness_df

def plot_stickiness(stickiness_df):
    plt.figure(figsize=(12, 8))
    
    ax = plt.subplot(111)
    bars = ax.bar(stickiness_df['Month'], stickiness_df['Stickiness'], color='skyblue', alpha=0.7)
    
    ax.plot(stickiness_df['Month'], stickiness_df['Stickiness'], 'o-', color='darkblue', linewidth=2)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title('Player Stickiness (DAU/MAU) by Month', fontsize=15)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Stickiness (DAU/MAU)', fontsize=12)
    plt.ylim(0, min(1, stickiness_df['Stickiness'].max() * 1.2))  
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.text(0, 1.02, 'Maximum (1.0)', color='r', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('player_stickiness.png', dpi=300)
    
    return plt

def analyze_stickiness():
    csv_path = 'data/player_logged_in.csv'
    stickiness_df = calculate_stickiness(csv_path)
    plot = plot_stickiness(stickiness_df)
    return stickiness_df, plot
