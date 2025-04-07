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


def load_data(login_path, exit_path):
    logins_df = pd.read_csv(login_path)
    exits_df = pd.read_csv(exit_path)
    
    logins_df['Time_utc'] = pd.to_datetime(logins_df['Time_utc'])
    exits_df['Time_utc'] = pd.to_datetime(exits_df['Time_utc'])
    
    return logins_df, exits_df

def estimate_sessions(logins_df, exits_df):
    total_sessions = len(logins_df)
    
    session_lengths = exits_df['CurrentSessionLength'].dropna()
    
    median_session_time = session_lengths.median()
    
    return total_sessions, median_session_time

def calculate_sessions_per_user_per_month(logins_df):
    logins_df['Time_utc'] = pd.to_datetime(logins_df['Time_utc'])
    logins_df['Year_Month'] = logins_df['Time_utc'].dt.to_period('M')
    
    sessions_per_user = logins_df.groupby(['Year_Month', 'pid']).size().reset_index(name='sessions')
    
    monthly_avg = sessions_per_user.groupby('Year_Month')['sessions'].mean().reset_index()
    monthly_avg['Year_Month'] = monthly_avg['Year_Month'].astype(str)
    
    return monthly_avg

def visualize_metrics(total_sessions, median_session_time, monthly_sessions_per_user):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    print("Total Number of Sessions Played:", total_sessions)
    print("Median Session Time (minutes):", median_session_time)
    
    ax.plot(monthly_sessions_per_user['Year_Month'], monthly_sessions_per_user['sessions'], 
            marker='o', linestyle='-', color='salmon', linewidth=2)
    ax.bar(monthly_sessions_per_user['Year_Month'], monthly_sessions_per_user['sessions'], 
           alpha=0.5, color='salmon')
    
    ax.set_title('Average Sessions per User per Month', fontsize=14)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Average Sessions', fontsize=12)
    plt.xticks(rotation=45)
    ax.grid(True, axis='y', alpha=0.3)
    
    for i, v in enumerate(monthly_sessions_per_user['sessions']):
        ax.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('session_metrics.png', dpi=300)
    return fig

def analyze_sessions(login_path, exit_path):
    logins_df, exits_df = load_data(login_path, exit_path)
    
    total_sessions, median_session_time = estimate_sessions(logins_df, exits_df)
    monthly_sessions_per_user = calculate_sessions_per_user_per_month(logins_df)
    
    plot = visualize_metrics(total_sessions, median_session_time, monthly_sessions_per_user)
    
    return {
        'total_sessions': total_sessions,
        'median_session_time': median_session_time,
        'monthly_sessions_per_user': monthly_sessions_per_user,
        'plot': plot
    }

def analyze_total_sessions():
    results = analyze_sessions('data/player_logged_in.csv', 'data/exited_game.csv')
    print(f"Total Sessions: {results['total_sessions']}")
    print(f"Median Session Time (minutes): {results['median_session_time']}")
    print("Average Sessions per User per Month:")
    print(results['monthly_sessions_per_user'])
    results['plot'].show()
