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

# ----------------- Player Progression Analysis -----------------

import numpy as np
import seaborn as sns
from typing import Optional, List, Tuple

def load_detailed_progression_data(file_path: str, 
                                   use_cols: List[str] = ['pid', 'Time', 'CurrentJobName', 'LevelProgressionAmount']
                                   ) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(file_path, usecols=use_cols, parse_dates=['Time'])
        df = df.dropna(subset=['LevelProgressionAmount', 'CurrentJobName', 'pid'])
        df = df.sort_values(by=['pid', 'CurrentJobName', 'Time'])
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except KeyError as e:
        print(f"Error: Expected column {e} not found in the CSV.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        return None

def calculate_progression_rate(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or not all(col in df.columns for col in ['pid', 'Time', 'CurrentJobName', 'LevelProgressionAmount']):
        print("Error: DataFrame is invalid or missing required columns.")
        return None

    if df.empty:
        print("DataFrame is empty, cannot calculate rates.")
        return df

    df_sorted = df.sort_values(by=['pid', 'CurrentJobName', 'Time']).copy()
    
    df_sorted['TimeDelta'] = df_sorted.groupby(['pid', 'CurrentJobName'])['Time'].diff()
    df_sorted['ProgressionDelta'] = df_sorted.groupby(['pid', 'CurrentJobName'])['LevelProgressionAmount'].diff()

    df_sorted['TimeDeltaSeconds'] = df_sorted['TimeDelta'].dt.total_seconds()

    # Initialize rate column
    df_sorted['ProgressionRatePerSecond'] = np.nan

    # Calculate rate only where time delta is positive and progression delta is non-negative
    valid_rate_mask = (df_sorted['TimeDeltaSeconds'] > 0) & (df_sorted['ProgressionDelta'] >= 0)
    
    df_sorted.loc[valid_rate_mask, 'ProgressionRatePerSecond'] = (
        df_sorted.loc[valid_rate_mask, 'ProgressionDelta'] / df_sorted.loc[valid_rate_mask, 'TimeDeltaSeconds']
    )

    # Optional: Convert rate to per minute or per hour if seconds is too granular
    # df_sorted['ProgressionRatePerMinute'] = df_sorted['ProgressionRatePerSecond'] * 60

    return df_sorted

def analyze_progression_rates(rate_df: pd.DataFrame, 
                              group_by_col: Optional[str] = 'pid'
                              ) -> Optional[pd.DataFrame]:
                              
    if rate_df is None or 'ProgressionRatePerSecond' not in rate_df.columns:
        print("Error: DataFrame is invalid or missing 'ProgressionRatePerSecond' column.")
        return None
        
    if group_by_col and group_by_col not in rate_df.columns:
        print(f"Error: Grouping column '{group_by_col}' not found in DataFrame.")
        return None

    valid_rates = rate_df.dropna(subset=['ProgressionRatePerSecond'])
    valid_rates = valid_rates[valid_rates['ProgressionRatePerSecond'] != np.inf] 
    
    if valid_rates.empty:
        print("No valid progression rates found to analyze.")
        return pd.DataFrame()

    if group_by_col:
        agg_stats = valid_rates.groupby(group_by_col)['ProgressionRatePerSecond'].agg(['mean', 'median', 'std', 'count']).reset_index()
        agg_stats = agg_stats.rename(columns={
            'mean': f'MeanRatePerSecond', 
            'median': f'MedianRatePerSecond', 
            'std': f'StdDevRatePerSecond', 
            'count': 'NumMeasurements'
            })
        agg_stats = agg_stats.sort_values(by=f'MeanRatePerSecond', ascending=False)
    else:
         # Overall statistics if no grouping
         overall_stats = valid_rates['ProgressionRatePerSecond'].agg(['mean', 'median', 'std', 'count']).to_frame().T
         overall_stats = overall_stats.rename(columns={
            'mean': f'MeanRatePerSecond', 
            'median': f'MedianRatePerSecond', 
            'std': f'StdDevRatePerSecond', 
            'count': 'NumMeasurements'
            })
         agg_stats = overall_stats
         
    return agg_stats


def plot_rate_distribution(rate_df: pd.DataFrame, 
                           rate_column: str = 'ProgressionRatePerSecond',
                           quantile_cutoff: float = 0.99
                           ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
                           
    if rate_df is None or rate_column not in rate_df.columns:
        print(f"Error: DataFrame is invalid or missing '{rate_column}' column.")
        return None

    valid_rates = rate_df[rate_column].dropna()
    valid_rates = valid_rates[valid_rates != np.inf]
    
    if valid_rates.empty:
        print("No valid rates to plot.")
        return None

    cutoff_value = valid_rates.quantile(quantile_cutoff)
    plot_data = valid_rates[valid_rates <= cutoff_value]
    
    if plot_data.empty:
        print(f"No data left after applying quantile cutoff {quantile_cutoff}. Check data or cutoff.")
        # Optionally plot all data if filtering removed everything
        plot_data = valid_rates 
        title_suffix = "(All Data)"
    else:
         title_suffix = f"(Excluding Top {(1-quantile_cutoff)*100:.1f}%)"


    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(plot_data, kde=True, ax=ax[0])
    ax[0].set_title(f'Distribution of {rate_column} {title_suffix}')
    ax[0].set_xlabel(f'{rate_column}')
    ax[0].set_ylabel('Frequency')

    sns.boxplot(x=plot_data, ax=ax[1])
    ax[1].set_title(f'Box Plot of {rate_column} {title_suffix}')
    ax[1].set_xlabel(f'{rate_column}')

    plt.tight_layout()
    return fig, ax

def plot_average_rate_per_group(aggregated_stats_df: pd.DataFrame, 
                                group_col: str, 
                                rate_col: str = 'MeanRatePerSecond', 
                                top_n: Optional[int] = 20
                                ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
                                
    if aggregated_stats_df is None or group_col not in aggregated_stats_df.columns or rate_col not in aggregated_stats_df.columns:
        print("Error: DataFrame is invalid or missing required columns.")
        return None
        
    plot_df = aggregated_stats_df.sort_values(by=rate_col, ascending=False).copy()

    if top_n and len(plot_df) > top_n:
        plot_df = plot_df.head(top_n)
        title = f'Top {top_n} Average {rate_col} per {group_col}'
    else:
        title = f'Average {rate_col} per {group_col}'

    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.3)))
    sns.barplot(x=rate_col, y=group_col, data=plot_df, ax=ax, palette='viridis')
    ax.set_title(title)
    ax.set_xlabel(f'Average {rate_col}')
    ax.set_ylabel(group_col.replace('Current','').replace('Name','')) # Clean up label
    plt.tight_layout()
    return fig, ax
    
def plot_player_job_progression_curve(df: pd.DataFrame, 
                                      player_id: str, 
                                      job_name: str
                                      ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
                                      
    if df is None or not all(col in df.columns for col in ['pid', 'Time', 'CurrentJobName', 'LevelProgressionAmount']):
        print("Error: DataFrame is invalid or missing required columns.")
        return None
        
    player_job_data = df[(df['pid'] == player_id) & (df['CurrentJobName'] == job_name)].sort_values(by='Time')
    
    if player_job_data.empty:
        print(f"No data found for Player '{player_id}' and Job '{job_name}'.")
        return None
        
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(player_job_data['Time'], player_job_data['LevelProgressionAmount'], marker='.', linestyle='-')
    ax.set_title(f'Progression Curve for Player {player_id} on Job {job_name}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Level Progression Amount')
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, ax

# Add this function to your player_analysis.py script
# (Make sure the other functions like load_detailed_progression_data, 
# calculate_progression_rate, etc., are also in the same file)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple # Ensure this is imported at the top

# ... (keep all your other functions like load_detailed_progression_data, etc.) ...

def run_full_progression_rate_analysis(
    file_path: str, 
    use_cols: List[str] = ['pid', 'Time', 'CurrentJobName', 'LevelProgressionAmount'],
    top_n_players: Optional[int] = 20, 
    top_n_jobs: Optional[int] = 20, 
    rate_dist_cutoff: float = 0.99
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    
    print(f"Starting Full Progression Rate Analysis for: {file_path}")
    
    # --- Step 1: Load Data ---
    print("-" * 30)
    print(f"Loading data...")
    detail_data = load_detailed_progression_data(file_path, use_cols=use_cols)
    
    if detail_data is None:
        print("Failed to load data. Stopping analysis.")
        return None
    print(f"Successfully loaded {len(detail_data)} records.")
    
    # --- Step 2: Calculate Rates ---
    print("-" * 30)
    print("Calculating progression rates...")
    rate_data = calculate_progression_rate(detail_data) 
    
    if rate_data is None:
        print("Failed to calculate progression rates. Stopping analysis.")
        return None
    if rate_data.empty or rate_data['ProgressionRatePerSecond'].isnull().all():
         print("Rate calculation resulted in empty or all-null rates. Stopping analysis.")
         return None
    print("Progression rates calculated.")

    # --- Step 3: Analyze and Visualize Rates ---
    print("-" * 30)
    # --- 3a: Distribution ---
    print("Plotting distribution of progression rates...")
    fig_rate_dist, ax_rate_dist = plot_rate_distribution(rate_data, quantile_cutoff=rate_dist_cutoff)
    if fig_rate_dist:
        plt.show()
    else:
        print("Skipping rate distribution plot.")
        
    # --- 3b: Player Stats ---
    print("-" * 30)
    print("Analyzing average progression rate per player...")
    player_rate_stats = analyze_progression_rates(rate_data, group_by_col='pid')
    
    if player_rate_stats is not None and not player_rate_stats.empty:
        print("Player rate statistics calculated.")
        print("Plotting average progression rate per player...")
        fig_player_rate, ax_player_rate = plot_average_rate_per_group(
            player_rate_stats, 
            group_col='pid', 
            rate_col='MeanRatePerSecond', 
            top_n=top_n_players
        )
        if fig_player_rate:
            plt.show()
        else:
             print("Skipping player rate plot.")
    else:
        print("Could not calculate player rate statistics or data was empty.")
        player_rate_stats = pd.DataFrame() # Ensure it's an empty df

    # --- 3c: Job Stats ---
    print("-" * 30)
    print("Analyzing average progression rate per job...")
    job_rate_stats = analyze_progression_rates(rate_data, group_by_col='CurrentJobName')

    if job_rate_stats is not None and not job_rate_stats.empty:
        print("Job rate statistics calculated.")
        print("Plotting average progression rate per job...")
        fig_job_rate, ax_job_rate = plot_average_rate_per_group(
            job_rate_stats, 
            group_col='CurrentJobName', 
            rate_col='MeanRatePerSecond', 
            top_n=top_n_jobs
        )
        if fig_job_rate:
             plt.show()
        else:
             print("Skipping job rate plot.")
    else:
         print("Could not calculate job rate statistics or data was empty.")
         job_rate_stats = pd.DataFrame() # Ensure it's an empty df

    # --- 3d: Example Curve ---
    print("-" * 30)
    print("Generating example progression curve...")
    # Use valid data for selecting example
    valid_rate_data = rate_data.dropna(subset=['ProgressionRatePerSecond'])
    if player_rate_stats is not None and not player_rate_stats.empty and not valid_rate_data.empty:
         # Find player from stats who exists in the data with valid rates
         valid_pids_in_stats = player_rate_stats[player_rate_stats['pid'].isin(valid_rate_data['pid'].unique())]
         if not valid_pids_in_stats.empty:
             example_pid = valid_pids_in_stats.iloc[0]['pid'] 
             example_jobs = detail_data[detail_data['pid'] == example_pid]['CurrentJobName'].unique()
             if len(example_jobs) > 0:
                 example_job_name = example_jobs[0]
                 print(f"Plotting curve for Player '{example_pid}' on Job '{example_job_name}'...")
                 fig_curve, ax_curve = plot_player_job_progression_curve(detail_data, example_pid, example_job_name)
                 if fig_curve:
                      plt.show()
                 else:
                      print("Skipping example curve plot.")
             else:
                 print(f"Could not find job data for example player {example_pid}.")
         else:
              print("Could not find valid example player in stats to plot curve.")
    else:
        print("Skipping example curve plot due to missing stats or rate data.")

    print("-" * 30)
    print("Analysis complete.")
    
    return player_rate_stats, job_rate_stats

def analyze_peak_play_times(
    file_paths: List[str],
    timestamp_cols: List[str] = ['Time_utc', 'Time'] 
    ) -> Optional[pd.Series]:
    
    all_timestamps = []
    print(f"Analyzing peak times from {len(file_paths)} file(s)...")

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path, low_memory=False)
            
            col_to_use = None
            for col in timestamp_cols:
                if col in df.columns:
                    col_to_use = col
                    break
            
            if col_to_use is None:
                print(f"Warning: No suitable timestamp column {timestamp_cols} found in {file_path}. Skipping.")
                continue

            timestamps = pd.to_datetime(df[col_to_use], errors='coerce')
            timestamps = timestamps.dropna()
            all_timestamps.append(timestamps)

        except Exception as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}. Skipping.")

    if not all_timestamps:
        print("Error: No valid timestamps could be extracted from any provided files.")
        return None

    combined_timestamps = pd.concat(all_timestamps, ignore_index=True)
    
    if combined_timestamps.empty:
        print("Error: Combined timestamps list is empty after processing files.")
        return None
        
    if pd.api.types.is_datetime64_any_dtype(combined_timestamps) and combined_timestamps.dt.tz is None:
         try:
             combined_timestamps = combined_timestamps.dt.tz_localize('UTC')
             print("Info: Assuming timestamps are UTC and localized.")
         except TypeError:
              print("Warning: Could not automatically localize timestamps to UTC. Using naive hour extraction.")
              pass

    hours = combined_timestamps.dt.hour
    hourly_counts = hours.value_counts().sort_index()
    
    hourly_counts = hourly_counts.reindex(range(24), fill_value=0)

    print("Plotting hourly activity...")
    plt.figure(figsize=(12, 6))
    sns.barplot(x=hourly_counts.index, y=hourly_counts.values, color='skyblue')
    plt.title('Player Activity by Hour of Day (UTC)')
    plt.xlabel('Hour of Day (UTC)')
    plt.ylabel('Number of Recorded Events')
    plt.xticks(range(24))
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

    return hourly_counts