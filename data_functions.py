import pandas as pd


def print_unique_players():
    df = pd.read_csv('data/demographics.csv')
    unique_players = len(df['pid'].unique())
    print(f"Number of unique players: {unique_players}")