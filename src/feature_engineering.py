import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class FeatureEngineering:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def _calculate_weights(self, df):
        """
        Assigns a weight to each row based on recency.
        """
        current_year = df['date'].dt.year.max()
        
        def get_weight(row_year):
            if row_year >= current_year - 3:
                return 1.0
            else:
                decay = 0.8 ** (current_year - 3 - row_year)
                return max(decay, 0.1)
        
        return df['date'].dt.year.apply(get_weight)

    def process(self):
        print("Loading interim data...")
        df = pd.read_csv(self.input_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. FILTERING
        df = df[df['innings'] <= 2]
        
        # CRITICAL: Sort ensure the order for .values assignment later
        df = df.sort_values(['match_id', 'innings', 'over', 'ball'])
        
        print("Generating State Features...")
        
        # 2. MATCH STATE FEATURES
        group = df.groupby(['match_id', 'innings'])
        
        df['current_score'] = group['total_runs'].cumsum()
        df['wickets_fallen'] = group['is_wicket'].cumsum()
        df['legal_balls_bowled'] = group['is_legal'].cumsum()
        
        df['balls_left'] = 120 - df['legal_balls_bowled']
        df['wickets_left'] = 10 - df['wickets_fallen']
        
        df['crr'] = (df['current_score'] * 6) / df['legal_balls_bowled']
        df['crr'] = df['crr'].fillna(0)
        df.loc[np.isinf(df['crr']), 'crr'] = 0

        # 3. MOMENTUM FEATURES (FIXED)
        print("Calculating Momentum (Rolling Windows)...")
        
        # We use .values here to avoid Index Mismatch errors.
        # Since 'df' is sorted and 'group' respects that order, this aligns perfectly.
        df['runs_last_5'] = group['total_runs'].rolling(window=30, min_periods=1).sum().values
        df['wickets_last_5'] = group['is_wicket'].rolling(window=30, min_periods=1).sum().values

        # 4. TARGET VARIABLE
        total_scores = group['total_runs'].sum().reset_index()
        total_scores = total_scores.rename(columns={'total_runs': 'final_score'})
        
        df = df.merge(total_scores, on=['match_id', 'innings'], how='left')
        
        # 5. DATA CLEANING (Rain Rules)
        final_balls = group['is_legal'].count().reset_index().rename(columns={'is_legal': 'total_balls'})
        final_wickets = group['is_wicket'].sum().reset_index().rename(columns={'is_wicket': 'total_wickets'})
        
        validation_df = final_balls.merge(final_wickets, on=['match_id', 'innings'])
        
        valid_innings = validation_df[
            (validation_df['total_balls'] > 60) | (validation_df['total_wickets'] == 10)
        ]
        
        valid_ids = set(zip(valid_innings['match_id'], valid_innings['innings']))
        df['id_tuple'] = list(zip(df['match_id'], df['innings']))
        df = df[df['id_tuple'].isin(valid_ids)].drop(columns=['id_tuple'])

        # 6. RECENCY WEIGHTS
        print("Applying Recency Weights...")
        df['sample_weight'] = self._calculate_weights(df)

        # 7. FINAL SELECTION
        output_columns = [
            'match_id', 'venue', 'batting_team', 'bowling_team', 'innings',
            'ball', 'legal_balls_bowled', 'wickets_left', 'balls_left', 
            'current_score', 'crr', 'runs_last_5', 'wickets_last_5', 
            'sample_weight', 'final_score'
        ]
        
        final_df = df[output_columns]
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        final_df.to_csv(self.output_path, index=False)
        print(f"✅ Success! Feature Engineering Complete.")
        print(f"Training Data Shape: {final_df.shape}")
        print(f"Saved to: {self.output_path}")

if __name__ == "__main__":
    fe = FeatureEngineering(
        input_path='data/interim/match_data.csv',
        output_path='data/processed/train_data.csv'
    )
    fe.process()