import os
import json
import pandas as pd
from tqdm import tqdm  # pip install tqdm

class DataIngestion:
    def __init__(self, raw_data_path, output_path):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.columns = [
            'match_id', 'date', 'venue', 'batting_team', 'bowling_team', 
            'innings', 'over', 'ball', 
            'batter', 'bowler', 'runs_off_bat', 'extras', 'total_runs', 
            'is_wicket', 'is_wide', 'is_noball', 'is_legal'
        ]

    def parse_match(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        match_info = data['info']
        match_id = os.path.basename(file_path).split('.')[0]
        venue = match_info.get('venue', 'Unknown')
        dates = match_info.get('dates', ['Unknown'])
        date = dates[0] if isinstance(dates, list) else dates
        
        all_deliveries = []

        for innings_idx, innings_data in enumerate(data.get('innings', [])):
            team_batting = innings_data.get('team')
            teams = match_info.get('teams', [])
            team_bowling = 'Unknown'
            if len(teams) == 2:
                team_bowling = teams[1] if teams[0] == team_batting else teams[0]

            for over_data in innings_data.get('overs', []):
                over_num = over_data['over'] # 0-indexed (0, 1, 2...)
                
                # USER LOGIC: Over is 1-based (1, 2, 3...)
                over_display = over_num + 1
                
                # USER LOGIC: Ball counter restarts every over
                ball_counter = 1
                
                for delivery_idx, delivery in enumerate(over_data['deliveries']):
                    # Check for Extras FIRST
                    extras_data = delivery.get('extras', {})
                    is_wide = 1 if 'wides' in extras_data else 0
                    is_noball = 1 if 'noballs' in extras_data else 0
                    is_legal = 1 if (is_wide == 0 and is_noball == 0) else 0

                    # Calculate Ball Number (e.g., 0.1, 0.2)
                    # Note: We use the over_num (0-indexed) for the prefix 
                    # so Over 1 starts with 0.1, Over 20 starts with 19.1
                    ball_display = over_num + (ball_counter / 10.0)

                    # Extract Runs
                    runs = delivery.get('runs', {})
                    
                    row = {
                        'match_id': match_id,
                        'date': date,
                        'venue': venue,
                        'batting_team': team_batting,
                        'bowling_team': team_bowling,
                        'innings': innings_idx + 1,
                        'over': over_display,    # 1, 2, ... 20
                        'ball': ball_display,    # 0.1, 0.1 (wide), 0.2
                        'batter': delivery['batter'],
                        'bowler': delivery['bowler'],
                        'runs_off_bat': runs.get('batter', 0),
                        'extras': runs.get('extras', 0),
                        'total_runs': runs.get('total', 0),
                        'is_wicket': 1 if 'wickets' in delivery else 0,
                        'is_wide': is_wide,
                        'is_noball': is_noball,
                        'is_legal': is_legal
                    }
                    all_deliveries.append(row)
                    
                    # USER LOGIC: Only increment the counter if it was a LEGAL delivery
                    if is_legal:
                        ball_counter += 1
                    
        return all_deliveries

    def run(self):
        all_rows = []
        # Get list of all .json files
        if not os.path.exists(self.raw_data_path):
            print(f"Error: The folder {self.raw_data_path} does not exist.")
            return

        json_files = [f for f in os.listdir(self.raw_data_path) if f.endswith('.json')]
        
        print(f"Found {len(json_files)} matches in {self.raw_data_path}. Starting ingestion...")

        # Process files with a Progress Bar
        for file_name in tqdm(json_files, desc="Parsing Matches"):
            file_path = os.path.join(self.raw_data_path, file_name)
            try:
                match_data = self.parse_match(file_path)
                all_rows.extend(match_data)
            except Exception as e:
                print(f"Skipping {file_name}: {e}")

        # Create DataFrame
        print("Converting to DataFrame...")
        df = pd.DataFrame(all_rows, columns=self.columns)
        
        # Save to CSV
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        
        print(f"✅ Success! Ingestion complete.")
        print(f"Data saved to: {self.output_path}")
        print(f"Total Balls Processed: {df.shape[0]}")
        print(f"Total Matches Processed: {df['match_id'].nunique()}")

if __name__ == "__main__":
    # Configure paths relative to the project root
    ingestion = DataIngestion(
        raw_data_path='data/raw_json', 
        output_path='data/interim/match_data.csv'
    )
    ingestion.run()