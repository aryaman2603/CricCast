import os
import json
import pandas as pd
from tqdm import tqdm

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

    def _normalize_teams(self, df):
        """Standardizes IPL team names to their modern equivalents."""
        team_map = {
            # Modern Rebrands
            'Delhi Daredevils': 'Delhi Capitals',
            'Kings XI Punjab': 'Punjab Kings',
            'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
            'Rising Pune Supergiant': 'Rising Pune Supergiants', # Normalize spelling
            
            # Franchise Continuity (Optional - Use caution here)
            # 'Deccan Chargers': 'Sunrisers Hyderabad', # Usually treated as same legacy
            # 'Pune Warriors': 'Rising Pune Supergiants' # Different owners, maybe keep separate?
            # For now, let's keep defunct teams distinct unless they are direct rebrands
        }
        df['batting_team'] = df['batting_team'].replace(team_map)
        df['bowling_team'] = df['bowling_team'].replace(team_map)
        return df

    def _normalize_venues(self, df):
        """Merges duplicate venue names and handles renames."""
        venue_map = {
            # --- Format Fixes (Removing City Names) ---
            'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',
            'Brabourne Stadium, Mumbai': 'Brabourne Stadium',
            'Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy',
            'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
            'Eden Gardens, Kolkata': 'Eden Gardens',
            'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'Himachal Pradesh Cricket Association Stadium',
            'M Chinnaswamy Stadium, Bengaluru': 'M. Chinnaswamy Stadium',
            'M Chinnaswamy Stadium': 'M. Chinnaswamy Stadium', # Normalize dot
            'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium',
            'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium',
            'Maharashtra Cricket Association Stadium, Pune': 'Maharashtra Cricket Association Stadium',
            'Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh': 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
            'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
            'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Punjab Cricket Association IS Bindra Stadium',
            'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium', # Old name mapped to new
            'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium',
            'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium',
            'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium',
            'Wankhede Stadium, Mumbai': 'Wankhede Stadium',
            'Zayed Cricket Stadium, Abu Dhabi': 'Sheikh Zayed Stadium',
            
            # --- Historic Renames ---
            'Feroz Shah Kotla': 'Arun Jaitley Stadium',
            'Sardar Patel Stadium, Motera': 'Narendra Modi Stadium',
            'Narendra Modi Stadium, Ahmedabad': 'Narendra Modi Stadium',
            'Subrata Roy Sahara Stadium': 'Maharashtra Cricket Association Stadium',
            'Vidarbha Cricket Association Stadium, Jamtha': 'Vidarbha Cricket Association Stadium',
        }
        df['venue'] = df['venue'].replace(venue_map)
        return df

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
                over_num = over_data['over'] # 0-indexed
                over_display = over_num + 1
                ball_counter = 1
                
                for delivery_idx, delivery in enumerate(over_data['deliveries']):
                    extras_data = delivery.get('extras', {})
                    is_wide = 1 if 'wides' in extras_data else 0
                    is_noball = 1 if 'noballs' in extras_data else 0
                    is_legal = 1 if (is_wide == 0 and is_noball == 0) else 0

                    ball_display = over_num + (ball_counter / 10.0)
                    runs = delivery.get('runs', {})
                    
                    row = {
                        'match_id': match_id,
                        'date': date,
                        'venue': venue,
                        'batting_team': team_batting,
                        'bowling_team': team_bowling,
                        'innings': innings_idx + 1,
                        'over': over_display,
                        'ball': ball_display,
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
                    
                    if is_legal:
                        ball_counter += 1
                    
        return all_deliveries

    def run(self):
        all_rows = []
        if not os.path.exists(self.raw_data_path):
            print(f"Error: The folder {self.raw_data_path} does not exist.")
            return

        json_files = [f for f in os.listdir(self.raw_data_path) if f.endswith('.json')]
        print(f"Found {len(json_files)} matches. Parsing...")

        for file_name in tqdm(json_files):
            file_path = os.path.join(self.raw_data_path, file_name)
            try:
                match_data = self.parse_match(file_path)
                all_rows.extend(match_data)
            except Exception as e:
                print(f"Skipping {file_name}: {e}")

        df = pd.DataFrame(all_rows, columns=self.columns)
        
        # --- CLEANING STEP ---
        print("Normalizing Team and Venue names...")
        df = self._normalize_teams(df)
        df = self._normalize_venues(df)
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"✅ Ingestion Complete. Data saved to {self.output_path}")

if __name__ == "__main__":
    ingestion = DataIngestion(
        raw_data_path='data/raw_json', 
        output_path='data/interim/match_data.csv'
    )
    ingestion.run()