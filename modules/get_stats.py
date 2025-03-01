import requests
import csv
import pandas as pd

def get_stats(codes=[], common_team=None):
    all_match_data = []
    
    for code in codes:
        api_url = f"https://www.fotmob.com/api/matchDetails?matchId={code}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "X-Mas": "eyJib2R5Ijp7InVybCI6Ii9hcGkvbWF0Y2g/aWQ9NDUzNDc0NyIsImNvZGUiOjE3NDA3MjI0NDUzNzIsImZvbyI6InByb2R1Y3Rpb246OTU5OWMyYzVmNjJjMGU0NmVkNzFkNDIyYWFhM2NjOWY4ZmYyYWVkNC11bmRlZmluZWQifSwic2lnbmF0dXJlIjoiNEVFMUREMzAzQjYzRDRGRTk5RTlGNkFFQjUyQzM1RDYifQ=="
        }
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            data = response.json()

            home_team = data["header"]["teams"][0]["name"]
            away_team = data["header"]["teams"][1]["name"]
            home_score = data["header"]["teams"][0]["score"]
            away_score = data["header"]["teams"][1]["score"]
            
            print(f"{home_team} {home_score} - {away_score} {away_team}")

            # Define stat categories
            stat_categories = {
                "Top stats": ("Periods", "All", 0, ["Corners", "Fouls committed", "Shots on target"]),
                "Pass": ("Periods", "All", 3, ["Throws", "Offsides"]),
                "Defence": ("Periods", "All", 4, ["Keeper saves", "Tackles won"]),
                "Discipline": ("Periods", "All", 6, ["Yellow cards", "Red cards"]),
            }

            if common_team == home_team:
                match_stats = {
                    "Match": f"{home_team} vs {away_team}",
                    "Home Team": home_team,
                    "Away Team": away_team,
                    "Home Score": home_score,
                    "Away Score": away_score,
                }
            else: 
                match_stats = {
                    "Match": f"{home_team} vs {away_team}",
                    "Home Team": away_team,
                    "Away Team": home_team,
                    "Home Score": away_score,
                    "Away Score": home_score,
                }
            
            # Fetch stats and structure in tabular format
            for category, (period_key, all_key, index, stat_titles) in stat_categories.items():
                if period_key in data["content"]["stats"] and all_key in data["content"]["stats"][period_key] and index < len(data["content"]["stats"][period_key][all_key]["stats"]):
                    stats = data["content"]["stats"][period_key][all_key]["stats"][index]["stats"]
                stats = data["content"]["stats"][period_key][all_key]["stats"][index]["stats"]
                if stats:
                    for stat in stats:
                        if stat["title"] in stat_titles:
                            home_stat = stat["stats"][0] if stat["stats"][0] is not None else 0
                            away_stat = stat["stats"][1] if stat["stats"][1] is not None else 0
                            if common_team == home_team:
                                match_stats[f"{stat['title']}_Home"] = home_stat
                                match_stats[f"{stat['title']}_Away"] = away_stat
                            else:
                                match_stats[f"{stat['title']}_Home"] = away_stat
                                match_stats[f"{stat['title']}_Away"] = home_stat
                else:
                    print(f"{category} doesn't exists")

            all_match_data.append(match_stats)

            print(f"Fetched data for match ID: {code}")
        else:
            print(f"Failed to fetch data for match ID: {code}")

    
    df = pd.DataFrame(all_match_data)
    excel_file = f"{common_team}.xlsx"
    df.to_excel(excel_file, index=False)

    print(f"Saved all match stats to {excel_file}")