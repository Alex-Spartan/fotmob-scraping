import requests
from openpyxl import Workbook

def get_stats(codes=[], common_team=None):
    all_match_data = []
    
    for code in codes:
        print(f"Fetching data for match ID: {code}")
        api_url = f"https://www.fotmob.com/api/matchDetails?matchId={code}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "X-Mas": "eyJib2R5Ijp7InVybCI6Ii9hcGkvdGVhbXM/aWQ9ODUzNSZjY29kZTM9SU5EIiwiY29kZSI6MTc0MDM4MTM4ODg2MCwiZm9vIjoicHJvZHVjdGlvbjo5NjE3ZGNhYWVmOGVlMjRmOWQzZTkwMzdiYjkxOTQ2NDBiMDBmYjY0LXVuZGVmaW5lZCJ9LCJzaWduYXR1cmUiOiJGRUU5RUQ4RjNFOEQwMTg5NTc0NDNCMUM3MEExQTk4MSJ9"
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
                "Pass stats": ("Periods", "All", 3, ["Throws", "Offsides"]),
                "Defence stats": ("Periods", "All", 4, ["Keeper saves", "Tackles won"]),
                "Discipline stats": ("Periods", "All", 6, ["Yellow cards", "Red cards"]),
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
                stats = data["content"]["stats"][period_key][all_key]["stats"][index]["stats"]
                for stat in stats:
                    if stat["title"] in stat_titles:
                        if common_team == home_team:
                            match_stats[f"{stat['title']}_Home"] = stat["stats"][0]
                            match_stats[f"{stat['title']}_Away"] = stat["stats"][1]
                            print(f"{stat['title']} - {stat['stats'][0]} - {stat['stats'][1]}")
                        else:
                            match_stats[f"{stat['title']}_Home"] = stat["stats"][1]
                            match_stats[f"{stat['title']}_Away"] = stat["stats"][0]
                            print(f"{stat['title']} - {stat['stats'][1]} - {stat['stats'][0]}")

            all_match_data.append(match_stats)

        else:
            print(f"Failed to fetch data for match ID: {code}")
    
    # Create Excel file
    wb = Workbook()
    ws = wb.active
    ws.title = "Match Stats"
    
    # Set Column Headers
    if all_match_data:
        headers = list(all_match_data[0].keys())
        ws.append(headers)
    
        # Append Match Data
        for match in all_match_data:
            ws.append([match[h] for h in headers])
    
    wb.save(f"{common_team}.xlsx")
    print(f"Saved all match stats to {common_team}.xlsx")
