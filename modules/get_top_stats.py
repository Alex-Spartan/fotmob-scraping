import requests
from openpyxl import Workbook


def get_top_stats(codes=[]):
    all_match_data = []
    for code in codes:
        print(f"Fetching data for match ID: {code}")
        api_url = f"https://www.fotmob.com/api/matchDetails?matchId={code}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "X-Mas": "eyJib2R5Ijp7InVybCI6Ii9hcGkvbWF0Y2hEZXRhaWxzP21hdGNoSWQ9NDUwNjYzOSIsImNvZGUiOjE3NDAyNDc2NzYxMDMsImZvbyI6InByb2R1Y3Rpb246OTYxN2RjYWFlZjhlZTI0ZjlkM2U5MDM3YmI5MTk0NjQwYjAwZmI2NC11bmRlZmluZWQifSwic2lnbmF0dXJlIjoiOEY5NzVCNkQzQkY0OTczRUY2M0VDOURGMjgxMUI2NjQifQ=="
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
            
            for category, (period_key, all_key, index, stat_titles) in stat_categories.items():
                stats = data["content"]["stats"][period_key][all_key]["stats"][index]["stats"]
                for stat in stats:
                    if stat["title"] in stat_titles:
                        all_match_data.append([f"{home_team} vs {away_team}", stat["title"], stat["stats"][0], stat["stats"][1]])
                        print(f"{stat['title']} - {stat['stats'][0]} - {stat['stats'][1]}")
        else:
            print(f"Failed to fetch data for match ID: {code}")
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Match Stats"
    ws.append(["Match", "Stat", "Home Team", "Away Team"])
    
    for row in all_match_data:
        ws.append(row)
    
    wb.save("all_match_stats.xlsx")
    print("Saved all match stats to all_match_stats.xlsx")