import requests
from openpyxl import Workbook

def get_top_stats(codes=[]):
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

            top_stats = data["content"]["stats"]["Periods"]["All"]["stats"][0]["stats"]
            stat_titles = [
                "Corners",
                "Fouls committed",
                "Shots on target",
            ]

            pass_stats = data["content"]["stats"]["Periods"]["All"]["stats"][3]["stats"]
            pass_titles = [
                "Throws",
                "Offsides",
            ]
            defence_stats = data["content"]["stats"]["Periods"]["All"]["stats"][4]["stats"]
            defence_titles = [
                "Keeper saves",
                "Tackles won",
            ]
            discipline_stats = data["content"]["stats"]["Periods"]["All"]["stats"][6]["stats"]
            discipline_titles = [
                "Yellow cards",
                "Red cards",
            ]
            wb = Workbook()
            ws = wb.active

            ws.title = home_team  # Change to away_team if you want the away team as the title

            ws.append(["Stat", home_team, away_team])

            print("Top stats")

            for stat in top_stats:
                if stat["title"] in stat_titles:
                    ws.append([stat["title"], stat["stats"][0], stat["stats"][1]])
                    print("Stat: ", stat["title"])


            print("Pass stats")
            for stat in pass_stats:
                if stat["title"] in pass_titles:
                    ws.append([stat["title"], stat["stats"][0], stat["stats"][1]])
                    print("Stat: ", stat["title"])

            print("Defence stats")

            for stat in defence_stats:
                if stat["title"] in defence_titles:
                    ws.append([stat["title"], stat["stats"][0], stat["stats"][1]])
                    print("Stat: ", stat["title"])


            print("Discipline stats")
            for stat in discipline_stats:
                if stat["title"] in discipline_titles:
                    ws.append([stat["title"], stat["stats"][0], stat["stats"][1]])


            # Save the workbook
            wb.save(f"{home_team}_vs_{away_team}_stats.xlsx")
            print(f"Saved stats to {home_team}_vs_{away_team}_stats.xlsx\n")

        else:
            print("Failed to fetch data")