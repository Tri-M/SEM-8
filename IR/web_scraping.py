import requests
from bs4 import BeautifulSoup
from prettytable import PrettyTable

def scrape_nobel_laureates():
    url = "https://en.wikipedia.org/wiki/List_of_Nobel_laureates"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    laureates = []
    for row in soup.select('table.wikitable tr')[1:]:
        columns = row.find_all(['td', 'th'])
        if len(columns) >= 3:
            year = columns[0].text.strip()
            winner = columns[2].text.strip()
            winner_url = columns[2].find('a')['href'] if columns[2].find('a') else None

            laureates.append({'year': year, 'winner': winner, 'winner_url': winner_url})

    return laureates

def print_table(data):
    table = PrettyTable()
    table.field_names = ["Year", "Winner Name", "URL"]

    for entry in data:
        table.add_row([entry['year'], entry['winner'], entry['winner_url']])

    print(table)

if __name__ == "__main__":
    nobel_laureates = scrape_nobel_laureates()
    print_table(nobel_laureates)
