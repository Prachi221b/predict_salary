from glassdoor_scraper import scraper as gs
import pandas as pd

path = "C:/Windows/chromedriver.exe"  # or correct path

df = gs.get_jobs(
    'data scientist',
    1000,
    False,
    path,
    15
)

df.to_csv("glassdoor_jobs.csv", index=False)
print("CSV saved successfully")