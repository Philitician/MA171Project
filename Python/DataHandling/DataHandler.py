import pandas as pd
import requests
import os.path
from os import path
import re
import numpy as np

class DataHandler:
    pattern = "([A-Z])\w+"
    necessary_cols = ["TotalCases", "TotalDeaths", "TotalTests"]
    all_cols = ["Country", "TotalCases", "TotalDeaths", "TotalTests", "TotCases_1M_Pop", "Deaths_1M_pop", "Tests_1M_Pop"]
    us_cols = ["USAState", "TotalCases", "TotalDeaths", "TotalTests", "Tot_Cases_1M_Pop", "Deaths_1M_Pop", "Tests_1M_Pop"]

    def create_csv(self, url, domain):
        data = self.get_request(self, url)
        if domain == "us":
            df = self.create_us_df(self, data)
        else:
            df = self.create_all_df(self, data)

        df = self.prune(self, df, domain)

        file_path = self.create_path(self, url)
        print(file_path)
        df.to_csv(file_path, index=False)

    def get_request(self, url):
        result = requests.get(url)

        # Print status code
        print(f' GET request response (status {result.status_code}) '.center(60, '='))

        # Check status code
        if result.status_code == 200:
            print(' Success '.center(60, '-'))
        else:
            print(' Failed '.center(60, '-'))

        # Print response text
        print(result.text)
        return result.json()

    def create_all_df(self, data):
        data = data["reports"][0]["table"][0]
        df = pd.DataFrame(data)
        return df

    def create_us_df(self, data):
        data = data["data"][0]["table"]
        df = pd.DataFrame(data)

        return df

    def check_us_file(self, url, domain):
        file_path = self.create_path(self, url)
        if path.isfile(file_path):
            return pd.read_csv(file_path)
        return self.create_csv(self, url, domain)

    def create_path(self, url):
        match = re.search(self.pattern, url)
        match = match.group()
        return "data/{}.csv".format(match)

    def prune(self, df, domain):
        if domain == "all":
            df = df[self.all_cols].replace("", np.nan).dropna()
        elif domain == "us":
            df = df[self.us_cols].replace("", np.nan).dropna()

        print(df.head)
        return df