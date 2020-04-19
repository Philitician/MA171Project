import pandas as pd

df = pd.read_csv("data/case-fatality-rate-of-covid-19-vs-median-age.csv")

features = df.columns
countries = df.Entity.unique()
n = len(countries)

ageDf = df[df["Median Age"].notna()]
df = df[df["Case fatality rate of COVID-19 (%)"].notna()]

print("Unique countries: {}".format(n))

ageDf = ageDf.drop_duplicates(["Entity"], keep="last")
print("Shape of age df: {}".format(ageDf.shape))

df = df.drop_duplicates(["Entity"], keep="last")
print("Shape of df: {}".format(df.shape))
