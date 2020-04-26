from DataHandling import DataHandler
from Statistics.LinearRegression import LinearRegression

all_url = "https://covid19-server.chrismichael.now.sh/api/v1/AllReports"
us_url = "https://covid19-server.chrismichael.now.sh/api/v1/CasesInAllUSStates"
dh = DataHandler.DataHandler



us_data = dh.get_csv(dh, us_url, "us")
all_data = dh.get_csv(dh, all_url, "all")
print("US Data shape: {}".format(us_data.shape))
print("Global Data shape: {}".format(all_data.shape))

theta = 0.025

lr = LinearRegression(us_data["ConfirmationRate"].values, us_data["Deaths_1M_Pop"].values, "USAState", "ConfirmationRate", "Deaths_1M_Pop")
lr.plotFrequency()
lr.plotCredibility(lr.Interval(theta))
lr.plotCredibility(lr.Interval(theta, 1), pred=True)


lr = LinearRegression(us_data["Tot_Cases_1M_Pop"].values, us_data["ConfirmationRate"].values, "USAState", "Tot_Cases_1M_Pop", "ConfirmationRate")
lr.plotCredibility(lr.Interval(theta))
lr.plotCredibility(lr.Interval(theta, 1), pred=True)

lr = LinearRegression(us_data["Tot_Cases_1M_Pop"].values, us_data["Deaths_1M_Pop"].values, "USAState", "Tot_Cases_1M_Pop", "Deaths_1M_Pop")
lr.plotCredibility(lr.Interval(theta))
lr.plotCredibility(lr.Interval(theta, 1), pred=True)
