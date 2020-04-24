import numpy as np
from DataHandling import DataHandler
from Statistics.LinearRegression import LinearRegression
from Statistics.LinearRegressionCentered import LinearRegressionCentered


all_url = "https://covid19-server.chrismichael.now.sh/api/v1/AllReports"
us_url = "https://covid19-server.chrismichael.now.sh/api/v1/CasesInAllUSStates"
dh = DataHandler.DataHandler

us_data = dh.get_csv(dh, us_url, "us")
all_data = dh.get_csv(dh, all_url, "all")
print("US Data shape: {}".format(us_data.shape))
print("Global Data shape: {}".format(all_data.shape))

# quite decent
LinearRegression("USAState", us_data["ConfirmationRate"].values, us_data["Deaths_1M_Pop"].values, "ConfirmationRate", "Deaths_1M_Pop")
LinearRegression("USAState", us_data["Tot_Cases_1M_Pop"].values, us_data["ConfirmationRate"].values, "Tot_Cases_1M_Pop", "ConfirmationRate")
LinearRegression("USAState", us_data["Tot_Cases_1M_Pop"].values, us_data["Deaths_1M_Pop"].values, "Tot_Cases_1M_Pop", "Deaths_1M_Pop")
