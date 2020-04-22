from DataHandling import DataHandler
from Statistics import Inference

all_url = "https://covid19-server.chrismichael.now.sh/api/v1/AllReports"
us_url = "https://covid19-server.chrismichael.now.sh/api/v1/CasesInAllUSStates"
dh = DataHandler.DataHandler

us_data = dh.get_csv(dh, us_url, "us")
all_data = dh.get_csv(dh, all_url, "all")
print("US Data shape: {}".format(us_data.shape))
print("Global Data shape: {}".format(all_data.shape))

#lr = Inference.LinearRegression(all_data["Country"], all_data["ConfirmationRate"].values, all_data["CFR"].values)

#print(lr.mu_x)