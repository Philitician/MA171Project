from DataHandling import DataHandler

all_url = "https://covid19-server.chrismichael.now.sh/api/v1/AllReports"
us_url = "https://covid19-server.chrismichael.now.sh/api/v1/CasesInAllUSStates"
dh = DataHandler.DataHandler

us_data = dh.check_csv(dh, us_url, "us")

print(us_data.shape)
print(us_data.columns)

all_data = dh.check_csv(dh, all_url, "all")

print(all_data.shape)
print(all_data.columns)