import pandas as pd

phiusiil = pd.read_csv("data_set/phiusiil.csv")
openphish = pd.read_csv("data_set/openphish.csv")

url_content = pd.concat([phiusiil, openphish], ignore_index=True)
url_content.to_csv("data_set/url_content.csv", index=False)