import pandas as pd

phiusiil = pd.read_csv("data/phiusiil.csv")
openphish = pd.read_csv("data/openphish.csv")

print("Combining datasets...\n")
url_content = pd.concat([phiusiil, openphish], ignore_index=True)
url_content.to_csv("data/url_content.csv", index=False)
print("SAVED to data/url_content.csv")