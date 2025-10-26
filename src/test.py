import pandas as pd

df = pd.read_csv("data/phi+open.csv")

# df_label0 = df[df["label"] == 1]

# df_sample = df_label0.sample(n=48000, random_state=42) 

# df_sample.to_csv("data/label_50k.csv", index=False)

# print("Saved 50k random label 0 URLs to data/label0_50k.csv")

# df1 = pd.read_csv("data/url_content_50.csv")
# df2 = pd.read_csv("data/label_50k.csv")
# df_concat = pd.concat([df1, df2], ignore_index=True)
# df_shuffled = df_concat.sample(frac=1, random_state=42).reset_index(drop=True)
# df_shuffled.to_csv("data/url_content.csv", index=False)


df = pd.read_csv("data/url_content.csv")
label_counts = df["label"].value_counts()
print(label_counts)