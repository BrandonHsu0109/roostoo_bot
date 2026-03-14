import pandas as pd

df = pd.read_csv(r"logs\ticker_history.csv")
df["tmin"] = (df["ts"] // 60000) * 60000

mins = sorted(df["tmin"].unique())
cut = int(len(mins) * 0.7)

train_mins = set(mins[:cut])
test_mins = set(mins[cut:])

train = df[df["tmin"].isin(train_mins)].drop(columns=["tmin"])
test = df[df["tmin"].isin(test_mins)].drop(columns=["tmin"])

train.to_csv(r"logs\ticker_train.csv", index=False)
test.to_csv(r"logs\ticker_test.csv", index=False)

print("train minutes:", len(train_mins), "rows:", len(train))
print("test minutes:", len(test_mins), "rows:", len(test))