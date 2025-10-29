import urllib.request
import os
import zipfile
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "./dataset/sms_spam_collection.zip"
extracted_path = "./dataset/sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(
        url, zip_path, extract_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} allready exists.")
        return
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)

    print(f"File downloaded and saved as {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

df = pd.read_csv( 
 data_file_path, sep="\t", header=None, names=["Label", "Text"] 
)


### 非垃圾数据远高于垃圾数据，所以需要进行平衡
def create_balance_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df
balanced_df = create_balance_dataset(df)
print(balanced_df["Label"].value_counts())

# 将string类型的标签转换为数字1，2
balanced_df["Label"] = balanced_df["Label"].map({"ham" : 0, "spam" : 1})

# 划分数据集
def random_split(df, train_frac, val_frac, test_frac):
    train_df, tmp_df = train_test_split(
        df,
        train_size=train_frac,
        random_state=123,
        shuffle=True
    )
    val_frac = val_frac / (val_frac + test_frac)

    val_df, test_df = train_test_split(
        tmp_df,
        train_size=val_frac,
        random_state=123,
        shuffle=True
    )
    return train_df, val_df, test_df

train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1, 0.2)
train_df.to_csv("./dataset/train.csv", index=None)
val_df.to_csv("./dataset/val.csv", index=None)
test_df.to_csv("./dataset/test.csv", index=None)