import pandas as pd
from pathlib import Path
from tqdm import tqdm
xls_path = Path("data-bin/train01/红外数据整理.xls")
image_dir = Path("data-bin/train01/images")
df  = pd.read_excel(xls_path)
# time_df = df["时间"]

# for idx,row in df.iterrows():
for idx in tqdm(range(len(df)),total=len(df),leave=False):
    time_stamp = df.iloc[idx]["时间"]
    image_name = time_stamp.strftime(r"%Y%m%d%H%M") + ".bmp"
    image_path = image_dir / image_name
    if not image_path.exists():
        print(f"Image not found: {image_path}")
    # print(f"{time_stamp.year}-{time_stamp.month}-{time_stamp.day} {time_stamp.hour}:{time_stamp.minute}:{time_stamp.second}")

