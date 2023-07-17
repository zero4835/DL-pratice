import pandas as pd

# 讀取 TSV 檔案
df = pd.read_csv('./data/train7.tsv', sep='\t')

# 輸出資料框的前幾行，以便檢查是否成功讀取 'text' 欄位
print(df.head())

# 添加一個 index 欄位，即 row number
df.insert(0, 'index', range(len(df)))

# 過濾掉一些不必要的文本信息
# if 'text' in df:
#     df['text'] = df['text'].str.replace('@[^\s]+', '')  # 過濾掉 @ 提及
#     df['text'] = df['text'].str.replace('\[.*?\]', '')  # 過濾掉 [表情符號]

# 將處理後的資料存回新的 TSV 檔
df.to_csv('./data/train7.tsv', sep='\t', index=False)
