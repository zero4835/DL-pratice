import pandas as pd

# 讀取tsv檔案
df = pd.read_csv('./data/train3.tsv', sep='\t')

# 添加一個index欄位，即row number
df.insert(0, 'index', range(len(df)))

# 過濾掉一些不必要的文本信息
df['text'] = df['text'].str.replace('@[^\s]+', '')  # 過濾掉@提及
df['text'] = df['text'].str.replace('\[.*?\]', '')  # 過濾掉[表情符號]

# 將處理後的資料存回新的tsv檔
df.to_csv('./data/train5.tsv', sep='\t', index=False)
