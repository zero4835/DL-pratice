import csv
from opencc import OpenCC

# 建立 OpenCC 轉換器物件，將簡體中文轉換為繁體中文
cc = OpenCC('s2t')

# 指定輸入和輸出的 CSV 檔案名稱
input_file = 'C:/Users/ROUSER6/Desktop/DEEP_LEARNING_Pratice/ChineseNlpCorpus/weibo_senti_100k/weibo_senti_100k/weibo_senti_100k.csv'
output_file = 'C:/Users/ROUSER6/Desktop/DEEP_LEARNING_Pratice/ChineseNlpCorpus/weibo_senti_100k/weibo_senti_100k/chinese.csv'

# 開啟輸入檔案，讀取 CSV 資料
with open(input_file, 'r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    # 讀取 CSV 的標題列
    header = next(reader)
    # 將標題列中的簡體中文轉換為繁體中文
    header = [cc.convert(item) for item in header]

    # 開啟輸出檔案，寫入 CSV 資料
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        # 將轉換後的標題列寫入輸出檔案
        writer.writerow(header)
        
        # 將輸入檔案的每一列資料逐一轉換並寫入輸出檔案
        for row in reader:
            row = [cc.convert(item) for item in row]
            writer.writerow(row)
