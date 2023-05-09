import os
import opencc
# 設定轉換器
converter = opencc.OpenCC('s2t.json')

# 設定原始檔案目錄和轉換後檔案儲存目錄
input_dir = 'C:/Users/ROUSER6/Desktop/THUCNews/THUCNews/text'
output_dir = 'C:/Users/ROUSER6/Desktop/THUCNews/THUCNews/transferTochinese'

# 檢查輸出目錄是否存在，若不存在就創建一個
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 轉換每個檔案
for filename in os.listdir(input_dir):
    # 檢查是否為txt檔案
    if not filename.endswith('.txt'):
        continue

    # 讀取原始檔案
    with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
        text = f.read()

    # 轉換文本
    converted_text = converter.convert(text)

    # 將轉換後的文本儲存到新檔案中
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        f.write(converted_text)
