import opencc

# 创建简繁转换器
converter = opencc.OpenCC('s2t.json')  # 's2t.json'是转换规则，表示从简体到繁体

# 读取原始文件
file_path = r'C:\Users\ROUSER6\Desktop\DEEP_LEARNING_Pratice\chinese_text_cnn-master\chinese_text_cnn-master\data\train.tsv'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 进行简繁转换
converted_content = converter.convert(content)

# 将转换后的内容写入新的文件
output_file_path = r'C:\Users\ROUSER6\Desktop\DEEP_LEARNING_Pratice\chinese_text_cnn-master\chinese_text_cnn-master\data\train_traditional.tsv'
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(converted_content)
