import pandas as pd
import random
import addIndextotsv as ad

def split_data_and_add_index(input_file, train_output_file, val_output_file, test_output_file, num_train_samples, num_val_samples, num_test_samples):
    # 讀取.tsv檔到DataFrame
    df = pd.read_csv(input_file, delimiter='\t')

    # 隨機抽取指定數量的索引作為驗證集
    val_indices = random.sample(range(len(df)), num_val_samples)
    # 将剩余的索引用于训练和测试集
    remaining_indices = list(set(range(len(df))) - set(val_indices))

    # 隨機抽取指定數量的索引作為訓練集
    train_indices = random.sample(remaining_indices, num_train_samples)
    # 将剩余的索引用于测试集
    test_indices = list(set(remaining_indices) - set(train_indices))[:num_test_samples]

    # 根据索引提取数据集的子集
    train_set = df.loc[train_indices]
    val_set = df.loc[val_indices]
    test_set = df.loc[test_indices]

    # 將訓練集、驗證集和測試集保存為.tsv檔
    train_set.to_csv(train_output_file, sep='\t', index=False)
    val_set.to_csv(val_output_file, sep='\t', index=False)
    test_set.to_csv(test_output_file, sep='\t', index=False)
    
    ad.addIndextotsv(train_output_file)
    ad.addIndextotsv(val_output_file)
    ad.addIndextotsv(test_output_file)

# 使用例子：
input_tsv_file = './data/new_trains.tsv'
train_output_tsv_file = './data/self_train.tsv'
val_output_tsv_file = './data/self_dev.tsv'
test_output_tsv_file = './data/self_test.tsv'

num_train_samples_to_extract = 5000  # 訓練集隨機抽取的樣本數量
num_val_samples_to_extract = 900  # 驗證集隨機抽取的樣本數量
num_test_samples_to_extract = 100  # 測試集隨機抽取的樣本數量

split_data_and_add_index(input_tsv_file,
                         train_output_tsv_file,
                         val_output_tsv_file,
                         test_output_tsv_file,
                         num_train_samples_to_extract,
                         num_val_samples_to_extract,
                         num_test_samples_to_extract)
