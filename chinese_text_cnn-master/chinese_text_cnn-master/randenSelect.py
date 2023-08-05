import pandas as pd
import random
import addIndextotsv as ad

def split_data_and_add_index(input_file, train_output_file, val_output_file, test_output_file, num_train_samples, num_val_samples, num_test_samples):
    # 讀取.tsv檔到DataFrame
    df = pd.read_csv(input_file, delimiter='\t')

    # 確保指定的num_train_samples不大於數據集大小
    num_train_samples = min(num_train_samples, len(df))
    # 確保指定的num_val_samples不大於數據集大小
    num_val_samples = min(num_val_samples, len(df))
    # 確保指定的num_test_samples不大於數據集大小
    num_test_samples = min(num_test_samples, len(df))

    # 隨機抽取指定數量的索引作為驗證集
    val_indices = random.sample(range(len(df)), num_val_samples)
    val_set = df.loc[val_indices]
    
    # 將驗證集從原始數據集中刪除，剩下的即為訓練集
    train_set = df.drop(index=val_indices)
    
    # 再從訓練集中隨機抽取指定數量的樣本，合併到訓練集中
    train_indices = random.sample(range(len(train_set)), num_train_samples)
    train_subset = train_set.iloc[train_indices]  # 修正此行，使用 iloc 選取指定索引的子集
    train_set = pd.concat([train_set, train_subset])
    train_set = df.loc[train_indices]

    test_set = df.drop(index=train_indices)
    # 再從剩餘的訓練集中隨機抽取指定數量的樣本，作為測試集
    remaining_indices = random.sample(range(len(test_set)), num_test_samples)
    test_indices = random.sample(remaining_indices, num_test_samples)
    test_set = df.loc[test_indices]

    # 將index列加入訓練集、驗證集和測試集
    # train_set['index'] = range(len(train_set))
    # val_set['index'] = range(len(val_set))
    # test_set['index'] = range(len(test_set))

    # 將訓練集、驗證集和測試集保存為.tsv檔
    train_set.to_csv(train_output_file, sep='\t', index=False)
    val_set.to_csv(val_output_file, sep='\t', index=False)
    test_set.to_csv(test_output_file, sep='\t', index=False)
    
    ad.addIndextotsv(train_output_file)
    ad.addIndextotsv(val_output_file)
    ad.addIndextotsv(test_output_file)
    
# 使用例子：

input_tsv_file = './data/new_trains_10000.tsv'
train_output_tsv_file = './data/self_train.tsv'
val_output_tsv_file = './data/self_dev.tsv'
test_output_tsv_file = './data/self_test.tsv'

# input_tsv_file = './data/trains_10000.tsv'
# train_output_tsv_file = './data/self_train1.tsv'
# val_output_tsv_file = './data/self_dev1.tsv'
# test_output_tsv_file = './data/self_test1.tsv'

num_train_samples_to_extract = 2000  # 訓練集隨機抽取的樣本數量
num_val_samples_to_extract = 250  # 驗證集隨機抽取的樣本數量
num_test_samples_to_extract = 20  # 測試集隨機抽取的樣本數量

split_data_and_add_index(input_tsv_file,
                         train_output_tsv_file,
                         val_output_tsv_file,
                         test_output_tsv_file,
                         num_train_samples_to_extract,
                         num_val_samples_to_extract,
                         num_test_samples_to_extract)
