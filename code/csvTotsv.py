import csv

def csv_to_tsv(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # 跳過表頭

        with open(output_file, 'w', newline='', encoding='utf-8') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')
            for row in csv_reader:
                label = int(row[0])
                if label == 0:
                    label = 1
                else:
                    label = 0
                text = row[1]
                tsv_writer.writerow([label, text])

if __name__ == "__main__":
    input_file = "../chinese_text_cnn-master/chinese_text_cnn-master/data/train7.csv"  # 輸入的CSV檔案名稱
    output_file = "../chinese_text_cnn-master/chinese_text_cnn-master/data/train7.tsv"  # 輸出的TSV檔案名稱
    csv_to_tsv(input_file, output_file)
