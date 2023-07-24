import os
import pandas as pd
map = {'AN':0, 'DI':1, 'FE':2,'HA':3,'SA':4,'SU':5,'NE':6}
def data_label(path):

    files_dir = os.listdir(path)
    # 存放文件名和标签的列表
    path_list = []
    label_list = []
    # 遍历所有文件，取出文件名和对应的标签分别放入path_list和label_list列表
    for file_dir in files_dir:
        if os.path.splitext(file_dir)[1] == '.tiff':
            path_list.append(file_dir)
            index = os.path.splitext(file_dir)[0][3:5]
            label_list.append(map[index])
    # 将两个列表写进dataset.csv文件
    path_s = pd.Series(path_list)
    label_s = pd.Series(label_list)
    df = pd.DataFrame()
    df['path'] = path_s
    df['label'] = label_s
    df.to_csv(path + '\\dataset.csv', index=False, header=False)


def main():
    # 指定文件夹路径
    train_path = 'D:/FERNet-master/FERNet-master/datasets/cnn_train'
    val_path = 'D:/FERNet-master/FERNet-master/datasets/cnn_val'
    data_label(train_path)
    data_label(val_path)


if __name__ == '__main__':
    main()
