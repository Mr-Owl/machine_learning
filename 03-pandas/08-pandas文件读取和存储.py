import pandas as pd

if __name__ == '__main__':
    # csv文件
    # 1.1 read_csv
    # pandas.read_csv(filepath_or_buffer, sep=',', usecols)
    # filepath_or_buffer: 文件路径
    # sep: 分隔符，默认用 "," 隔开
    # usecols: 指定读取的列名，列表形式
    data = pd.read_csv("./data/stock_day.csv", usecols=["open", "close"])
    print(data)
    print("-------------------------------")
    # 1.2 to_csv
    # DataFrame.to_csv(path_or_buf=None, sep=', ’,
    # columns=None, header=True, index=True, mode='w', encoding=None)
    # path_or_buf :文件路径
    # sep :分隔符，默认用","隔开
    # columns :选择需要的列索引
    # header :boolean or list of string, default True,是否写进列索引值
    # index:是否写进行索引
    # mode:'w'：重写, 'a' 追加
    data.to_csv("./data/test.csv", columns=["close"])
    print(pd.read_csv("./data/test.csv"))
    print("--------------------------------")
    data = pd.read_csv("./data/stock_day.csv", usecols=["open", "close"])
    data.to_csv("./data/test.csv", columns=["close"], index=False)
    print(pd.read_csv("./data/test.csv").head())
    print("--------------------------------")
    # hdf5
    # 2.1 read_hdf与to_hdf
    # HDF5文件的读取和存储需要指定一个键，值为要存储的DataFrame

    # pandas.read_hdf(path_or_buf，key =None，** kwargs)
    # 从h5文件当中读取数据
    # path_or_buffer:文件路径
    # key:读取的键  一个键以上的时候,必须传入
    # return:Theselected object
    day_close = pd.read_hdf("./data/day_close.h5")
    print(day_close.head())
    print("-------------------------------")
    # DataFrame.to_hdf(path_or_buf, key, *\kwargs*)
    day_close.to_hdf("./data/test.h5", key="day_close")
    new_data = pd.read_hdf("./data/test.h5", key="day_close")
    print(new_data.head())
    # 注意：优先选择使用HDF5文件存储
    # HDF5在存储的时候支持压缩，使用的方式是blosc，这个是速度最快的也是pandas默认支持的
    # 使用压缩可以提磁盘利用率，节省空间
    # HDF5还是跨平台的，可以轻松迁移到hadoop上面
    print("------------------------------")
    # 3 JSON
    # JSON是我们常用的一种数据交换格式，前面在前后端的交互经常用到，
    # 也会在存储的时候选择这种格式。所以我们需要知道Pandas如何进行读取和存储JSON格式。
    # 3.1 read_json
    # pandas.read_json(path_or_buf=None, orient=None, typ='frame', lines=False)
    # 将JSON格式准换成默认的Pandas DataFrame格式
    # orient : string,Indication of expected JSON string format.
    # 'split' : dict like {index -> [index], columns -> [columns], data -> [values]}
    # split 将索引总结到索引，列名到列名，数据到数据。将三部分都分开了
    # 'records' : list like [{column -> value}, ... , {column -> value}]
    # records 以columns：values的形式输出
    # 'index' : dict like {index -> {column -> value}}
    # index 以index：{columns：values}...的形式输出
    # 'columns' : dict like {column -> {index -> value}},默认该格式
    # colums 以columns:{index:values}的形式输出
    # 'values' : just the values array
    # values 直接输出值
    # lines : boolean, default False
    # 按照每行读取json对象
    # typ : default ‘frame’， 指定转换成的对象类型series或者dataframe
    data = pd.read_json("./data/Sarcasm_Headlines_Dataset.json",
                        orient="records",
                        lines=True)
    print(data.head())
    # 3.3 to_json
    # DataFrame.to_json(path_or_buf=None, orient=None, lines=False)
    # 将Pandas 对象存储为json格式
    # path_or_buf=None：文件地址
    # orient:存储的json形式，{‘split’,’records’,’index’,’columns’,’values’}
    # lines:一个对象存储为一行
    # data.to_json("./data/test.json", orient="records")  # 不按行写入,都在一行
    data.to_json("./data/test.json", orient="records", lines=True)

