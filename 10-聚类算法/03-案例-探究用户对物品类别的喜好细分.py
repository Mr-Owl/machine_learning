import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main():
    # 1 数据获取
    order_product = pd.read_csv("./data/instacart/order_products__prior.csv")
    products = pd.read_csv("./data/instacart/products.csv")
    orders = pd.read_csv("./data/instacart/orders.csv")
    aisles = pd.read_csv("./data/instacart/aisles.csv")
    # 2 数据基本处理
    # 2.1 表格合并
    table1 = pd.merge(order_product, products, on=["product_id", "product_id"])
    table2 = pd.merge(table1, orders, on=["order_id", "order_id"])
    table = pd.merge(table2, aisles, on=["aisle_id", "aisle_id"])
    # 2.2 交叉表合并
    data = pd.crosstab(table["user_id"], table["aisle"])
    print(data.shape)
    print(data.head())
    # 2.3 数据截取(电脑配置足够不需要截取)
    new_data = data[:1000]
    # 3 特征工程 - pca
    transfer = PCA(n_components=0.9)
    trans_data = transfer.fit_transform(new_data)
    # 4 机器学习
    estimators = KMeans(n_clusters=5)
    y_pre = estimators.fit_predict(trans_data)
    # 5 模型评估
    print(silhouette_score(trans_data, y_pre))


if __name__ == '__main__':
    main()
