import pandas as pd
from sklearn.datasets import load_iris,fetch_20newsgroups,load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def knncls():
    data = pd.read_csv('./data/FBlocation/train.csv')

    print(data)

    data = data.query('x > 1.0 & x < 1.25 & y>2.5 & y<2.75')
    
    #处理时间
    time_value = pd.to_datetime(data['time'],unit='s')
    
    #日期格式转为字典格式
    time_value = pd.DatetimeIndex(time_value)
    data['day'] = time_value.day
    data['weekday'] = time_value.weekday
    data['hour'] = time_value.hour
    #删除时间
    data.drop(['time'],axis=1)

    #签到数量少于n个目标位置的删除
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 3].reset_index() 

    data = data[data['place_id'].isin(tf.place_id)]
    y = data['place_id']
    x = data.drop(['place_id','row_id'],axis=1)
    
    #进行数据的分割：训练集合 & 测试集合
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

    #特征工程（标准化）
    std = StandardScaler()
    #对测试集合训练集的特征值进行标准化
    x_train = std.fit_transform(x_train)
    x_test  = std.transform(x_test)
    
    #进行算法流程
    knn = KNeighborsClassifier(n_neighbors=5)
    #fit ， predict ，score
    knn.fit(x_train,y_train)

    #得出预测结果
    y_predict = knn.predict(x_test)

    #得出准确率
    knn.score(x_test,y_test)
    

if __name__ == '__main__':
    knncls()
