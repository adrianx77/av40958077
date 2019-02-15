# coding=utf-8
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import numpy as np
import jieba

def dictvec():
    """
    字典数据抽取
    :return: None
    """
    #实例化
    dict = DictVectorizer(sparse=False)
    data = dict.fit_transform([{"city":"北京","temperature":100},{"city":"上海","temperature":60},{"city":"深圳","temperature":30},{"city":"重庆","temperature":26}])

    print(dict.get_feature_names())

    print(dict.inverse_transform(data))

    print(data)


    return None


def cutword():
    con1 = jieba.cut("卫健委发布卫生健康对口支援工作有关情况，卫健委方面介绍，对口支援工作开展以来，工作力度不断加大，各项举措不断深化，已实现所有国家级贫困县县医院远程医疗全覆盖。")

    con2 = jieba.cut("截至2018年底，三级医院已派出超过6万人次医务人员参与贫困县县级医院管理和诊疗工作，门诊诊疗人次超过3000万，管理出院患者超过300万，住院手术超过50万台。通过派驻人员的传、帮、带，帮助县医院新建临床专科5900个，开展新技术、新项目超过3.8万项。已有超过400家贫困县医院成为二级甲等医院，30余家贫困县医院达到三级医院水平。三级医院优质医疗服务有效下沉，贫困县县医院服务能力和管理水平明显提升")

    con3 = jieba.cut('“组团式”援疆援藏工作踏实推进。截至2018年底，已派出两批共315名专家支援新疆8所受援医院，累计诊疗患者9.32万人，手术1.08万台，实施新项目近500个，急危重症抢救成功率达90%。派出四批共699名专家支援西藏8所受援医院，目前已有332种“大病”不出自治区、1914种“中病”不出地市、常见的“小病”在县域内就能得到及时治疗。')

    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    return c1,c2,c3

def tfidfvec():
    c1,c2,c3 = cutword()

    tf = TfidfVectorizer()
    data = tf.fit_transform([c1,c2,c3])

    print(tf.get_feature_names())

    print(data.toarray())    
    return None


def countvec():
    cv = CountVectorizer()
    data = cv.fit_transform(['人生苦短,我喜欢Python','人生漫长，何不用python'])

    print(cv.get_feature_names())

    print(data.toarray())


def mm():
    """
    归一化
    """   
    mm = MinMaxScaler(feature_range=(0,1))
    data = mm.fit_transform([[90,2,10,40],[60,4,15,45],[75,3,13,46]])
    print(data)

def stand():
    std = StandardScaler()
    data = std.fit_transform([[1,-1,3],[2,4,2],[4,6,-1]])
    print (data)

def im():
    """
    缺失值
    """
    im = Imputer(missing_values='NaN',strategy='mean',axis=0)
    data = im.fit_transform([[1,2],[np.nan,3],[7,6]])
    print(data)
    return None


def var():
    """
    方差
    """
    var = VarianceThreshold(threshold=0.0)
    data = var.fit_transform([[0,2,0,3],[0,1,4,3],[0,1,1,3]])
    print (data)
    return None

def pca():
    """
    主成分分析
    """
    pca = PCA(n_components=0.9)
    data = pca.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])

    print(data)

if __name__ == "__main__":
    pca()