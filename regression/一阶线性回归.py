import random

import matplotlib.pyplot as plt
# 线性回归
import numpy as np  # 快速操作结构数组的工具
# [{
#   key: same,
#   value:
# }]
from sklearn.linear_model import LinearRegression
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.optimizers import SGD

# X, Y = getData(100, 25, 10)
# print("X:", X)
# print("Y:", Y)
#
# numIterations = 100000
# alpha = 0.0005
# theta = np.ones(X.shape[1])
# theta = graientDescent(X, Y, theta, alpha, X.shape[0], numIterations)
# print(theta)

inputArray = [{"key_as_string": "2019-12-01T13:10:00.000Z", "key": 1575205800000, "doc_count": 6362},
              {"key_as_string": "2019-12-01T13:20:00.000Z", "key": 1575206400000, "doc_count": 7038},
              {"key_as_string": "2019-12-01T13:30:00.000Z", "key": 1575207000000, "doc_count": 7299},
              {"key_as_string": "2019-12-01T13:40:00.000Z", "key": 1575207600000, "doc_count": 6296},
              {"key_as_string": "2019-12-01T13:50:00.000Z", "key": 1575208200000, "doc_count": 6688},
              {"key_as_string": "2019-12-01T14:00:00.000Z", "key": 1575208800000, "doc_count": 9755},
              {"key_as_string": "2019-12-01T14:10:00.000Z", "key": 1575209400000, "doc_count": 6750},
              {"key_as_string": "2019-12-01T14:20:00.000Z", "key": 1575210000000, "doc_count": 6640},
              {"key_as_string": "2019-12-01T14:30:00.000Z", "key": 1575210600000, "doc_count": 7104},
              {"key_as_string": "2019-12-01T14:40:00.000Z", "key": 1575211200000, "doc_count": 6737},
              {"key_as_string": "2019-12-01T14:50:00.000Z", "key": 1575211800000, "doc_count": 6527},
              {"key_as_string": "2019-12-01T15:00:00.000Z", "key": 1575212400000, "doc_count": 6721},
              {"key_as_string": "2019-12-01T15:10:00.000Z", "key": 1575213000000, "doc_count": 5570},
              {"key_as_string": "2019-12-01T15:20:00.000Z", "key": 1575213600000, "doc_count": 6260},
              {"key_as_string": "2019-12-01T15:30:00.000Z", "key": 1575214200000, "doc_count": 6295},
              {"key_as_string": "2019-12-01T15:40:00.000Z", "key": 1575214800000, "doc_count": 5829},
              {"key_as_string": "2019-12-01T15:50:00.000Z", "key": 1575215400000, "doc_count": 5593},
              {"key_as_string": "2019-12-01T16:00:00.000Z", "key": 1575216000000, "doc_count": 8351},
              {"key_as_string": "2019-12-01T16:10:00.000Z", "key": 1575216600000, "doc_count": 5469},
              {"key_as_string": "2019-12-01T16:20:00.000Z", "key": 1575217200000, "doc_count": 6546},
              {"key_as_string": "2019-12-01T16:30:00.000Z", "key": 1575217800000, "doc_count": 7538},
              {"key_as_string": "2019-12-01T16:40:00.000Z", "key": 1575218400000, "doc_count": 4366},
              {"key_as_string": "2019-12-01T16:50:00.000Z", "key": 1575219000000, "doc_count": 4722},
              {"key_as_string": "2019-12-01T17:00:00.000Z", "key": 1575219600000, "doc_count": 4473},
              {"key_as_string": "2019-12-01T17:10:00.000Z", "key": 1575220200000, "doc_count": 3326},
              {"key_as_string": "2019-12-01T17:20:00.000Z", "key": 1575220800000, "doc_count": 3769},
              {"key_as_string": "2019-12-01T17:30:00.000Z", "key": 1575221400000, "doc_count": 3662},
              {"key_as_string": "2019-12-01T17:40:00.000Z", "key": 1575222000000, "doc_count": 2637},
              {"key_as_string": "2019-12-01T17:50:00.000Z", "key": 1575222600000, "doc_count": 3023},
              {"key_as_string": "2019-12-01T18:00:00.000Z", "key": 1575223200000, "doc_count": 2982},
              {"key_as_string": "2019-12-01T18:10:00.000Z", "key": 1575223800000, "doc_count": 2341},
              {"key_as_string": "2019-12-01T18:20:00.000Z", "key": 1575224400000, "doc_count": 2498},
              {"key_as_string": "2019-12-01T18:30:00.000Z", "key": 1575225000000, "doc_count": 2388},
              {"key_as_string": "2019-12-01T18:40:00.000Z", "key": 1575225600000, "doc_count": 1968},
              {"key_as_string": "2019-12-01T18:50:00.000Z", "key": 1575226200000, "doc_count": 1950},
              {"key_as_string": "2019-12-01T19:00:00.000Z", "key": 1575226800000, "doc_count": 2486},
              {"key_as_string": "2019-12-01T19:10:00.000Z", "key": 1575227400000, "doc_count": 1588},
              {"key_as_string": "2019-12-01T19:20:00.000Z", "key": 1575228000000, "doc_count": 1817},
              {"key_as_string": "2019-12-01T19:30:00.000Z", "key": 1575228600000, "doc_count": 1967},
              {"key_as_string": "2019-12-01T19:40:00.000Z", "key": 1575229200000, "doc_count": 1553},
              {"key_as_string": "2019-12-01T19:50:00.000Z", "key": 1575229800000, "doc_count": 1399},
              {"key_as_string": "2019-12-01T20:00:00.000Z", "key": 1575230400000, "doc_count": 1966},
              {"key_as_string": "2019-12-01T20:10:00.000Z", "key": 1575231000000, "doc_count": 1229},
              {"key_as_string": "2019-12-01T20:20:00.000Z", "key": 1575231600000, "doc_count": 1355},
              {"key_as_string": "2019-12-01T20:30:00.000Z", "key": 1575232200000, "doc_count": 1384},
              {"key_as_string": "2019-12-01T20:40:00.000Z", "key": 1575232800000, "doc_count": 1084},
              {"key_as_string": "2019-12-01T20:50:00.000Z", "key": 1575233400000, "doc_count": 1230},
              {"key_as_string": "2019-12-01T21:00:00.000Z", "key": 1575234000000, "doc_count": 13040},
              {"key_as_string": "2019-12-01T21:10:00.000Z", "key": 1575234600000, "doc_count": 1205},
              {"key_as_string": "2019-12-01T21:20:00.000Z", "key": 1575235200000, "doc_count": 1964},
              {"key_as_string": "2019-12-01T21:30:00.000Z", "key": 1575235800000, "doc_count": 1565},
              {"key_as_string": "2019-12-01T21:40:00.000Z", "key": 1575236400000, "doc_count": 1390},
              {"key_as_string": "2019-12-01T21:50:00.000Z", "key": 1575237000000, "doc_count": 1474},
              {"key_as_string": "2019-12-01T22:00:00.000Z", "key": 1575237600000, "doc_count": 25835},
              {"key_as_string": "2019-12-01T22:10:00.000Z", "key": 1575238200000, "doc_count": 8990},
              {"key_as_string": "2019-12-01T22:20:00.000Z", "key": 1575238800000, "doc_count": 4389},
              {"key_as_string": "2019-12-01T22:30:00.000Z", "key": 1575239400000, "doc_count": 4361},
              {"key_as_string": "2019-12-01T22:40:00.000Z", "key": 1575240000000, "doc_count": 3628},
              {"key_as_string": "2019-12-01T22:50:00.000Z", "key": 1575240600000, "doc_count": 2477},
              {"key_as_string": "2019-12-01T23:00:00.000Z", "key": 1575241200000, "doc_count": 9402},
              {"key_as_string": "2019-12-01T23:10:00.000Z", "key": 1575241800000, "doc_count": 2162},
              {"key_as_string": "2019-12-01T23:20:00.000Z", "key": 1575242400000, "doc_count": 2571},
              {"key_as_string": "2019-12-01T23:30:00.000Z", "key": 1575243000000, "doc_count": 2624},
              {"key_as_string": "2019-12-01T23:40:00.000Z", "key": 1575243600000, "doc_count": 2535},
              {"key_as_string": "2019-12-01T23:50:00.000Z", "key": 1575244200000, "doc_count": 2851}]
outputArray = [{"key_as_string": "2019-12-01T13:10:00.000Z", "key": 1575205800000, "doc_count": 568},
               {"key_as_string": "2019-12-01T13:20:00.000Z", "key": 1575206400000, "doc_count": 548},
               {"key_as_string": "2019-12-01T13:30:00.000Z", "key": 1575207000000, "doc_count": 587},
               {"key_as_string": "2019-12-01T13:40:00.000Z", "key": 1575207600000, "doc_count": 554},
               {"key_as_string": "2019-12-01T13:50:00.000Z", "key": 1575208200000, "doc_count": 612},
               {"key_as_string": "2019-12-01T14:00:00.000Z", "key": 1575208800000, "doc_count": 622},
               {"key_as_string": "2019-12-01T14:10:00.000Z", "key": 1575209400000, "doc_count": 559},
               {"key_as_string": "2019-12-01T14:20:00.000Z", "key": 1575210000000, "doc_count": 560},
               {"key_as_string": "2019-12-01T14:30:00.000Z", "key": 1575210600000, "doc_count": 618},
               {"key_as_string": "2019-12-01T14:40:00.000Z", "key": 1575211200000, "doc_count": 648},
               {"key_as_string": "2019-12-01T14:50:00.000Z", "key": 1575211800000, "doc_count": 564},
               {"key_as_string": "2019-12-01T15:00:00.000Z", "key": 1575212400000, "doc_count": 669},
               {"key_as_string": "2019-12-01T15:10:00.000Z", "key": 1575213000000, "doc_count": 487},
               {"key_as_string": "2019-12-01T15:20:00.000Z", "key": 1575213600000, "doc_count": 527},
               {"key_as_string": "2019-12-01T15:30:00.000Z", "key": 1575214200000, "doc_count": 578},
               {"key_as_string": "2019-12-01T15:40:00.000Z", "key": 1575214800000, "doc_count": 706},
               {"key_as_string": "2019-12-01T15:50:00.000Z", "key": 1575215400000, "doc_count": 524},
               {"key_as_string": "2019-12-01T16:00:00.000Z", "key": 1575216000000, "doc_count": 517},
               {"key_as_string": "2019-12-01T16:10:00.000Z", "key": 1575216600000, "doc_count": 411},
               {"key_as_string": "2019-12-01T16:20:00.000Z", "key": 1575217200000, "doc_count": 443},
               {"key_as_string": "2019-12-01T16:30:00.000Z", "key": 1575217800000, "doc_count": 521},
               {"key_as_string": "2019-12-01T16:40:00.000Z", "key": 1575218400000, "doc_count": 333},
               {"key_as_string": "2019-12-01T16:50:00.000Z", "key": 1575219000000, "doc_count": 384},
               {"key_as_string": "2019-12-01T17:00:00.000Z", "key": 1575219600000, "doc_count": 304},
               {"key_as_string": "2019-12-01T17:10:00.000Z", "key": 1575220200000, "doc_count": 264},
               {"key_as_string": "2019-12-01T17:20:00.000Z", "key": 1575220800000, "doc_count": 295},
               {"key_as_string": "2019-12-01T17:30:00.000Z", "key": 1575221400000, "doc_count": 214},
               {"key_as_string": "2019-12-01T17:40:00.000Z", "key": 1575222000000, "doc_count": 240},
               {"key_as_string": "2019-12-01T17:50:00.000Z", "key": 1575222600000, "doc_count": 229},
               {"key_as_string": "2019-12-01T18:00:00.000Z", "key": 1575223200000, "doc_count": 211},
               {"key_as_string": "2019-12-01T18:10:00.000Z", "key": 1575223800000, "doc_count": 235},
               {"key_as_string": "2019-12-01T18:20:00.000Z", "key": 1575224400000, "doc_count": 218},
               {"key_as_string": "2019-12-01T18:30:00.000Z", "key": 1575225000000, "doc_count": 178},
               {"key_as_string": "2019-12-01T18:40:00.000Z", "key": 1575225600000, "doc_count": 186},
               {"key_as_string": "2019-12-01T18:50:00.000Z", "key": 1575226200000, "doc_count": 160},
               {"key_as_string": "2019-12-01T19:00:00.000Z", "key": 1575226800000, "doc_count": 192},
               {"key_as_string": "2019-12-01T19:10:00.000Z", "key": 1575227400000, "doc_count": 160},
               {"key_as_string": "2019-12-01T19:20:00.000Z", "key": 1575228000000, "doc_count": 174},
               {"key_as_string": "2019-12-01T19:30:00.000Z", "key": 1575228600000, "doc_count": 219},
               {"key_as_string": "2019-12-01T19:40:00.000Z", "key": 1575229200000, "doc_count": 166},
               {"key_as_string": "2019-12-01T19:50:00.000Z", "key": 1575229800000, "doc_count": 147},
               {"key_as_string": "2019-12-01T20:00:00.000Z", "key": 1575230400000, "doc_count": 180},
               {"key_as_string": "2019-12-01T20:10:00.000Z", "key": 1575231000000, "doc_count": 118},
               {"key_as_string": "2019-12-01T20:20:00.000Z", "key": 1575231600000, "doc_count": 165},
               {"key_as_string": "2019-12-01T20:30:00.000Z", "key": 1575232200000, "doc_count": 156},
               {"key_as_string": "2019-12-01T20:40:00.000Z", "key": 1575232800000, "doc_count": 103},
               {"key_as_string": "2019-12-01T20:50:00.000Z", "key": 1575233400000, "doc_count": 116},
               {"key_as_string": "2019-12-01T21:00:00.000Z", "key": 1575234000000, "doc_count": 161},
               {"key_as_string": "2019-12-01T21:10:00.000Z", "key": 1575234600000, "doc_count": 135},
               {"key_as_string": "2019-12-01T21:20:00.000Z", "key": 1575235200000, "doc_count": 129},
               {"key_as_string": "2019-12-01T21:30:00.000Z", "key": 1575235800000, "doc_count": 159},
               {"key_as_string": "2019-12-01T21:40:00.000Z", "key": 1575236400000, "doc_count": 225},
               {"key_as_string": "2019-12-01T21:50:00.000Z", "key": 1575237000000, "doc_count": 113},
               {"key_as_string": "2019-12-01T22:00:00.000Z", "key": 1575237600000, "doc_count": 127},
               {"key_as_string": "2019-12-01T22:10:00.000Z", "key": 1575238200000, "doc_count": 166},
               {"key_as_string": "2019-12-01T22:20:00.000Z", "key": 1575238800000, "doc_count": 130},
               {"key_as_string": "2019-12-01T22:30:00.000Z", "key": 1575239400000, "doc_count": 146},
               {"key_as_string": "2019-12-01T22:40:00.000Z", "key": 1575240000000, "doc_count": 142},
               {"key_as_string": "2019-12-01T22:50:00.000Z", "key": 1575240600000, "doc_count": 174},
               {"key_as_string": "2019-12-01T23:00:00.000Z", "key": 1575241200000, "doc_count": 165},
               {"key_as_string": "2019-12-01T23:10:00.000Z", "key": 1575241800000, "doc_count": 160},
               {"key_as_string": "2019-12-01T23:20:00.000Z", "key": 1575242400000, "doc_count": 185},
               {"key_as_string": "2019-12-01T23:30:00.000Z", "key": 1575243000000, "doc_count": 185},
               {"key_as_string": "2019-12-01T23:40:00.000Z", "key": 1575243600000, "doc_count": 220},
               {"key_as_string": "2019-12-01T23:50:00.000Z", "key": 1575244200000, "doc_count": 207}]

key_name = "key"
val_name = "doc_count"


def tranToDict(listOfDict):
    """
    [{item},{item}] => (k,v)
    :param listOfDict: [{item},{item}]
    :return: (k,v)
    """
    print("tranToDict in:" + str(listOfDict))
    kvDict = dict()
    for dictItem in listOfDict:
        key = dictItem[key_name]
        val = dictItem[val_name]
        kvDict[key] = val
    print("tranToDict out:" + str(kvDict))
    return kvDict


def narrowResponse(inputListDict, outputDict):
    """
    :param inputListDict: 自变量
    :param outputDict: 因变量
    :return: Narrow Array [[input,output]]
    """
    x = []
    y = []
    narrow = []
    print("narrowResponse in:" + str(inputListDict))
    print("narrowResponse in:" + str(outputDict))
    for inputItem in inputListDict:
        searchKey = inputItem[key_name]
        inputValue = inputItem[val_name]
        outputValue = outputDict[searchKey]
        x.append(inputValue)
        y.append(outputValue)
        narrow.append([inputValue, outputValue])
    print("narrowResponse out:" + str(x))
    print("narrowResponse out:" + str(y))
    print("narrowResponse out:" + str(narrow))
    return narrow, x, y


def showXYGraph(x, y):
    plt.title("webpower request/exception per 10min")
    plt.plot(x, y, 'ro')
    plt.show()


def gradientDescent(x, y, theta, alpha, m, numIterations):  # 梯度下降算法
    xTrain = x.transpose()
    for i in range(0, numIterations):  # 重复多少次
        hypothesis = np.dot(x, theta)  # h函数
        loss = hypothesis - y

        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d / cost:%f" % (i, cost))
        graient = np.dot(xTrain, loss) / m
        theta = theta - alpha * graient
    return theta


def getData(numPoints, bias, variance):  # 自己生成待处理数据
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    for i in range(0, numPoints):
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y


def regression(x_data, y_data):
    # 构建一个顺序模型
    model = Sequential()

    # 在模型中添加一个全连接层
    # 神经网络结构：1-10-1，即输入层为1个神经元，隐藏层10个神经元，输出层1个神经元。

    # 激活函数加法1
    model.add(Dense(units=10, input_dim=1))
    model.add(Activation('tanh'))
    model.add(Dense(units=1))
    model.add(Activation('tanh'))

    # 激活函数加法2
    # model.add(Dense(units=10, input_dim=1, activation='relu'))
    # model.add(Dense(units=1, activation='relu'))

    # 定义优化算法
    sgd = SGD(lr=0.3)
    # sgd: Stochastic gradient descent,随机梯度下降法
    # mse: Mean Squared Error, 均方误差
    model.compile(optimizer=sgd, loss='mse')

    # 进行训练
    for step in range(3001):
        # 每次训练一个批次
        cost = model.train_on_batch(x_data, y_data)
        # 每500个batch打印一次cost值
        if step % 500 == 0:
            print('cost: ', cost)
    # 打印权值和偏置值
    W, b = model.layers[0].get_weights()
    print('W：', W, ' b: ', b)
    print(len(model.layers))

    # 把x_data输入网络中，得到预测值y_pred
    y_pred = model.predict(x_data)

    # 显示随机点
    plt.scatter(x_data, y_data)
    # 显示预测结果
    plt.plot(x_data, y_pred, 'r-', lw=3)
    plt.show()


def 线性回归(data):
    # 生成X和y矩阵
    dataMat = np.array(data)
    X = dataMat[:, 0:1]  # 变量x
    y = dataMat[:, 1]  # 变量y

    # ========线性回归========
    model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    model.fit(X, y)  # 线性回归建模
    print('系数矩阵:\n', model.coef_)
    print('线性回归模型:\n', model)
    # 使用模型预测
    predicted = model.predict(X)

    # 绘制散点图 参数：x横轴 y纵轴
    plt.scatter(X, y, marker='x')
    plt.plot(X, predicted, c='r')

    # 绘制x轴和y轴坐标
    plt.xlabel("x")
    plt.ylabel("y")

    # 显示图形
    plt.show()


if __name__ == '__main__':
    outputDict = tranToDict(outputArray)
    narrow, x, y = narrowResponse(inputArray, outputDict)
    showXYGraph(x, y)

    # regression(x, y)
    线性回归(narrow)
