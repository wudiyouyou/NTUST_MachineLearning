#! coding = utf-8
 
'''
作业要求：
    单独完成Bank Marketing数据集的分类任务
    1）一份文档，关于实现这个分类任务，至少包括实验目的、数据描述、数据预处理方法、模型建立、模型评估、结果可视化等几个部分
    2）分类任务的代码，Python实现，使用Anaconda中的jupyter noterbook作为开发环境，用“学号_姓名.ipynb”提交代码
    3）不少于三种分类方法
 
数据下载地址：http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
 
作者：刘宇
开发时间：2018-4-27
'''
 
# 导入全局必须的包,用做科学计算等
import pandas as pd

import os
# 相对路径
print(os.path.abspath('.'))
# 可能因为版本问题，有部分内容会出现警告，此处禁止显示各类警告，让结果更加清晰
import warnings
 
warnings.filterwarnings("ignore")
 
 
# Bank数据处理
class BankDataHandle:
    '''
        主要用数据处理：
            readData:读取数据
            deleUnknowData:删除部分为Unknow的数据
            splitData:进行数据切割，test_size默认为0.25
            upData:通过SMOTE方法保持正负样本平衡
            downData:通过删除数据的方法保证正负样本平衡
            virtualizationData:虚拟化
            standardScalerData:规范化
    '''
 
    def __init__(self, string_key=None, num_key=None):
        '''
        初始化
        :param string_key: 原数据中为string的key
        :param num_key: 原数据中为num的key
        * 以上两个参数，默认是bank-full.csv的
        '''
 
        # string参数进行初始化
        if string_key:
            self.temp_string_key = string_key
        else:
            self.temp_string_key = ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome",
                                    "y", "month"]
 
        # num参数进行初始化
        if num_key:
            self.temp_number_key = num_key
        else:
            self.temp_number_key = ["age", "balance", "day", "duration", "pdays"]
 
    def readData(self, path, csplit=";"):
        '''
        读取数据，主要通过pandas进行数据读取
        :param path: 数据所放的位置
        :param csplit: 切割数据的内容，原来pandas.read_csv默认是“，”，此处默认为“；”
        :return: 得到的数据，DF
        '''
        bank_full_data = pd.read_csv(path, csplit)
        return bank_full_data
 
    def deleUnknowData(self, bank_full_data):
        '''
        用于删除Unknow的数据
        按照道理来说，遇到Unknow的数据，通常有三种处理方法：
            1：删除
            2：填补
                2.1：平均值
                2.2：众数
                2.3：其他
            3：不处理
        但是，本数据由于很难估量缺失值对结果的影响，而且确实列数众多，所以暂不考虑填充和不处理，直接进行删除，但是此种做法也有弊端，那就是导致可用数据大大减少！
        * 如果可以进行程序的优化，可以在此处进行优化！
        :param bank_full_data: 原始数据
        :return: 将所有Unknow行删掉的数据
        '''
        for eve_key in self.temp_string_key:
            bank_full_data = bank_full_data[bank_full_data[eve_key] != 'unknown']
        return bank_full_data
 
    def splitData(self, x_data, y_data, test_size=None):
        '''
        进行数据切割，将数据切分为学习集和验证集，默认的test_size是0.25
        :param x_data: 未切割的x数据
        :param y_data: 未切割的y数据
        :param test_size: 验证集比例
        :return: 返回切割好的数据 x_train, x_test, y_train, y_test
        '''
 
        # 通过sklearn中的model_selection.train_test_spilit进行切割
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size)
        return (x_train, x_test, y_train, y_test)
 
    def upData(self, x_data, y_data):
        '''
        通过SMOTE方法进行正负样本平衡
        :param x_data: 不平衡的X数据
        :param y_data: 不平衡的X数据对应的lable
        :return: 返回平衡之后的样本数据
        '''
 
        # SMOTE算法，网上有很多实现方法，这里使用了imblearn中的over_sampling.SMOTE
        from imblearn.over_sampling import SMOTE
        # 建立SMOTE对象
        oversampler = SMOTE(random_state=0)
        # 生成数据，使得正负样本平衡
        x_data, y_data = oversampler.fit_sample(x_data, y_data)
        # 返回生成后的数据
        return (x_data, y_data)
 
    def downData(self, bank_full_data):
        '''
        通过切割方法，实现正负样本平衡，这种方法一般不采用，虽然更加贴近客观事实，但是会让数据大量丢失
        :param bank_full_data: 整体数据
        :return: 平衡后的样本数据，DF
        '''
 
        # 先分别对正负样本进行统计，列出正负样本
        y_yes_data = bank_full_data[bank_full_data['y'] == 1]
        y_no_data = bank_full_data[bank_full_data['y'] == 0]
 
        # 根据判断，使大的样本变成和小的样本一致的样本
        if len(y_yes_data) > len(y_no_data):
            y_yes_data = y_yes_data.sample(frac=len(y_no_data) / len(y_yes_data))
        else:
            y_no_data = y_no_data.sample(frac=len(y_yes_data) / len(y_no_data))
 
        # 正负样本重新合成新的整体样本
        return pd.concat([y_yes_data, y_no_data])
 
    def virtualizationData(self, total_data):
        '''
        虚拟化，这部分主要是将有的部分非数值数据进行数值化，例如：
            1：有些数据是yes，no，这样的数据变成1,0
            2：有些数据是有几种类型，A,B,C，这样的数据变成1,2,3
        :param total_data: 传入的整体数据
        :return: 虚拟化后的数据
        '''
 
        temp_key_value = {}
        # 由于整体数据有一部分是数值，所以这里只对非数值的列进行处理，非数值的列是self.temp_string_key
        for eve_key in self.temp_string_key:
            # 首先获取全部该列内容，然后set一下，去掉重复
            temp_data = set(total_data[eve_key])
 
            temp = {}
 
            # 根据情况，初始化标记起始值
            if len(temp_data) > 2:
                start_num = 1
            else:
                start_num = 0
 
            # 建立字典数据
            for eve_temp_data in temp_data:
                temp[eve_temp_data] = start_num
                start_num = start_num + 1
            temp_key_value[eve_key] = temp
 
        # 根据字典数据，实现虚拟化
        for eve_key in self.temp_string_key:
            dict_list = temp_key_value[eve_key]
            for eve_data_key, eve_data_value in dict_list.items():
                total_data.loc[total_data[eve_key] == eve_data_key, eve_key] = eve_data_value
 
        # 返回虚拟化数据
        return total_data
 
    def standardScalerData(self, total_data):
        '''
        标准化，一般情况下，数据范围不一样，通常都会预处理，例如A和B两个因素同时影响一个数据，A的范围是0-1，B的范围是1-100000，这样在预测时候
        可能默认将B的权重“变大”，影响结果，所以此时通常会做一些预处理，例如：
            1：标准化：去均值，方差规模化
            2：正则化
            3：特征的二值化
            4：归一化
            ... ...
        :param total_data:输入的数据
        :return:返回规范化之后的结果
        '''
 
        # 直接通过sklearn中的preprocessing进行规范化
        from sklearn import preprocessing
 
        # 主要对默认的self.temp_number_key列进行规范化
        for eve_key in self.temp_number_key:
            scaler = preprocessing.StandardScaler()
            total_data[eve_key] = scaler.fit_transform(total_data[eve_key].reshape(-1, 1))
 
        # 返回规范化之后的数据
        return total_data
 
 
# 模型
class BankClassifier:
    '''
    主要是模型的整合，其中在实例化的时候，需要传递默认参数：x_train, y_train, x_test, y_test
    各个方法具体功能：
        modelScore：模型评分，由于是分类问题，此处只通过简单的score进行评分验证
        drawPicture：绘图（可视化）
        SVC：支持向量机算法 （svm.SVC - 分类）
        KNN：临近取样（neighbors.KNeighborsClassifier）

        RF：随机森林算法（ensemble.RandomForestClassifier）

        LG：逻辑回归算法（linear_model.LogisticRegression）
    '''
 
    def __init__(self, x_train, y_train, x_test, y_test):
        '''
        初始化，需要传递学习集和验证集
        :param x_train: 学习集X
        :param y_train: 学习集Y
        :param x_test: 验证集X
        :param y_test: 验证集Y
        '''
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
 
    def modelScore(self, dataModel):
        '''
        简单的评分
        :param dataModel: 模型
        :return: 分数
        '''
        return dataModel.score(self.x_test, self.y_test)
 
    def drawPicture(self,x_plot,y_plot,picture):
        '''
        绘图方法
        :param x_plot: x数据
        :param y_plot: y数据
        :param picture: 图像名字
        :return: 无返回
        '''
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(x_plot, y_plot, 'o')
        plt.savefig(picture)
 
 
    def RF(self, arges=None, score=True, picture=None):
        '''
        RF：随机森林
        :param arges: 是否进行调参
            *这里的调参，只是简单的参数调账，是调整n_estimators
            *这里默认不进行调参是5，调参之后是1-200,step是1
        :param score:是否输出评分，是的话，输出评分，否的话输出模型和最优参数
        :picture: 是否输出图像，如果输出，直接写入名字即可
        :return:输出对应结果
        '''
 
        # 引入RandomForestClassifier
        from sklearn.ensemble import RandomForestClassifier
 
        # 根据arges设置是否调参和调参范围
        if not arges:
            start_num = 10
            end_num = 11
        else:
            start_num = 1
            end_num = 200
 
        # 用来存储临时结果
        temp_list = []
 
        # 进行模型训练
        for step in range(start_num, end_num):
            rf_model = RandomForestClassifier(n_estimators=step)
            rf_model.fit(self.x_train, self.y_train)
            temp_list.append((self.modelScore(rf_model), step))
 
        # 根据绘图需求进行图像绘制
        if picture:
            # 建立X,Y数据
            x_plot = []
            y_plot = []
            for eve_plot_data in temp_list:
                x_plot.append(eve_plot_data[1])
                y_plot.append(eve_plot_data[0])
 
            # 绘图
            self.drawPicture(x_plot, y_plot, picture)
 
        # 根据结果，输出最大的评分，或者最优的模型和参数
        if score:
            return max(temp_list, key=lambda x: x[0])[0]
        else:
            return_step = max(temp_list, key=lambda x: x[0])[1]
            return_model = RandomForestClassifier(n_estimators=return_step)
            return_model.fit(self.x_train, self.y_train)
            return (return_step, return_model)


    def DT_max_depth(self, arges=None, score=True, picture=None):
        '''
        DT：决策树
        :param arges: 是否进行调参
            *这里的调参，只是简单的参数调账，是调整
            *这里默认不进行调参是5，调参之后是1-200,step是1
        :param score:是否输出评分，是的话，输出评分，否的话输出模型和最优参数
        :picture: 是否输出图像，如果输出，直接写入名字即可
        :return:输出对应结果
        '''
        # max_depth， min_samples_split，min_samples_leaf， min_weight_fraction_leaf， max_leaf_nodes， min_impurity_split。
 
        # 引入RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier 
 
        # 根据arges设置是否调参和调参范围
        if not arges:
            start_num = 10
            end_num = 11
        else:
            start_num = 2
            end_num = 40    
 
        # 用来存储临时结果
        temp_list = []
 
        # 进行模型训练
        for step in range(start_num, end_num):
            rf_model = DecisionTreeClassifier(max_depth=step)
            rf_model.fit(self.x_train, self.y_train)
            temp_list.append((self.modelScore(rf_model), step))
 
        # 根据绘图需求进行图像绘制
        if picture:
            # 建立X,Y数据
            x_plot = []
            y_plot = []
            for eve_plot_data in temp_list:
                x_plot.append(eve_plot_data[1])
                y_plot.append(eve_plot_data[0])
 
            # 绘图
            self.drawPicture(x_plot, y_plot, picture)
 
        # 根据结果，输出最大的评分，或者最优的模型和参数
        if score:
            return max(temp_list, key=lambda x: x[0])[0]
        else:
            return_step = max(temp_list, key=lambda x: x[0])[1]
            return_model = RandomForestClassifier(max_depth=return_step)
            return_model.fit(self.x_train, self.y_train)
            return (return_step, return_model)

    def DT_min_samples_split(self, arges=None, score=True, picture=None):
        '''
        DT：决策树
        :param arges: 是否进行调参
            *这里的调参，只是简单的参数调账，是调整
            *这里默认不进行调参是5，调参之后是1-200,step是1
        :param score:是否输出评分，是的话，输出评分，否的话输出模型和最优参数
        :picture: 是否输出图像，如果输出，直接写入名字即可
        :return:输出对应结果
        '''
        # max_depth， min_samples_split，min_samples_leaf， min_weight_fraction_leaf， max_leaf_nodes， min_impurity_split。
 
        # 引入RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier 
 
        # 根据arges设置是否调参和调参范围
        if not arges:
            start_num = 2
            end_num = 3
        else:
            start_num = 2
            end_num = 10
 
        # 用来存储临时结果
        temp_list = []
 
        # 进行模型训练
        for step in range(start_num, end_num):
            rf_model = DecisionTreeClassifier(min_samples_split=step)
            rf_model.fit(self.x_train, self.y_train)
            temp_list.append((self.modelScore(rf_model), step))
 
        # 根据绘图需求进行图像绘制
        if picture:
            # 建立X,Y数据
            x_plot = []
            y_plot = []
            for eve_plot_data in temp_list:
                x_plot.append(eve_plot_data[1])
                y_plot.append(eve_plot_data[0])
 
            # 绘图
            self.drawPicture(x_plot, y_plot, picture)
 
        # 根据结果，输出最大的评分，或者最优的模型和参数
        if score:
            return max(temp_list, key=lambda x: x[0])[0]
        else:
            return_step = max(temp_list, key=lambda x: x[0])[1]
            return_model = RandomForestClassifier(min_samples_split=return_step)
            return_model.fit(self.x_train, self.y_train)
            return (return_step, return_model)


    def DT_min_samples_leaf(self, arges=None, score=True, picture=None):
        '''
        DT：决策树
        :param arges: 是否进行调参
            *这里的调参，只是简单的参数调账，是调整
            *这里默认不进行调参是5，调参之后是1-200,step是1
        :param score:是否输出评分，是的话，输出评分，否的话输出模型和最优参数
        :picture: 是否输出图像，如果输出，直接写入名字即可
        :return:输出对应结果
        '''
        # max_depth， min_samples_split，min_samples_leaf， min_weight_fraction_leaf， max_leaf_nodes， min_impurity_split。
 
        # 引入RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier 
 
        # 根据arges设置是否调参和调参范围
        if not arges:
            start_num = 1
            end_num = 2
        else:
            start_num = 1
            end_num = 10
 
        # 用来存储临时结果
        temp_list = []
 
        # 进行模型训练
        for step in range(start_num, end_num):
            rf_model = DecisionTreeClassifier(min_samples_leaf=step)
            rf_model.fit(self.x_train, self.y_train)
            temp_list.append((self.modelScore(rf_model), step))
 
        # 根据绘图需求进行图像绘制
        if picture:
            # 建立X,Y数据
            x_plot = []
            y_plot = []
            for eve_plot_data in temp_list:
                x_plot.append(eve_plot_data[1])
                y_plot.append(eve_plot_data[0])
 
            # 绘图
            self.drawPicture(x_plot, y_plot, picture)
 
        # 根据结果，输出最大的评分，或者最优的模型和参数
        if score:
            return max(temp_list, key=lambda x: x[0])[0]
        else:
            return_step = max(temp_list, key=lambda x: x[0])[1]
            return_model = RandomForestClassifier(min_samples_leaf=return_step)
            return_model.fit(self.x_train, self.y_train)
            return (return_step, return_model)

 
 
# 以下内容是进行实例化和数据实际处理
 
 
# 此处设定好数据地址，此处使用的是自己本机中的bank-full.CSV数据
bank_full_data_path = r"./data/bank/bank-full.csv"
 
# 数据前期预处理
 
print("BANKDATA对象建立 ... ...")
bankData = BankDataHandle()
 
print("读取数据 ... ...")
read_data = bankData.readData(bank_full_data_path)
 
print("删除Unknow的数据 ... ...")
dele_unknow_data = bankData.deleUnknowData(read_data)
 
print("虚拟化数据 ... ...")
virtualization_data = bankData.virtualizationData(dele_unknow_data)
 
print("标准化数据 ... ...")
standardScaler_data = bankData.standardScalerData(virtualization_data)
 
print("由于正负样本不一致，此处分为两种方法进行处理；")
 
# print("正样本数量%d，负样本数量%d"%(len(standardScaler_data[standardScaler_data['y'] == 1]),len(standardScaler_data[standardScaler_data['y']==0])))
# 此处，发现正负样本数量不一致: 正样本数量1786，负样本数量6056
# 所以，此处采用两种样本平衡方法，一种是向上，使用SMOTE生成样本数据，另一种是向下，使用数据shuffle之后进行切割
# 其中，方法一采用向上平衡，方法二采用向下平衡
 
print("\n\n方法1：通过SMOTE算法，生成部分样本：")
 
print("| - 分割x_data与y_data ... ...")
x_data = standardScaler_data[
    ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month",
     "duration", "campaign", "pdays", "previous", "poutcome"]]
y_data = standardScaler_data["y"]
 
print("| - 训练集和验证机切分，默认是test_size：0.25 ... ...")
x_train, x_test, y_train, y_test = bankData.splitData(x_data, y_data)
 
print("| - SMOTE进行正负样本平衡 ... ...")
x_data, y_data = bankData.upData(x_train, y_train)
# print("正样本数量%d，负样本数量%d"%(list(y_data).count(1),list(y_data).count(0)))
# 平衡之后的样本数量数据：正样本数量4547，负样本数量4547
 
print("| - 分类模型开始建立 ... ...")
modelData = BankClassifier(x_data, y_data, x_test, y_test)
 
print("| - " + "-" * 30 + "\n| - 固定参数：")
print("| - name ：\t%s\t , score ：\t%.5f\t" % ("RF                    ", modelData.RF()))
print("| - name ：\t%s\t , score ：\t%.5f\t" % ("DT_max_depth          ", modelData.DT_max_depth()))
print("| - name ：\t%s\t , score ：\t%.5f\t" % ("DT_min_samples_split  ", modelData.DT_min_samples_split()))
print("| - name ：\t%s\t , score ：\t%.5f\t" % ("DT_min_samples_leaf   ", modelData.DT_min_samples_leaf()))
# print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("临近取样", modelData.KNN()))
# print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("逻辑回归", modelData.LG()))
# print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("支持向量机", modelData.SVC()))
print("| - " + "-" * 30 + "\n| - 调整参数：") 
print("| - name ：\t%s\t , score ：\t%.5f\t" % ("RF                    ", modelData.RF(arges=True,picture="updata_rf")))
print("| - name ：\t%s\t , score ：\t%.5f\t" % ("DT_max_depth          ", modelData.DT_max_depth(arges=True,picture="_max_depth")))
print("| - name ：\t%s\t , score ：\t%.5f\t" % ("DT_min_samples_split  ", modelData.DT_min_samples_split(arges=True,picture="__min_samples_split")))
print("| - name ：\t%s\t , score ：\t%.5f\t" % ("DT_min_samples_leaf   ", modelData.DT_min_samples_leaf(arges=True,picture="__min_samples_leaf")))

# print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("临近取样", modelData.KNN(arges=True,picture="updata_knn")))
# print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("逻辑回归", modelData.LG(arges=True,picture="updata_lg")))
# print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("支持向量机", modelData.SVC(arges=True,picture="updata_svm")))
 
# print("\n\n方法2：将多的样本数量进行减少，来保持正负样本平衡：")
# down_data = bankData.downData(standardScaler_data)
# # print("正样本数量%d，负样本数量%d"%(len(down_data[down_data['y'] == 1]),len(down_data[down_data['y']==0])))
# # 平衡之后的样本数量数据：正样本数量1786，负样本数量1786
 
# print("| - 分割x_data与y_data ... ...")
# x_data = down_data[
#     ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month",
#      "duration", "campaign", "pdays", "previous", "poutcome"]]
# y_data = down_data["y"]
 
# print("| - 训练集和验证机切分，默认是test_size：0.25 ... ...")
# x_train, x_test, y_train, y_test = bankData.splitData(x_data, y_data)
 
# print("| - 分类模型开始建立 ... ...")
# modelData = BankClassifier(x_train, y_train, x_test, y_test)
 
# print("| - " + "-" * 30 + "\n| - 输出结果：")
# # print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("随机森林", modelData.RF()))
# print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("决策树 限制深度          ", modelData.DT_max_depth()))
# print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("决策树 限制节点中最低样本数", modelData.DT_min_samples_split()))
# print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("决策树 限制叶子中最低样本数", modelData.DT_min_samples_leaf()))

# # print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("临近取样", modelData.KNN()))
# # print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("逻辑回归", modelData.LG()))
# # print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("支持向量机", modelData.SVC()))
# print("| - " + "-" * 30 + "\n| - 调参优化：")
# # print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("随机森林", modelData.RF(arges=True,picture="downdata_rf")))
# print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("决策树 限制深度          ", modelData.DT_max_depth(arges=True,picture="downdata_max_depth")))
# print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("决策树 限制节点中最低样本数", modelData.DT_min_samples_split(arges=True,picture="downdata__min_samples_split")))
# print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("决策树 限制叶子中最低样本数", modelData.DT_min_samples_leaf(arges=True,picture="downdata__min_samples_leaf")))

# print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("临近取样", modelData.KNN(arges=True,picture="downdata_knn")))
# print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("逻辑回归", modelData.LG(arges=True,picture="downdata_lg")))
# print("| - 模型名称：\t%s\t , 模型评分：\t%.5f\t" % ("支持向量机", modelData.SVC(arges=True,picture="downdata_svm")))