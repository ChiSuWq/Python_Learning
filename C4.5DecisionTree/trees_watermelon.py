# -*- coding: UTF-8 -*-
"""
Filename: trees_watermelon.py
"""

from math import log
import operator
import copy
import treePlotter
import sys
"""
parameter : dataSet
return: shannonEnt (Value） shannonEnt又称信息熵  
"""
def calcShannonEnt(dataSet): 
	numEntries = len(dataSet)  #numEntries 实例总数
	labelCounts= {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1 #get()方式为防止labelCounts[currentlabel]不存在
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		shannonEnt -= prob * log(prob, 2)  # log(x,base)  base为下标基数 -log(prob, 2)为对应概率密度信息的信息量
	return shannonEnt 

"""
创建样本
:para:
:return: dataSet, labels
"""
def createDataSet():
	"""
	sample = [色泽，根蒂，敲声，纹理，脐部，触感，密度，含糖率，好瓜]
	
	色泽 = ['qinglv'，'wuhei'，'qianbai']
	根蒂 = ['quansuo'，'shaoquan'，'yingting']
	敲声 = ['zhuoxiang'，'chenmen'，'qingcui']
	纹理 = ['qingxi'，'shaohu'，'mohu']
	脐部 = ['aoxian'，'shaoao'，'pingtan']
	触感 = ['yinghua'，'ruannian']
	好瓜 = ['yes'，'no']
	"""

	"""
	dataSet = [['qinglv', 'quansuo', 'zhuoxiang','qingxi','aoxian','yinghua',0.697, 0.460,'yes'],
			   ['wuhei', 'quansuo', 'chenmen', 'qingxi','aoxian','yinghua',0.774,0.376,'yes'],
			   ['wuhei', 'quansuo', 'zhuoxiang','qingxi','aoxian','yinghua',0.634,0.264,'yes'],
			   ['qinglv','quansuo','chenmen','qingxi','aoxian','yinghua',0.608,0.318,'yes'],
			   ['qianbai','quansuo','zhuoxiang','qingxi','aoxian','yinghua',0.556,0.215,'yes'],
			   ['qinglv','shaoquan','zhuoxiang','qingxi','shaoao','ruannian',0.403,0.237,'yes'],
			   ['wuhei','shaoquan','zhuoxiang','shaohu','shaoao','ruannian',0.481,0.149,'yes'],
			   ['wuhei','shaoquan','zhuoxiang','qingxi','shaoao','yinghua',0.437,0.211,'yes'],

			   ['wuhei','shaoquan','chenmen','shaohu','shaoao','yinghua',0.666,0.091,'no'],
			   ['qinglv','yingting','qingcui','qingxi','pingtan','ruannian',0.243,0.267,'no'],
			   ['qianbai','yingting','qingcui','mohu','pingtan','yinghua',0.245,0.057,'no'],
			   ['qianbai','quansuo','zhuoxiang','mohu','pingtan','ruannian',0.343,0.099,'no'],
			   ['qinglv','shaoquan','zhuoxiang','shaohu','aoxian','yinghua',0.639,0.131,'no'],
			   ['qianbai','shaoquan','chenmen','shaohu','aoxian','yinghua',0.657,0.198,'no'],
			   ['wuhei','shaoquan','zhuoxiang','qingxi','pingtan','yinghua',0.360,0.370,'no'],
			   ['qianbai','quansuo','zhuoxiang','mohu','pingtan','yinghua',0.593,0.042,'no'],
			   ['qinglv','quansuo','chenmen','shaohu','shaoao','yinghua',0.719,0.103,'no']

			   ]
	"""
	dataSet = [
			   ['青绿','蜷缩','浊响','清晰','凹陷','硬滑',0.697,0.46,'好瓜'],
			   ['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑',0.774,0.376,'好瓜'],
			   ['乌黑','蜷缩','浊响','清晰','凹陷','硬滑',0.634,0.264,'好瓜'],
			   ['青绿','蜷缩','沉闷','清晰','凹陷','硬滑',0.608,0.318,'好瓜'],
			   ['浅白','蜷缩','浊响','清晰','凹陷','硬滑',0.556,0.215,'好瓜'],
			   ['青绿','稍蜷','浊响','清晰','稍凹','软粘',0.403,0.237,'好瓜'],
			   ['乌黑','稍蜷','浊响','稍糊','稍凹','软粘',0.481,0.149,'好瓜'],
			   ['乌黑','稍蜷','浊响','清晰','稍凹','硬滑',0.437,0.211,'好瓜'],

			   ['乌黑','稍蜷','沉闷','稍糊','稍凹','硬滑',0.666,0.091,'坏瓜'],
			   ['青绿','硬挺','清脆','清晰','平坦','软粘',0.243,0.267,'坏瓜'],
			   ['浅白','硬挺','清脆','模糊','平坦','硬滑',0.245,0.057,'坏瓜'],
			   ['浅白','蜷缩','浊响','模糊','平坦','软粘',0.343,0.099,'坏瓜'],
			   ['青绿','稍蜷','浊响','稍糊','凹陷','硬滑',0.639,0.161,'坏瓜'],
			   ['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑',0.657,0.198,'坏瓜'],
			   ['乌黑','稍蜷','浊响','清晰','稍凹','软粘',0.36,0.37,'坏瓜'],
			   ['浅白','蜷缩','浊响','模糊','平坦','硬滑',0.593,0.042,'坏瓜'],
			   ['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑',0.719,0.103,'坏瓜']

			   ]
	#labels =['seze', 'gendi','qiaosheng','wenli','qibu','chugan','midu','hantanglv']
	labels = ['色泽','根蒂','敲声','纹理','脐部','触感','密度','含糖率']
	return dataSet, labels


"""
按照给定的离散型特征划分数据集，并在划分后的数据集中去除这一特征
:para: dataSet 数据集, axis(对应特征的轴作为索引) 划分数据集的特征, value 特征的特定值
"""
def splitDataSetByDiscrete(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]  #不包括axis
			reducedFeatVec.extend(featVec[axis + 1:]) #已去除featVec[axis]
			retDataSet.append(reducedFeatVec) # 生成新的featVec返回到划分好的数据集中
	return retDataSet

"""
按照连续性特征划分数据集，注意划分后不需要在数据集中去除这一特征
:para: dataSet 数据集, axis(对应特征的轴为索引), value 划分值, mode 划分大于value或小于value的子数据集的flag
"""
def splitDataSetByContinu(dataSet, axis, value, mode='lt'):
	retDataSet = []
	if mode=='lt':
		for featVec in dataSet:
			if featVec[axis] <= value:
				retDataSet.append(featVec) 
	else:
		for featVec in dataSet:
			if featVec[axis] > value:
				retDataSet.append(featVec) 	
	return retDataSet

"""
选择最好的数据集划分方式,包括离散型特征和连续型特征
para: dataSet
return: bestFeature, bestValue(连续型特征)
"""

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet) #原始信息熵
	bestInfoGain = 0.0; bestFeature = -1;
	bestLocalInfoGain, bestValue = 0.0, -1#存放连续值的局部最大信息熵和划分值
	for i in range(numFeatures):
		if not isinstance(dataSet[0][i], float):#如果不好瓜连续值
			featlist = [example[i] for example in dataSet] 
			uniqueVals = set(featlist)
			newEntropy = 0.0
			for value in uniqueVals:
				subDataSet = splitDataSetByDiscrete(dataSet, i, value)
				prob = len(subDataSet)/len(dataSet)
				newEntropy += prob * calcShannonEnt(subDataSet)
			#print('第%d个特征的信息熵好瓜%s' %(i, newEntropy))
			infoGain = baseEntropy - newEntropy
			if infoGain > bestInfoGain:
				bestInfoGain = infoGain
				bestFeature = i
		else:
			#如果好瓜连续值,对特征值进行排序
			featlist = [example[i] for example in dataSet]
			featlist.sort()
			splitValues = [(featlist[i]+featlist[i+1]) * 0.5 for i in range(len(featlist)-1)]
			for splitValue in splitValues:
				newEntropy = 0.0
				for mode in ['lt','gt']:
					subDataSet = splitDataSetByContinu(dataSet, i, splitValue, mode)
					prob = len(subDataSet)/len(dataSet)
					newEntropy += prob * calcShannonEnt(subDataSet)
				infoGain = baseEntropy - newEntropy
				print('numFeatures {} splitValue {}, infoGain {:2f}, bestInfoGain {:2f}'.format(
								numFeatures,splitValue, infoGain, bestInfoGain))
				if infoGain > bestLocalInfoGain:
					bestLocalInfoGain = infoGain
					bestValue = splitValue
			if bestLocalInfoGain > bestInfoGain:
				bestInfoGain = bestLocalInfoGain
				bestFeature = i

	return bestFeature, bestValue



"""
通过信息增益的方式构造树(存储结构为字典)
para: dataSet, labels
return: myTree
"""
def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	bestFeat, bestValue = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	print(bestFeatLabel, bestValue)
	if bestFeatLabel in ['含糖率', '密度']:
		#如果当前最好的特征为连续型特征，不需要删除特征
		myTree = {bestFeatLabel:{} } #利用dict生成多叉树
		myTree[bestFeatLabel]['value'] = bestValue
		for mode in ['lt', 'gt']:
			subLabels = labels[:]
			myTree[bestFeatLabel][mode] = createTree(splitDataSetByContinu(dataSet, bestFeat, bestValue, mode), subLabels)
	else:
		#离散型特征
		myTree = {bestFeatLabel:{} } #利用dict生成多叉树
		del(labels[bestFeat])
		featValues = [example[bestFeat] for example in dataSet]
		uniqueVals = set(featValues)
		for value in uniqueVals:
			subLabels = labels[:]
			myTree[bestFeatLabel][value] = createTree(splitDataSetByDiscrete(dataSet, bestFeat, value),subLabels)
	return myTree

"""
预测测试样本的分类
para: inputTree, featLabels, testVec
return: classLabel
"""
def classify(inputTree, featLabels, testVec):
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	key = testVec[featIndex]
	if firstStr in ['含糖率','密度']:
		#当前划分特征为连续型特征则比较key与划分value的大小，离散型则保持原形
		key = 'lt' if key<secondDict['value'] else 'gt'
	if type(secondDict[key]).__name__ == 'dict':
		classLabel = classify(secondDict[key], featLabels, testVec)
	else:
		classLabel = secondDict[key]
	return classLabel


"""
使用pickle模块存储树  
storeTree 存树
para: inputTree, filename
return:
grabTree 取树
para: filename
return: pickle.load(fr) (字典) 
"""
def storeTree(inputTree, filename):
	import pickle
	fw = open(filename, 'wb')
	pickle.dump(inputTree, fw)
	fw.close()

def grabTree(filename):
	import pickle
	fr = open(filename, 'rb')
	return pickle.load(fr)

if __name__ == '__main__':
	melonData, melonLabels = createDataSet()
	melonTree = createTree(melonData, copy.deepcopy(melonLabels))
	print(melonTree)
	treePlotter.createPlot(melonTree)