import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as opt
def LoadData():
    f = open('ex2data2.txt',encoding='utf8')
    data = pd.read_csv(f,header=None,names=['test1','test2','accepted'])
    positive=data[data['accepted'].isin(['1'])]#正样本
    negtive=data[data['accepted'].isin(['0'])]
    return positive,negtive,data

def ScanData():
    pos,neg,data=LoadData()
    #print(data.describe())
    print(data.head())
    plt.scatter(pos['test1'],pos['test2'],marker='^',color='red',label='accepted')
    plt.scatter(neg['test1'],neg['test2'],marker='3',color='blue',label='refused')
    plt.legend()
    plt.xlabel('test1')
    plt.ylabel('test2')
    #plt.show()
    return plt

def Sigmoid(z):
    return 1/(1+np.exp(-z))

def Cost(theta,x,y):
    part1 = np.log(Sigmoid(x.dot(theta))).dot(-y)
    part2 = np.log(1-Sigmoid(x.dot(theta))).dot(1-y)
    return (part1-part2)/(len(x))

def Gradient(theta,x,y):
    return x.transpose().dot(Sigmoid(x.dot(theta))-y)/len(x)

def regularized_cost(theta,x,y,Lambda=2):
    return Cost(theta,x,y)+(Lambda/(2*len(x)))*(theta.dot(theta))

def regularized_gradient(theta,x,y,Lambda=2):
    return Gradient(theta,x,y)+Lambda*theta/len(x)

def feature_mapping(x1, x2, power=8):
    data = {}
    for i in np.arange(power + 1):
        for p in np.arange(i + 1):
            data["f{}{}".format(i - p, p)] = np.power(x1, i - p) * np.power(x2, p)
    return pd.DataFrame(data)

def DecisionBoundary(theta):
    x=np.linspace(-1,1.25,200)
    x1,x2=np.meshgrid(x,x)
    z=feature_mapping(x1.reshape(-1,),x2.reshape(-1,)).dot(theta).values
    z=z.reshape(x1.shape)
    plt=ScanData()
    plt.title('DecisionBoundary')
    plt.contour(x1,x2,z,0)
    plt.show()

def main():
    pos,neg,data=LoadData()
    fdata=feature_mapping(data['test1'],data['test2'])
    x=fdata.values
    y=data['accepted']
    theta=np.zeros(x.shape[1])
    parameters=opt.minimize(fun=regularized_cost,x0=theta,method='tnc',jac=regularized_gradient,args=(x,y,1))
    theta=parameters.x
    print(parameters)
    DecisionBoundary(theta)
if __name__ == '__main__':
   main()

