#import theano
import numpy as np
from numpy import random
#from scipy.sparse import csc_matrix,csr_matrix
from sklearn.linear_model import SGDRegressor
from scipy import linalg
from itertools import islice
from sklearn.linear_model import ElasticNet
import sys
from numba import jit
from sklearn.grid_search import RandomizedSearchCV
from itertools import chain
from sklearn import metrics
import pdb

###############################
#NON THEANO 
##############################

class TSData:
    numTestRows = 3000
    batchSize = 3000
    def generateFeatures(self):
        D1 = np.loadtxt("Desktop/rnns/C1-5.dat.txt")
        D2=np.loadtxt("Desktop/rnns/C6-10.dat.txt")
        Z= np.vstack((D1,D2)) #Need to join the two parts of the data
        
        Cont_data = np.loadtxt("Desktop/rnns/C.cont.txt",comments="=")
        return (Z[:,:-1],Z[:,-1][:,np.newaxis]),(Cont_data[:,:-2], Cont_data[:,2][:,np.newaxis])
    
    @staticmethod
    def getScore(truelabel,predlabel):
        return np.sum(map(lambda (x,y):(x-y)**2,zip(truelabel,predlabel) ))/\
            np.sum(map(lambda (x,y):(y - truelabel[x-1])**2,islice(enumerate(truelabel),1,None)))
            
    @staticmethod
    def splitTrainTest(features, vals):
        trainfeat,trainvals = features[:-TSData.numTestRows,:],vals[:-TSData.numTestRows,:]
        testfeat ,testvals = features[-TSData.numTestRows:,:], vals[-TSData.numTestRows:,:]
        return (trainfeat, trainvals), (testfeat,testvals)
class SeriesDData:
    numTestRows = 3000
    def generateFeatures(self):
        D1 = np.loadtxt("Desktop/rnns/D1.dat.txt")
        D2=np.loadtxt("Desktop/rnns/D2.dat.txt")
        Z= np.vstack((D1,D2)) #Need to join the two parts of the data
        Cont_data = np.loadtxt("Desktop/rnns/D.cont.txt",comments="=")
        return (np.arange(Z.shape[0])[:,np.newaxis], Z), (np.arange(Cont_data.shape[0])[:,np.newaxis], Cont_data)
    
    @staticmethod
    def getScore(truelabel,predlabel):
        return metrics.mean_absolute_error(truelabel, predlabel)
        

class Model:
    #we assume hidden to next hidden connections.
    def __init__(self, T=None,u_size=None,y_size=None,reservoir_size=None,alpha=0.1,num_max_W = 0.01,target_spectral=0.9,\
                 scale_input_weights=1,scale_output_weights=1):
        self.reservoir_size = reservoir_size
        self.alpha = alpha
        self.num_max_W = num_max_W
        self.target_spectral = target_spectral
        self.u_size = u_size
        self.y_size = y_size
        self.T = T
        self.reservoir_size = reservoir_size
        self.scale_input_weights = scale_input_weights
        self.scale_output_weights = scale_output_weights
        #U is features x timesteps
        #W is the matrix for weights within reservoir x
        #W_in is matrix from input u
        #W_out is matrix from x to output y
        #First choose the number of nodes to fill. 10
    def initialize(self):
        memory=10
        self.reservoir_size = self.reservoir_size if self.reservoir_size != None else self.u_size * memory
        self.W = np.zeros((self.reservoir_size,self.reservoir_size))
        self.W_out = np.zeros((self.y_size,self.reservoir_size + self.u_size +1))
        self.Wfb = np.ones((self.reservoir_size, self.y_size))

        #set the values
        
        self.W_in = 0.01*random.randn(self.reservoir_size,self.u_size+1)
        def initWeights(M,numEntries):
            for i in range(self.reservoir_size):
                indices=  random.randint(0,M.shape[1]-1,numEntries)
                M[i,indices] = random.randn(1,numEntries)
            return M
        
        self.W_in = self.scale_input_weights*initWeights(self.W_in,self.num_max_W )
        self.W = initWeights(self.W, self.num_max_W)
        #self.Wfb = initWeights(np.zeros((self.reservoir_size, self.y_size)), 1)
        #self.W,_ = linalg.qr(self.W)
        #Tune W to reduce spectral radius
        
        max_eig = sorted(np.absolute(linalg.eigvals(self.W)),reverse=True)[0]
        if max_eig!=0:
            self.W = self.target_spectral*self.W/max_eig
        
        #orthonormalizing
        self.W,_ = linalg.qr(self.W)
        self.W = self.target_spectral * self.W
        
        #self.W = self.W.tocsr()

        #W_in is size of x x size of u +1
        #Un is size of u  
        #Xn is   size of x + 1  x T
        Xn = lambda Un,prevX,prevY,useoutput=False: (1- self.alpha)*prevX + self.alpha*\
        np.tanh(self.W_in.dot(np.hstack(([1],Un))) + self.W.dot(prevX) + (self.Wfb.dot(prevY).ravel() if useoutput else 0))
        #for a sequence u get x
        def getX(U,Y,train_flag):
            prevX = np.array([0]*self.reservoir_size)
            #X = np.zeros((reservoir_size ,U.shape[1]))
            X=[]
            for i in range(U.shape[0]):
                prevX = (1- self.alpha)*prevX + self.alpha*\
                np.tanh(self.W_in.dot(np.hstack(([1],U[i,:]))) + self.W.dot(prevX) +\
                         self.Wfb.dot(Y[i] if train_flag else self.clf.predict(prevX)).ravel())
                #prevX = Xn(U[i,:],prevX,Y[i] if train_flag else self.clf.predict(prevX))
                X.append(prevX)
            return X
        self.getX = getX
        #Yn = lambda Un,Xnn:self.W_out.dot(np.hstack(([1],Un,Xnn)))
        self.y = lambda U: [self.clf.predict(x1) for x1 in getX(U,None,train_flag=False)]
        #Get X from sequence x where every batchSize U is in sequence and items after batchSize are in next sequence
        def getXBatched(U,Y,batchSize,trainflag=True):
            return chain(*[self.getX(U[i*batchSize:(i+1)*batchSize], Y[i*batchSize:(i+1)*batchSize], trainflag) \
                    for i in range(0,U.shape[0]/batchSize)])
        self.getXBatched = getXBatched
    
    def fit(self, U, Y):
        self.initialize()
        #learn X
        #X = self.getX(U,Y)
        X = self.getXBatched(U,Y,TSData.batchSize)
        print("Starting to train the model...")

        #clf = ElasticNet(alpha=5,l1_ratio=0.5,max_iter=50000)
        #for x1,y1 in izip(X,Y):
        #    clf.partial_fit(x1[np.newaxis,:], y1)
        #If not using generator
        X = np.array([i for i in X])
        #X = np.array(X)
        print(X.shape)
        print(Y.shape)
        clf = SGDRegressor(n_iter=100)
        clf.fit(X,np.ravel(Y))
        print(metrics.mean_absolute_error(clf.predict(X),Y))
        print(TSData().getScore(Y, clf.predict(X)))
        self.clf = clf
        #self.WWout = linalg.pinv(X).dot(Y)
        #self.clf = lambda:None
        #self.clf.predict = lambda x:self.WWout.T.dot(x).tolist()
    
    def predict(self,X):
        return np.array(self.y(X))
    
    def get_params(self,deep=True):
        return {"T":self.T,"u_size":self.u_size,"y_size":self.y_size,"reservoir_size":self.reservoir_size,"alpha":self.alpha,"num_max_W":self.num_max_W,\
                "target_spectral":self.target_spectral,"scale_input_weights":self.scale_input_weights,\
                "scale_output_weights":self.scale_output_weights}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self

def scorer(estimator,X,Y):
    Ypred = estimator.predict(X)
    return TSData.getScore(Y, Ypred)

def do_work(reservoir_size=None,alpha=0.1,num_max_W = 10,memory=10,target_spectral=0.1):
    (features,vals),(finalTestFeats,finalTestVals) = TSData().generateFeatures() 
    (trainfeat, trainY), (testfeat, testY) = TSData.splitTrainTest(features,vals)
    #(trainfeat, trainY), (testfeat,testY) = SeriesDData().generateFeatures()
    #trainfeat,trainY = features[:-TSData.numTestRows,:],vals[:-TSData.numTestRows,:]    
    #testfeat ,testY = features[-TSData.numTestRows:,:], vals[-TSData.numTestRows:,:]
    #get mean std for each feature
    trainMean = np.mean(trainfeat,axis=0)
    trainStd = np.std(trainfeat, axis = 0)
    trans = lambda x,mn=trainMean,std=trainStd: (x - np.tile(mn,(x.shape[0],1)))/np.tile(std,(x.shape[0],1))
    
    trainYMean = np.mean(trainY,axis=0)
    trainYstd = np.std(trainY,axis=0)
    pdb.set_trace()
    trainfeat = trans(trainfeat)
    testfeat = trans(testfeat)
    #finalTestFeats = trans(finalTestFeats)
    trainY = trans(trainY,mn=trainYMean,std=trainYstd)
    testY = trans(testY, mn=trainYMean, std=trainYstd)
    #finalTestVals = trans(finalTestVals, mn=trainYMean, std=trainYstd)
    '''
    model = Model(features.shape[0],features.shape[1],vals.shape[1],reservoir_size,alpha,num_max_W ,memory,target_spectral)
    model.fit(trainfeat,trainY)
    Ypred = model.y(testfeat)
    Ytrainpred = model.y(trainfeat)
    print(str.format("train metric: {} , test metric{}", TSData.getScore(trainY, Ytrainpred), TSData.getScore(testY, Ypred)))
    
    return (testY,Ypred),(trainY,Ytrainpred)
    '''
    class distr:
        def __init__(self,lower,upper):
            self.lower = lower
            self.upper = upper
        
        def rvs(self):
            return random.rand()*(self.upper-self.lower) + self.lower
    
    params = {"reservoir_size":range(500,5000,100),"alpha":distr(0,1),"num_max_W":distr(10,100),\
              "scale_input_weights":distr(0,1),"scale_output_weights":distr(0,1)}
    model=Model(T=features.shape[0], u_size=features.shape[1],y_size=vals.shape[1])
    clf = RandomizedSearchCV(model,params,scoring = scorer,n_jobs=8,verbose=3)
    clf.fit(trainfeat,trainY)
    print(str.format("Best score: {}, params: {}", clf.best_score_, clf.best_params_))
    
def train_model(reservoir_size=None,alpha=0.1,num_max_W = 10,target_spectral=1.1):
    (features,vals),(finalTestFeats,finalTestVals) = TSData().generateFeatures()
    (trainfeat, trainvals), (testfeat, testvals) = TSData.splitTrainTest(features,vals)
    
    #(trainfeat, trainvals), (testfeat, testvals) = SeriesDData().generateFeatures()
    ##transform train/test
    trainMean = np.mean(trainfeat,axis=0)
    trainStd = np.std(trainfeat, axis = 0)
    trans = lambda x,mn=trainMean,std=trainStd: (x - np.tile(mn,(x.shape[0],1)))/np.tile(std,(x.shape[0],1))
    
    trainYMean = np.mean(trainvals,axis=0)
    trainYstd = np.std(trainvals,axis=0)
    trainfeat = trans(trainfeat)
    testfeat = trans(testfeat)
    trainvals = trans(trainvals,mn=trainYMean,std=trainYstd)
    testvals = trans(testvals, mn=trainYMean, std=trainYstd)
    
    model = Model(trainfeat.shape[0],trainfeat.shape[1],trainvals.shape[1],reservoir_size,alpha,num_max_W, target_spectral)
    
    model.fit(trainfeat,trainvals)
    Ypred = model.y(testfeat)
    Ytrainpred = model.y(trainfeat)
    print(str.format("train metric: {} , test metric{}", TSData.getScore(trainvals, Ytrainpred), TSData.getScore(testvals, Ypred)))
    #return model
    
    return (testvals,Ypred),(trainvals,Ytrainpred)
if __name__=="__main__":
    reservoir_size="reservoir_size";alpha="alpha";num_max_W = "num_max_W";memory="memory";target_spectral="target_spectral"
    args={reservoir_size:500,alpha:0.2,num_max_W:100,memory:2,target_spectral:0.8}
    if len(sys.argv):
        for k,v in eval(sys.argv[1]).iteritems():
            args[k] = v
    train_model(args[reservoir_size], args[alpha], args[num_max_W])
    #do_work(args[reservoir_size], args[alpha], args[num_max_W], args[memory],args[target_spectral])
    
