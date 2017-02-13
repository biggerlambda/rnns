import numpy as np
from numpy import random, newaxis
from scipy import linalg
import theano
from theano import tensor as T
from collections import OrderedDict
from itertools import islice
import sys

###############################
#NON THEANO 
##############################
#theano.config.compute_test_value = 'warn'
class TSData:
    numTestRows = 3000
    def generateFeatures(self):
        D1 = np.loadtxt("data/rnns/C1-5.dat.txt")
        D2=np.loadtxt("data/rnns/C6-10.dat.txt")
        Z= np.vstack((D1,D2)) #Need to join the two parts of the data
        return Z[:,:-1],Z[:,-1][:,np.newaxis]

    @staticmethod
    def getScore(truelabel,predlabel):
        return np.sum(map(lambda (x,y):(x-y)**2,zip(truelabel,predlabel) ))/np.sum(map(lambda (x,y):(y - truelabel[x-1])**2,islice(enumerate(truelabel),1,None)))
    
class Model:
    #we assume hidden to next hidden connections.
    def __init__(self,u_size,y_size,reservoir_size,alpha, num_max_W, memory, target_spectral):
        #timesteps = 5
        #U is features x timesteps
        #W is the matrix for weights within reservoir x
        #W_in is matrix from input u
        #W_out is matrix from x to output y
        #First choose the number of nodes to fill. 10
        reservoir_size = reservoir_size if reservoir_size != None else u_size * memory
        
        self.alpha = alpha
        #set the values
        
        def initWeights(M,numEntries):
            for i in range(M.shape[0]):
                indices=  random.randint(0,M.shape[1]-1,numEntries)
                M[i,indices] = random.randn(1,numEntries)
            return M
        
        self.W_in = initWeights(random.rand(reservoir_size,u_size+1),num_max_W).astype(theano.config.floatX)
        
        initM = initWeights(np.zeros((reservoir_size,reservoir_size)), num_max_W)
        max_eig = sorted(np.absolute(linalg.eigvals(initM)),reverse=True)[0]
        if max_eig!=0:
            initM = initM*target_spectral/max_eig
        self.W = initM.astype(theano.config.floatX)
        #These are the weights that would be tuned
        self.W_out = theano.shared(np.zeros((y_size,reservoir_size + u_size +1)).astype(theano.config.floatX))
        #self.W_fb = theano.shared(np.zeros((reservoir_size, y_size)))
        
        #W_in is size of x x size of u +1
        #Un is size of u  
        #Xn is   size of x + 1  x T
        #for a sequence u get x
        def recurrence(u_t,prevX):
            x_t = (1-self.alpha)*prevX + self.alpha*T.tanh\
            (T.dot(self.W_in, T.vertical_stack(T.as_tensor_variable(np.ones((1,1)).astype(theano.config.floatX)),u_t[:,np.newaxis])[:,0])\
                                                            + T.dot(self.W,prevX))
            return x_t
        u = T.fmatrix()
        #provide with random input
        #u.tag.test_value =np.random.rand(5,2).astype(theano.config.floatX)
        x,_ = theano.scan(fn = recurrence, sequences=u, outputs_info=[T.zeros((reservoir_size)).astype(theano.config.floatX)])
        timesteps = T.iscalar()
        y = T.dot(self.W_out, T.vertical_stack(T.ones((1,timesteps)).astype(theano.config.floatX),u.T,x.T))
        self.predict = theano.function(inputs=[u,timesteps],outputs=y)
        #the true labels
        y0 = T.fmatrix()
        #provide with random input
        #y0.tag.test_value = np.random.rand(5,1).astype(theano.config.floatX)
        cost = T.sum((y.T - y0)**2)
        #cost = T.sum(y**2)
        g = T.grad(cost,self.W_out)
        lr = T.scalar()
        updates = OrderedDict([(self.W_out, self.W_out - lr*g)])
        self.train = theano.function(inputs=[u,y0,lr,timesteps],outputs=cost,updates=updates,on_unused_input='warn')

def do_crossValidation(params, func, data):
    raise NotImplementedError
    
def do_work(userargs=None):
    alpha="alpha";reservoir_size="reservoir_size";num_max_W="num_max_w";lr="lr";memory="memory";target_spectral="target_spectral";batch_size="batch_size"
    numepochs = "numepochs"
    defaultargs = {numepochs:10, batch_size:1000,lr:0.01,alpha:0.1,reservoir_size:500,num_max_W:10,memory:10,target_spectral:0.5,batch_size:10000}
    minbatchsize= np.inf
    args = defaultargs
    if userargs:
        for k,v in eval(userargs).iteritems():
            args[k] = v
    features,vals = TSData().generateFeatures()
    features = features.astype(theano.config.floatX)
    vals = vals.astype(theano.config.floatX)
    #features=features[:5000,:];vals=vals[:5000,:]
    trainfeat,trainvals = features[:-3000,:],vals[:-3000,:]
    testfeat ,testvals = features[-3000:,:], vals[-3000:,:]
    #get mean std for each feature
    trainMean = np.mean(trainfeat,axis=0)
    trainStd = np.std(trainfeat, axis = 0)
    trans = lambda x,mn=trainMean,std=trainStd: (x - np.tile(mn,(x.shape[0],1)))/np.tile(std,(x.shape[0],1))
    trainfeat = trans(trainfeat)
    testfeat = trans(testfeat)
    model = Model(features.shape[1],vals.shape[1], args[reservoir_size],args[alpha],args[num_max_W], args[memory],args[target_spectral])
    #if size is too less then just do regular grad descent otherwise sgd
    if trainfeat.shape[0] < minbatchsize:
        costnow = np.inf
        cost = 0
        while costnow - cost>1e-1:
            cost = costnow
            costnow = model.train(trainfeat,trainvals,args[lr],trainfeat.shape[0])
            print(costnow, cost)
    else:
        epoch=0
        numepochs = args[numepochs]; batch_size=args[batch_size]
        while epoch < numepochs:
            it =0 
            while it*batch_size <=trainfeat.shape[1]:
                model.train(trainfeat[:,it*batch_size:iter*(batch_size+1)],args[lr],0)
                it +=1
            epoch +=1
    Ypred = model.predict(testfeat,testfeat.shape[0])
    Ytrainpred = model.predict(trainfeat, trainfeat.shape[0])
    #Save both of these
    np.savetxt("data/Ypred.txt", np.hstack((testvals,Ypred.T)), delimiter=".")
    np.savetxt("data/Ytrainpred.txt", np.hstack((trainvals,Ytrainpred.T)), delimiter=".")
    #end save
    print(str.format("train metric: {} , test metric{}", TSData.getScore(trainvals, Ytrainpred), TSData.getScore(testvals, Ypred)))
    return (testvals,Ypred),(trainvals,Ytrainpred)


if __name__=="__main__":
    do_work(sys.argv[1] if len(sys.argv) else None)
