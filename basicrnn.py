import theano
import numpy as np
from theano import tensor as T 
from collections import OrderedDict
import os,math,sys
from random import shuffle
from theano.ifelse import ifelse

class XORData:
    numTestRows = 50
    def generateFeatures(self,folder, saveFeatures):
        Z=np.loadtxt("xor.txt")
        return Z[:,0][:,np.newaxis], Z[:,1][:,np.newaxis]
    
    def generateFile(self):
        batchSize = 100
        Y=[];X=[]
        for i in xrange(10000):
            batch_x = np.random.randint(0,1,(batchSize,1))
            Y.append(batch_x ^ np.vstack(([0],batch_x))[:-1])
            X.append(batch_x)
        return np.array(X)[:,0], np.array(Y)[:,0]
    
def shuffleOrder(lenarr, sentenceSize):
    arr = range(lenarr)
    items = range(0,int(math.ceil(lenarr*1.0/sentenceSize)))
    shuffle(items)
    return reduce(lambda x,y: x + arr[y*sentenceSize:(y+1)*sentenceSize],items,[])


class model:
    names = ['Wx', 'Wh', 'W', 'bh', 'b', 'h0']
    def __init__(self,nout,nh,nf,batch_size=100, activation=lambda x:x):
        '''
        nh: dimension of hidden layer
        nf: number of features
        '''
        #set the non-zero Ws 
        print("nout:{} nh:{} nf:{}".format(nout,nh,nf))
        self.Wx  = theano.shared(np.random.uniform(0,0.02,size=(nf, nh)).astype(theano.config.floatX)) #This is Whv
        self.Wh  = theano.shared(np.random.uniform(0,0.02,size=(nh, nh)).astype(theano.config.floatX))#this is Whh 
        self.W   = theano.shared(np.random.uniform(0,0.02,size=(nh, nout)).astype(theano.config.floatX))#This is Whz
        self.bh  = theano.shared(np.zeros(nh, dtype=theano.config.floatX))
        self.b   = theano.shared(np.zeros(nout, dtype=theano.config.floatX))
        self.h0  = theano.shared(np.zeros(nh, dtype=theano.config.floatX))
        
        self.params = [ self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
        self.names = model.names

        #give as input the features x and the value of hidden units before this one
        def recurrence(x_t,h_tm):
            #TODO append to x_t the features from y_T
            h_t = T.dot(x_t, self.Wx) + T.dot(h_tm, self.Wh) + self.bh
            s_t = activation(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]
        
        self.x = T.fmatrix() #input features
        
        [self.h, self.s],_ = theano.scan(fn = recurrence, sequences=self.x, outputs_info=[self.h0,None])        
        
        #cost 
        self.y = T.fmatrix() #real labels
        #sqrd loss
        #self.cost = (T.sum((T.sub(self.s,self.y))**2) + 0.001*(self.Wx.norm(2) + self.Wh.norm(2) + self.W.norm(2)))/batch_size
        
        #log loss
        z= T.nnet.sigmoid(self.s)
        self.cost= T.sum(-self.y*T.log(z + 1e-14) - (1-self.y)*T.log((1-z) + 1e-14)) 
        #self.predict = theano.function(inputs=[self.x],outputs=self.s,on_unused_input="ignore")
        self.predict = theano.function(inputs=[self.x],outputs=z,on_unused_input="ignore")
        
    def setModel(self,train_x,train_y,batch_size,uselrdict=False):
        self.shared_train_x = theano.shared(train_x.astype(theano.config.floatX),borrow=True)
        self.shared_train_y = theano.shared(train_y.astype(theano.config.floatX),borrow=True)
        gradients = T.grad(self.cost,self.params)
        lr = T.scalar('lr')
        lrdict = T.vector("lrdict")
        index = T.lscalar("index")        
        if theano.config.device == "cpu":
            gradients = [theano.printing.Print("gradient "+n)(g) for (n,g) in zip(self.names,gradients)]
        learning_rate = lr
        if uselrdict:            
            item2index = dict(zip(self.names,range(len(self.names))))
            learning_rate = lrdict[item2index][n]
        updates = OrderedDict(( p, p-learning_rate*ifelse(T.gt(g.norm(2), 10), 10*g/g.norm(2), g) )\
                               for p, g, n in zip( self.params, gradients,self.names))
        self.train = theano.function( inputs = [lr,index,lrdict],\
                                  outputs = [self.cost,self.h,self.s],\
                                  updates = updates,\
                                  givens={self.x:self.shared_train_x[index * batch_size: (index + 1) * batch_size],\
                                          self.y:self.shared_train_y[index * batch_size: (index + 1) * batch_size]},\
                                     on_unused_input="warn"\
                                 )
    def save(self, folder): 
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, name + '.npy'), param.get_value())
    
    @classmethod
    def load(cls,folder):
        loadedvars = {}
        for name in model.names:
            loadedvars[name] = np.load(os.path.join(folder, name + '.npy'))
        #Now create a model object
        rnn = cls(loadedvars["W"].shape[1], loadedvars["Wh"].shape[0], loadedvars["Wx"].shape[0])
        
        for param, name in zip(rnn.params, model.names):
            param.set_value(loadedvars[name])
        return rnn
        
def do_work(userArgs=None):
    #just the text strings for our var names
    nepochs = "nepochs"; nh="nh"; sentenceSize="sentenceSize"
    batchSize="batchSize"; folder="folder"; clr="clr"; clrdecay="clrdecay"
    saveFeatures="saveFeatures"; loadFile="loadFile"; outfile="outfile"; lr_change="lr_change";testnum="testnum"
    
    #default args
    args={
          clrdecay:0.9,clr:0.1,nepochs:10, nh:5,\
          sentenceSize:30,batchSize:50,folder:"C:/Users/hichando/Desktop/rnns/",\
          saveFeatures:True,\
          loadFile:"C:/Users/hichando/Desktop/Hackathon_MSStore/data_wellformed_utf8.csv",\
          outfile:"pred.txt",lr_change:1,testnum:-1\
          }
    #Dictionary of initial learning rates
    lrdict = OrderedDict([("Wx",1e-5), ("Wh",0.01),("W",0.01),("bh",0.01),("b",8.0),("h0",0.01)])
    
    if userArgs:
        for k,v in eval(userArgs).iteritems():
            args[k] = v
    print args
    nepochs=args[nepochs]; nh=args[nh];sentenceSize = args[sentenceSize];
    features, Y = XORData().generateFeatures(folder= args[folder],saveFeatures=args[saveFeatures] )
    features = features.astype(theano.config.floatX)
    Y = features.astype(theano.config.floatX)
    numTestRows = XORData.numTestRows if args[testnum]<0 else args[testnum]
    
    #if batchsize is larger than max elements or negative, batchSize is equal to num elements - numtestrows
    if args[batchSize]<0 or args[batchSize]>Y.shape[0]-numTestRows: args[batchSize] = Y.shape[0] - numTestRows
    print("Total data read:{}. Number testrows{}".format(features.shape,numTestRows))
    rnn = model(Y.shape[1], nh, features.shape[1],args[batchSize])
    
    train = features[:-numTestRows]
    test = features[-numTestRows:]
    Y = Y[:-numTestRows]
    
    #get mean std for each feature
    trainMean = np.mean(train,axis=0)
    trainStd = np.std(train, axis = 0)
    
    trans = lambda x,mn,std: (x - np.tile(mn,(x.shape[0],1)))/np.tile(std,(x.shape[0],1))
    #invtrans = lambda x,mn,std: x*np.tile(std,(x.shape[0],1)) + np.tile(mn,(x.shape[0],1))
    
    tf = lambda x: trans(x,trainMean,trainStd)
    curr_lr = args[clr]
    try:
        for epoch in range(nepochs):
            print("Running epoch {}.".format(epoch))
            args[batchSize] = args[batchSize] if args[batchSize] < train.shape[0] else train.shape[0]
            rnn.setModel(tf(train), Y,args[batchSize])
            totalcost=[]
            for i in xrange(train.shape[0] / args[batchSize]):
                #print rnn.Wh.get_value()
                [cost, h0, currs] = rnn.train(curr_lr,i, np.array(lrdict.values(), dtype=theano.config.floatX))                
                print("Minibatch cost {}".format(cost.tolist()))
                '''
                print "Wx:" + rnn.Wx.get_value().__str__()
                print "Wh:" + rnn.Wh.get_value().__str__()
                print "W:" + rnn.W.get_value().__str__()
                print "bh:" + rnn.bh.get_value().__str__()
                print "b:" + rnn.b.get_value().__str__()
                print "bh:" + rnn.bh.get_value().__str__()
                '''
                totalcost.append(cost.tolist())
            print("Finished {} epochs. Current metric: {}".\
                  format(epoch+1,  np.mean(totalcost)))        
            
            if np.mod(epoch+1,args[lr_change]) == 0:
                curr_lr *= args[clrdecay]
                for k,v in lrdict.iteritems():
                    lrdict[k] *= args[clrdecay]
                
    except KeyboardInterrupt:
        print("Interrupted")
        
    pred  = np.array(rnn.predict(tf(test).astype(theano.config.floatX)))
    trainresults = np.array(rnn.predict(tf(train).astype(theano.config.floatX)))
    np.savetxt(args[folder]+args[outfile], pred, delimiter=",")
    rnn.save(args[folder])
    rnn.save
    np.savetxt("c:/Users/hichando/Desktop/pred-train.txt", trainresults, delimiter=",")

if __name__ == "__main__":
    do_work() if len(sys.argv)==1 else do_work(sys.argv[1])
