
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import logging
import os



class PyGNG:
    def __init__(
        self, 
        maxNeurons:int,
        ageMax:int,
        iterMax: int = 3000,
        eps: float=0.0001,
        epsb: float=0.14,
        epsn: float=0.006,
        lambda1: int =100,
        alpha: float=0.5,
        delta: float=0.995,
        show: bool=True,
        lambdaPlot: int =200
    ):
        self.maxNeurons = maxNeurons
        self.ageMax = ageMax
        self.iterMax = iterMax
        self.eps = eps
        self.epsb = epsb
        self.epsn = epsn
        self.lambda1 = lambda1
        self.alpha = alpha
        self.delta = delta
        self.show = show
        self.lambdaPlot = lambdaPlot

        self.A = np.zeros((self.maxNeurons, self.maxNeurons), dtype=np.int32)
        self.Aage = np.zeros((self.maxNeurons, self.maxNeurons), dtype=np.int32)
        self.active = np.zeros(self.maxNeurons, dtype=np.int32)
        
        self.xn = None
        self.ndims = None
        self.npts = None


    def _addRandomNeuron(self, data: np.ndarray) -> int:
        idx = np.random.randint(0,self.ndims)
        id = np.where(self.active==0)[0][0]
        self.xn[id,:] = data[idx,:] 
        self.active[id] = 1
        return id


    def _addNewNeuron(self, x1: np.ndarray, x2: np.ndarray) -> int:
        id = np.where(self.active==0)[0][0]
        logging.debug(f"add neuron @ {id}")
        self.xn[id,:] = 0.5*(x1 + x2)
        self.active[id] = 1
        return id

    def _findNeighborIdxs(self, id:int) -> np.ndarray:
        return np.where(self.A[id,:]==1)


    def fit(self, data:np.ndarray, plotPath:str="./plots") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fits the neural gas to the given data array. Data should be normalized in each direction. 
        For details see https://proceedings.neurips.cc/paper/1994/file/d56b9fc4b0f1be8871f5e1c40c0067e7-Paper.pdf

        Args:
            data (np.ndarray): Ndimensional Numpy array with dimension [npts, ndims]
            plotPath (str): path where plots are stored if show flag is True

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: [maxNeurons, ndims] = Node vectors, [maxNeurons] = Node active, [maxNeurons, maxNeurons] = Adjancency Matrix
        """

        # Get dimensions of data
        self.ndims = data.shape[1]
        self.npts = data.shape[0]

        # Init arrays
        self.currentNeurons = 0
        self.xn = np.zeros((self.maxNeurons, self.ndims))
        xn0 = np.zeros((self.maxNeurons, self.ndims))
        errorCumSum = np.zeros((self.maxNeurons))

        iter, eps = 0, 1e+7


        if self.show:
            import os
            plotNames = []
            if not os.path.exists(plotPath):
                os.mkdir(plotPath)


        while iter < self.iterMax and eps > self.eps:

            logging.debug(f"Iter: {iter} Neurons: {len(np.where(self.active == 1)[0])}/{self.maxNeurons-1}")

            # Add initial neurons when to small
            if len(np.where(self.active == 1)[0]) < 2:
                id0 = self._addRandomNeuron(data=data)
                id1 = self._addRandomNeuron(data=data)
                self.A[id0,id1], self.A[id1,id0] = 1,1

            # Store previous last position
            xn0 = self.xn.copy()

            # Only consider valid neurons
            idxNeuronsValid = self.active == 1

            # Generate random signal and calculate errors
            idxSig = np.random.randint(0,self.npts)
            d = np.square( self.xn[idxNeuronsValid] - data[idxSig,:]).sum(axis=1)

            # Find BPU and SBPU
            ibpu = np.argsort(d)[:2]

            # Add squared distance between input signal and BPU to local counter var
            errorCumSum[ibpu[0]] += d[ibpu[0]]

            # incrementAgeOfBPUNeighbors from s1
            idxNeighbors = self._findNeighborIdxs(ibpu[0])
      
            for idNeighbor in idxNeighbors[0]:
                self.Aage[idNeighbor,ibpu[0]] += 1
                self.Aage[ibpu[0],idNeighbor] += 1

            # Move Neighbors towards s1
            self.xn[idxNeighbors,:] += self.epsn*(data[idxSig,:] - self.xn[idxNeighbors,:])
            self.xn[ibpu[0],:] += self.epsb*(data[idxSig,:] - self.xn[ibpu[0],:])

            
            #  If s1 and s2 are connected by an edge, set the age of this edge to zero. If such an edge does not exist, create it
            self.A[ibpu[0],ibpu[1]], self.A[ibpu[1],ibpu[0]] = 1, 1
            self.Aage[ibpu[0],ibpu[1]], self.Aage[ibpu[1],ibpu[0]] = 0, 0

            # Remove edges with an age larger than amax
            idx, idy = np.where(self.Aage >= self.ageMax)
            for i,j in zip(idx, idy):
                self.A[i,j], self.A[j,i] = 0, 0
                self.Aage[i,j], self.Aage[j,i] = 0, 0

            # Remove nodes without any edge
            idx_del = np.where(self.A.sum(1) == 0)[0]
            self.active[idx_del] = 0

            # If number of signals are a multiple of lambda1
            if iter % self.lambda1 == 0:

                # Detertmine worst cumsum
                q = np.argmax(errorCumSum)
                idxNeighbors = self._findNeighborIdxs(q)
                f = np.argmax(errorCumSum[idxNeighbors])

                if len(np.where(self.active == 1)[0]) < self.maxNeurons-1:
                    w = self._addNewNeuron(x1=self.xn[q,:], x2=self.xn[f,:])

                    self.A[q,f], self.A[f,q] = 0, 0
                    self.Aage[q,f], self.Aage[f,q] = 0, 0

                    self.A[q,w], self.A[w,q] = 1, 1
                    self.A[f,w], self.A[w,f] = 1, 1
                    self.Aage[q,w], self.Aage[w,q] = 0, 0
                    self.Aage[f,w], self.Aage[w,f] = 0, 0

                    errorCumSum[q] = self.alpha*errorCumSum[q]
                    errorCumSum[f] = self.alpha*errorCumSum[f]
                    errorCumSum[w] = errorCumSum[q]

            # Decrease all error variables by multiplying them with a constant d
            errorCumSum[:] = self.delta*errorCumSum[:]

            eps = np.abs(self.xn - xn0).sum()

            if iter % 100 == 0:
                logging.info(f"Iter: {iter} eps: {eps:.4f}")

            iter+=1

            # ============== Ploting only ==============
            # Plottin
            if iter % self.lambdaPlot == 0 and self.show:

                plt.scatter(data[:,0], data[:,1], color="gray")

                for id1 in np.where(self.active==1)[0]:
                    
                    for id2 in self._findNeighborIdxs(id1)[0]:
                        plt.plot([self.xn[id1,0], self.xn[id2,0]], [self.xn[id1,1], self.xn[id2,1]],'b-')
                        plt.plot([self.xn[id1,0], self.xn[id2,0]], [self.xn[id1,1], self.xn[id2,1]],'r.')

                plotNames.append(f"./plots/{iter}_test.png")

                plt.title(f"Iter: {iter}")
                plt.grid(True)
                plt.savefig(f"./plots/{iter}_test.png")
                plt.close()

        # ============== Ploting only ==============
        if self.show:
            logging.debug("creating gif...")        
            import os
            os.system(f"convert {' '.join(plotNames)} gng.gif")

        
        return self.xn, self.active, self.A



if __name__ == "__main__":


    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


    data = np.random.normal(size=(500,2))

  #  data = np.loadtxt("GSE156455_201106_early.200pcs.csv", delimiter=';', usecols=list(range(1,20)), skiprows=1)
  #  xmin = data.min(0)
  #  xmax = data.max(0)
  #  data = (data-xmin)/(xmax-xmin)
   # data = data[:,:2]

    gng = PyGNG(maxNeurons=60, ageMax=25, iterMax=20000)
    gng.fit(data)


