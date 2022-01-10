import numpy as np
import os 
import matplotlib.pyplot as plt



class PyRRT:
    def __init__(self, maxIter:int, maxNodes: int, stepSize:float):
        self.maxIter = maxIter
        self.maxNodes = maxNodes
        self.stepSize = stepSize

        self.currentIdx = 0
        self.ndims = 0


    def run(self, x0, xlo, xup):

        self.ndims = x0.shape[1]

        xn = np.zeros((self.maxNodes, self.ndims))
        active = np.zeros(self.maxNodes)
        A = np.zeros((self.maxNodes,self.maxNodes))
        
        # Init condition
        xn[0,:] = x0
        active[0] = 1
        self.currentIdx = 1

        plotnames = []
        ctr = 0
        while True:

            # Add node & Finf closest neighbor
            xrand = np.random.rand(self.ndims)

            idx = np.argmin(np.square(xn[active==1,:]-xrand).sum(1))

            t = np.expand_dims((xrand - xn[idx,:]),0)
            t = t/np.sqrt(np.square(t).sum(1))
   
            # Calculate new node
            xnew = xn[idx,:] + self.stepSize*t[0,:]

            # Check if new node is in obstacle

            # Determine best neighbor for trimmed node
            idxN = np.argmin(np.square(xn[active==1,:]-xnew).sum(1))

            # Add new node to list
            xn[self.currentIdx,:] = xnew
            active[self.currentIdx] = 1

            A[idxN,self.currentIdx], A[self.currentIdx, idxN] = 1, 1
            self.currentIdx += 1
            
            # Check if new node is in goal area


            # Plot 
            if ctr % 3 == 0:
                for i in range(self.currentIdx-1):
                    inghbrs = np.where(A[i,:])[0]

                    for j in inghbrs:
                        plt.plot([xn[i,0], xn[j,0]], [xn[i,1], xn[j,1]], 'b-')
                        plt.plot([xn[i,0], xn[j,0]], [xn[i,1], xn[j,1]], 'r.')
                        
                plt.axis([0,1,0,1])
                plt.savefig(f"./plots/plot_{ctr}.png")
                plt.close()

                plotnames.append(f"./plots/plot_{ctr}.png")


            # Repeat
            ctr += 1
            if ctr >= self.maxIter or self.currentIdx >= self.maxNodes-1:
                break

        
        os.system(f"convert {' '.join(plotnames)} rrt.gif")



if __name__ == "__main__":

    x0 = np.asarray([[0.5,0.5]])

    p = PyRRT(maxIter=300, maxNodes=300, stepSize=0.02)
    p.run(x0, None, None)

