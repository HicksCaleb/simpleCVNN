import numpy as np
#we now use linear algebra to make a more efficient, complex valued neural network
class NeuralNetwork(object):
	def __init__(self,ninputs,noutputs, hidNodes, hidLayers, weights = None):
		self.ninputs = ninputs
		self.noutputs = noutputs
		self.hidNodes = hidNodes
		self.hidLayers = hidLayers
		dimensions = []
		dimensions.append(ninputs)
		for a in range(hidLayers):
			dimensions.append(hidNodes)
		dimensions.append(noutputs)
		if weights is not None:
			n=0
			self.weights=[]
			for a in range(hidLayers+1):
				x=dimensions[a]
				y=dimensions[a+1]
				self.weights.append(np.array(weights[n:n+x*y]).reshape(y,x))
				n+=x*y
		else:
			weights=[]
			for a in range(hidLayers+1):
				b=np.random.rand(dimensions[a+1],dimensions[a])
				c=np.random.rand(dimensions[a+1],dimensions[a])*1j
				weights.append(b+c)
			self.weights = weights
	def g(self,x):
		return np.tanh(x)/(1-(x-3)*np.exp(-x))	
	def f(self,a):
		return list(map(lambda x: self.g(x.real) + self.g(x.imag)*1j, a))
	def processLayer(self,n):
		if n>0:
			return self.f(np.dot(self.weights[n-1],self.processLayer(n-1)))
		else:
			return self.input
	def feedforward(self, input):
		self.input = input
		return self.processLayer(self.hidLayers+1)
	def exportWeights(self):
		exList = []
		for a in self.weights:
			exList = np.append(exList, a.ravel())
		return exList