import numpy as np
#we now use linear algebra to make a more efficient, complex valued neural network
class NeuralNetwork(object):
	def __init__(self,dimensions, weights = None):
		self.ninputs = dimensions[0]
		self.noutputs = dimensions[-1]
		self.hidLayers = len(dimensions)-2
		self.dimensions = dimensions

		if weights is not None:
			n=0
			self.weights=[]
			for a in range(self.hidLayers+1):
				x=self.dimensions[a]
				y=self.dimensions[a+1]
				self.weights.append(np.array(weights[n:n+x*y]).reshape(y,x))
				n+=x*y
		else:
			weights=[]
			for a in range(self.hidLayers+1):
				b=2*np.random.rand(self.dimensions[a+1],self.dimensions[a])-1
				c=(2*np.random.rand(self.dimensions[a+1],self.dimensions[a])-1)*1j
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
