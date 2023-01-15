#ifndef NEURALNETWORK_H 
#define NEURALNETWORK_H 
#endif
#include <iostream>
#include <cmath>
#include "Matrix.h"
#include <functional>
#include <vector>
using namespace std;
//WE create our Neuron Network here, defining an individual Neuron class and more

/*
	Input       Layer 1:             Layer 2 (last layer):
	
																	 [2.0 Neuron | F(pF(in))] \
																							...            \
																 > [2.1 Neuron | F(pF(in))] ----> [FOutput 1] 
																/												     /
  [in] -> [1.0 Neuron | F(in) ] -> [2.2 Neuron | F(pF(in))] /
																\
																 > [2.3 Neuron | F(pF(in))] 
																/	
	[in] -> [1.1 Neuron | F(in) ] -> [2.4 Neuron | F(pF(in))] \
																\                            \
																 > [2.5 Neuron | F(pF(in))] ----> [FOutput 2]
																							...            /
																	 [2.6 Neuron | F(pF(in))] /


 ***pF(in) = output of previous Function(in)
 *** EACH NEURON HAS A CONNECTION TO EVERY SIINGLE NEURON IN THE PROCEEDING LAYER, OVERLAPPING 
 
 ***NOTE: 1.0 Neuron signifies that this is the first Neuron (index = 0) of Layer '1'

 -Since each Neruon is webbed together this, in a 1-D plane, looks like the following:

 [1.0] -> [2.0], [2.1], [2.2], [2.3], [2.4], [2.5], [2.6]
 [1.1] -> [2.0], [2.1], [2.2], [2.3], [2.4], [2.5], [2.6]

 - There are a total of 2 * 7 connections, or 14 Connections 
 - ... which we can REPRESENT in a 2 by 7 MATRIX (or 7 by 2)

***NOTE: Each connection throughout all our layers will reresent a Weight:
 - so [1.0] -> [2.0] & [1.0] -> [2.1] are TWO SEPERATE connections w/ DISTINCT WEIGHTS which wil be stored in our MATRIX
 - WEIGHTS determine how much one neuron INFLUENCE the next one

 - [ w([1.0] -> [2.0]) , w([1.1] -> [2.0]) ]    This is a Weight Matrix, where each number is the output of a Weight function which takes in a connection btwn Neuron A and B
 - [ w([1.0] -> [2.1]) , w([1.1] -> [2.1]) ]
 - [ w([1.0] -> [2.2]) , w([1.1] -> [2.2]) ]
 - [ w([1.0] -> [2.3]) , w([1.1] -> [2.3]) ]
 - [ w([1.0] -> [2.4]) , w([1.1] -> [2.4]) ]
 - [ w([1.0] -> [2.5]) , w([1.1] -> [2.5]) ]
 - [ w([1.0] -> [2.6]) , w([1.1] -> [2.6]) ]

 ***NOTE if we add a third layer of Neurons, we simply add a new column w this corresponding form

  [ w([2.0] -> [3.0]) , w([2.1] -> [3.0]) ... w([2.6] -> [3.0]) ]
  [ w([2.0] -> [3.1]) , w([2.1] -> [3.1]) ... w([2.6] -> [3.1]) ]
  [ w([2.0] -> [3.2]) , w([2.1] -> [3.2]) ... w([2.6] -> [3.2]) ]
															...
  [ w([2.0] -> [3.N]) , w([2.1] -> [3.N]) ... w([2.6] -> [3.N]) ]

	- However, to calculate the propensity of each neuron from a SPECIFIC LAYER
	- We use a precise mathematical formula to calculate such a value:
	- [2.0] = [1.0] * w([1.0] -> [2.0]) + [1.1] * w([1.1] -> [2.0])
	- [2.1] = [1.0] * w([1.0] -> [2.1]) + [1.1] * w([1.1] -> [2.1])
	- [2.2] = [1.0] * w([1.0] -> [2.1]) + [1.1] * w([1.1] -> [2.1])

	- [3.0] = [2.0] * w([2.0] -> [3.0]) + [2.1] * w([2.1] -> [3.0]) + [2.2] * w([2.2] -> [3.0]) + [2.3] * w([2.3] -> [3.0]) ...
	- [3.1] = [2.0] * w([2.0] -> [3.1]) + [2.1] * w([2.1] -> [3.1]) + [2.2] * w([2.2] -> [3.1]) + [2.3] * w([2.3] -> [3.1]) ...
	...

	- Current Neuron Val =  Sum( (NodePrevi * w(NodePrevi -> CurrNeuron)) , (NodePrevi+1 * w(NodePrevi+1 -> CurrNeuron)) , (NodePrevi+2 * w(NodePrevi+2 -> CurrNeuron)), ..... )

	- i.o.w -> The sum of each Neuron from the N-1 layer * weight of the connection from that specific Neuron to our Desired Neuron 
- WE are ACTIVATING that neuron, giving it a value (information) it can pass on to the next layer
- NOTE: This is known as the Activated Value, transfered through each layer

*/

inline float Sigmoid(float x){ //Will squish values that are passsed onto neurons to be btween 0 and 1
	return 1.0 / ( 1 + exp(-x) ); //The Sigmoid function has bounds btween y = 0 & y = 1
}

inline float DeriSigmoid(float x){
	return (x) * (1 - (x)); //Where x = Sigmoid(in)
}


class NeuralNetwork{
	public:
	//Our tensors are initialized here (a vector holding Matrices)
	std::vector<uint32_t> _topology; //Initialization of a vector whcih will hold the number of neurons at Each layer
	// Example: our network above would be <2,7,2>
	std::vector<Matrix> _weightMatrix; //Holds the weight for each neuron calc from 2 CONSECUTIVE Layers
	std::vector<Matrix> _NeuronValue; //Holds the 'index' of each Neuron
	// The neuronVal matrix will consist of N by 1 matrices - mimicing the ac structure of our nerual networl
	// <[10,11], [20,21,22,23,24,25,26], [30,31....]>
	
	std::vector<Matrix> _biasMatrix; //Our Bias matrix, holds an offset value for the activation value
	float _learningRate; //cte

public:
	//std is a namespace for the Standard Template Libraries, and VECTORS are stl functions
	NeuralNetwork(std::vector<uint32_t> topology, float learningRate = 0.1f): //Our Constructor with initializer list for our corresponding matrices
		_topology(topology),
		_weightMatrix({}),
		_NeuronValue({}),
		_biasMatrix({}),
		_learningRate(learningRate)
	{

		for(uint32_t i = 0; i < _topology.size() - 1; i++){ //This will parse thru each 'layer' except the last one 
			//our weightmatrix has as many coloumns as layers, since each col will hold the neurons corresponding to one layer
			int cols = _topology[i+1]; //rmbr topology is simply a vector of ints , where each int reps number of neurons in an individual layer
			int rows = _topology[i];

			Matrix weightMatrix(cols, rows);
			weightMatrix = weightMatrix.applyFunction([](const float& f){ return (float)rand() / RAND_MAX; }); //We pass in a 'Lamda' function as our method param 
			//This initializes our weightMatrix w/ random values between 0 & 1;
			_weightMatrix.push_back(weightMatrix); //Inserting each Matrix into our TENSOR

			//We then push this Matrix into our Vector<Matrix> _weightM which is a dynamic container/TENSOR (collection of matrices)
			
			//We need one bias term per Weight function, this term adjust s the output weight so its only lights up the neuron when we wnat it to (make sures output is more 'intense')
			Matrix biasMatrix(cols,1);
			biasMatrix = biasMatrix.applyFunction([](const float& f){ return (float)rand() / RAND_MAX; }); 
			_biasMatrix.push_back(biasMatrix); //Inserting each Matrix into our TENSOR
			//... Since the weight function for a series of common layered Nodes is calculated using th sum of all the weights * activation values
			// H1 = (w11)(x1) +  (w12)(x2) + (w13)(x3) ... the Weighted Sum of H1
			// Where w11 is the activated value of Prev layer Node 1 and x1 is its activation value
			// ***NOTE OUR WEIGHTS ARE ARBRITRARY AS WHERE ARE ACTIVATION VALUES ARE SPECIFICALLY CHOSEN TO LIGHT UP A NEURON MORE/LESS DEPENDING ON OUR GOAL
		}

		_NeuronValue.resize(_topology.size()); //will have 3 containters, where a our ValueMatrices will be stored for each layer

	}

	bool FeedForward(std::vector<float> input){ 
		if(input.size() != _topology[0]){ //Our input values for our first layer of neurons should have as many elements as the number in topology[0]
					return false; 
		}

		Matrix values(input.size() , 1); //Instantiation of our first Layers Activation Values:
		// [ a01 ]
		// [ a11 ]
		// [ a21 ]
		// [ a31 ]
		//   ...
		// [ a0n ] - where n is the number of neruons in that layer

		//Feed input data into our values matrix 
		for(uint32_t i = 0; i < input.size(); i++){
			values._vals[i] = input[i]; //insert values from list of Activation Vals into our Matrix
		}

		//Feed values into next layers
		for(uint32_t i = 0; i < _weightMatrix.size(); i++){ // We want to pass values onto each layer except the LAST one which will simply hold our final output
					_NeuronValue[i] = values; //Our first layer of Neurons will simply be a matrix = values Matrix we created above
					values = values.muiltply(_weightMatrix[i]); //to calculate next layers neuron values we use this formula
					values = values.add(_biasMatrix[i]);
					// An+1 = W*An + B
					values = values.applyFunction(Sigmoid);
					//NOTE: we pass in function which takes in each element in the matrix and returns the output of the Sigmoid function we alr defined
					// o(Weight*NodePrev + Bias) = val from 0-1, activation weight
		}

		_NeuronValue[_weightMatrix.size()] = values;

		//Weights and Bias need to be adjusted by FIRST determining the ERROR in our predicted result vs the true result
		//Ex our Output was 0.5 but our targetOutput was 1 hence our error can be calculated with:
		//(1 - 0.5)^2 <-> Predicted - Target
		//And we average the Errors, we create a function
		//Our NEXT Step is to take the d/dx(error) = + , -
		//The sign of this will help us determine how much to change the weights (+ subtract weight a little, - add a little weight)
		//Positiv slop means bottom of hill is leftwards, a Negative Slope means the bottom/min is rightwards, bottom of hill
		//We are looking for local minima
		//Target => Error = 0 by modifying the weights value via
		//The amount we shift our Weights will be determined by using the learningRate
		// Algorithm : 
		// - Compute the Gradient of your Muiltivariable Function D/Dx C(x)
		// - Step in direction of Grad(C(x,y)) or in this case Add/Small amount from weights
		// - Repeat and make LearningRate/Step = const * D/Dx (C(x))
		return true;

	}


	bool PropogateBackwards(std:: vector<float> targetOutput){ ///This will be a N by 1 column Matrix with the values in the 2nd TO LAST LAYER of Neurons
		// Propogation Backwards 
		// When we determine how much each weight (strength of connection between two neurons)
		// ...should be shifted after forward feeding our inital values/weights we end at the
		// ...the last layer, adding the total changes of all the weights
		if(targetOutput.size() != _topology.back()){
			return false;
		}

		Matrix error(targetOutput.size(), 1); //Column Matrix that will hold our error values
		error._vals = targetOutput; //Our error matrix will be composed of the outputvalues of our network's last layer
		
		Matrix OutPred = _NeuronValue.back().negative(); //The final output values will be the last layer of our NeuronValue
		error = error.add(OutPred);
		//Error = (outValues - targetValues)^2
		
		for(int i = _weightMatrix.size() - 1; i >= 0; i--){ 
			Matrix transp =_weightMatrix[i].Transpose();
			Matrix prevErrors = error.muiltply(transp);  //Cost function * preLayer's Weight = prevLayer Error

			Matrix DerivativeError = _NeuronValue[i+1].applyFunction( DeriSigmoid ); // The Rate of Change of our Cost Function -> d/dx Sigmoid

			Matrix Gradient = error.muiltplyElement(DerivativeError); //Gradient is our rate of change of Error * error
			Gradient = Gradient.muiltplybyScalar(_learningRate); //adjust

			Matrix weightGradient = _NeuronValue[i].Transpose().muiltply(Gradient); //This is our gradient, THE AMOUNT WE WILL ADJUST EACH SPECIFIC WEIGHT BY!
			//dW = [.] * DerivError
			//		 [.]
			//		 [.]
			//		 
			// dW = our weight adjustment we need to apply 
			_weightMatrix[i] = _weightMatrix[i].add(weightGradient);

			_biasMatrix[i] = _biasMatrix[i].add(Gradient); //Bias
			error = prevErrors; //Now our new Error will be the Previous Layers
		}

		return true;
	}

	std:: vector<float> getPrediction(){
		return _NeuronValue.back()._vals;
	}

};

