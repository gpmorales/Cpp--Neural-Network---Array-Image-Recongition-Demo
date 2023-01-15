#include <iostream>
#include "NeuralNetwork.h"
#include <vector>
#include <cstdio>

using namespace std;

int main(){
    // 2 input neurons, 3 hidden neurons and 1 output neuron 
    std::vector<uint32_t> topology = {49,10,10,3}; //The dimensions of our Tensor / Neurons in each layer of our Network

    NeuralNetwork nn(topology, 0.1); //Neural Network object instantiation
    
    //sample dataset
    std::vector<std::vector<float>> targetInputs = { //Input data set
        /* XOR problem
        {0.0f, 0.0f},
        {1.0f, 1.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f}
        */

        /* 3 Bit Binary Number predictor
        {0.0f,0.0f,0.0f},
        {0.0f,0.0f,1.0f},
        {0.0f,1.0f,0.0f},
        {0.0f,1.0f,1.0f},
        {1.0f,0.0f,0.0f},
        {1.0f,0.0f,1.0f},
        {1.0f,1.0f,0.0f},
        {1.0f,1.0f,1.0f},
        */

        // Basic Image Recongnition
        {1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f, //Cross
         0.0f,1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,
         1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f},

        {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f, //Cross
         0.0f,1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
        
        {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f, //Cross
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},

        {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f, //Circle
         0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},

        {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f, //Circle
         0.0f,0.0f,1.0f,1.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,1.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},

        {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f, //Circle
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f},

        {0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f, //Circle
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},

        {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f, //Empty
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},

        {1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f, //Empty
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},

        {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f, //Empty
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
    }; 

    std::vector<std::vector<float>> targetOutputs = { //Expected output set
        /* XOR
        {0.0f},
        {0.0f},
        {1.0f},
        {1.0f}
        */

        /* 3-Bit stuff
        {1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
        {0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
        {0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
        {0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,0.0f},
        {0.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f},
        {0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f},
        {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,0.0f},
        {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f},
        */

        
        //Circle, Cross, Empty
        {0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
    };

    uint32_t epoch = 200000;
    
    //training the neural network with randomized data
    std::cout << "training start\n";

    for(uint32_t i = 0; i < epoch; i++){
        uint32_t index = rand() % 10;
        nn.FeedForward(targetInputs[index]);
        nn.PropogateBackwards(targetOutputs[index]);
    }

    std::cout << "training complete\n";


    //TESTING the Neural Network***
    nn.FeedForward(
        {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f, 
         0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,
         0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,
         0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,0.0f,
         1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f});

    std::vector<float> preds = nn.getPrediction(); //Returns value of final layer neurons after weights have been properly adjusted in training cycle

    std::cout << "Circle: " << preds[0] << " Cross: " << preds[1] << " Empty: " << preds[2] << std::endl;

    /*
    nn.FeedForward({1,1,1});
    int i = 0;
    for( std::vector<float> input : targetInputs){
        std::vector<float> preds = nn.getPrediction();
        std::cout << i << ": " << preds[i++] << std::endl;
    }
    */

    return 0;
}
