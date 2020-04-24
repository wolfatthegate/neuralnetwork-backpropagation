#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

enum neuralNetworkLayerType { inputLayer = 0, hiddenLayer = 1, outputLayer = 2 }; 

double applyActivationFunction(double weightedSum) {
	// activation function is a sigmoid function
	return (1.0 / (1 + exp(-1.0 * weightedSum)));  
}

double derivative(double output) {
	return output * (1.0 - output); 
}

struct neuron {
	
	double threshold = 0; 
	double weight[2] = {0, 0}; 
	double output = 0; 
	double error = 0; 
	neuralNetworkLayerType layerType; 
};

void forwardProp(double input[], struct neuron *neurons) {
	double weightedSum = 0; 
	for( int i = 0; i < (int) sizeof (neurons); i++){
		switch (neurons[i].layerType) {
			case 0: // input layer
				neurons[i].output = input[i];  
				break;
			case 1: // hidden layer
				weightedSum = neurons[i].threshold + 
								  neurons[i].weight[0] * neurons[0].output + 
		    		              neurons[i].weight[1] * neurons[1].output;
				neurons[i].output = applyActivationFunction(weightedSum); 
				break; 
			case 2: // output layer
				weightedSum = neurons[i].threshold + 
	    		                  neurons[i].weight[0] * neurons[2].output + 
	    		                  neurons[i].weight[1] * neurons[3].output;
		    	neurons[i].output = applyActivationFunction(weightedSum); 
				break; 
		}
	}
}

void backpropError(double targetResult, struct neuron *neurons){
	// calculating for output neurons
	neurons[4].error = (targetResult - neurons[4].output) * derivative(neurons[4].output);
	neurons[4].threshold = neurons[4].threshold + LEARNING_RATE * neurons[4].error;
	neurons[4].weight[0] = neurons[4].weight[0] + LEARNING_RATE * neurons[4].error * neurons[2].output; 
	neurons[4].weight[1] = neurons[4].weight[1] + LEARNING_RATE * neurons[4].error * neurons[3].output; 
	
	// calculating for hidden layer 1 
	neurons[3].error = (neurons[4].weight[1] * neurons[4].error) * derivative(neurons[3].output);
	neurons[3].threshold = neurons[3].threshold + LEARNING_RATE * neurons[3].error;
	neurons[3].weight[0] = neurons[3].weight[0] + LEARNING_RATE * neurons[3].error * neurons[0].output;
	neurons[3].weight[1] = neurons[3].weight[1] + LEARNING_RATE * neurons[3].error * neurons[1].output;

	// calculating for hidden layer 2 
	neurons[2].error = (neurons[4].weight[0] * neurons[4].error) * derivative(neurons[2].output);
	neurons[2].threshold = neurons[2].threshold + LEARNING_RATE * neurons[2].error;
	neurons[2].weight[0] = neurons[2].weight[0] + LEARNING_RATE * neurons[2].error * neurons[0].output;
	neurons[2].weight[1] = neurons[2].weight[1] + LEARNING_RATE * neurons[2].error * neurons[1].output;

}

void setNeurons(struct neuron *neurons){

	srand((long)time(NULL)); /* initialize rand() */
	for (int i = 0; i < 2; i ++){
		neurons[i].threshold = 0.5 - (rand()/(double)RAND_MAX); 
		neurons[i].weight[0] =  0.5 - (rand()/(double)RAND_MAX); 
		neurons[i].weight[1] =  0.5 - (rand()/(double)RAND_MAX); 
		neurons[i].layerType = inputLayer; 
	}

	for (int i = 2; i < 4; i ++){
		neurons[i].threshold = 0.5 - (rand()/(double)RAND_MAX); 
		neurons[i].weight[0] =  0.5 - (rand()/(double)RAND_MAX); 
		neurons[i].weight[1] =  0.5 - (rand()/(double)RAND_MAX); 
		neurons[i].layerType = hiddenLayer; 
	}

	neurons[4].threshold = 0.5 - (rand()/(double)RAND_MAX); 
	neurons[4].weight[0] =  0.5 - (rand()/(double)RAND_MAX); 
	neurons[4].weight[1] =  0.5 - (rand()/(double)RAND_MAX); 
	neurons[4].layerType = outputLayer; 
}

void printTrainingData(struct neuron *neurons){
	
	printf("[(I: %.2f), (I: %.2f), ", neurons[0].output, neurons[1].output); 
	printf("(H: %.2f, %.2f, %.2f, %.5f), ", neurons[2].weight[0], neurons[2].weight[1], neurons[2].threshold, neurons[2].output);
	printf("(H: %.2f, %.2f, %.2f, %.5f), ", neurons[3].weight[0], neurons[3].weight[1], neurons[3].threshold, neurons[3].output);
	printf("(O: %.2f, %.2f, %.2f, %.5f)]\n ", neurons[4].weight[0], neurons[4].weight[1], neurons[4].threshold, neurons[4].output);
}

void printResult(double result[]) {
	printf("    Input 1    |    Input 2    | Target Result |  Result    \n");
	printf("-------------------------------------------------------------\n");
	for(int i = 0; i < 4; i++ ) {
		for(int j = 0; j < 2; j++) {
			printf("    %.5f    |", TRAINING_DATA[i][0][j]); 
		}
		printf("    %.5f    |   %.5f   \n", TRAINING_DATA[i][1][0], result[i]);
	}
}
