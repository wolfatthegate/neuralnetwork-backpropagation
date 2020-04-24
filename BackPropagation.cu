/* 
 * CPU version of BackPropagation Neural Network written in CUDA. One can change the extension .cu to .c 
 * and compile the program with C compiler. 
 * 
 * This program is a rework of Source code for Neural Networks w/ JAVA (Tutorial 09) - Backpropagation 01
 * from http://zaneacademy.com
 * 
 * Author - Waylon Luo
 * Date - April 23, 2020 
 *
 */

#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <algorithm>
#include <vector>

#define LEARNING_RATE 0.80 
#define NUMB_OF_EPOCHS 100
#define TD_X 4 // training data in x- dimension
#define TD_Y 2 // training data in y- dimension
#define TD_Z 2 // training data in z- dimension   

double rand_double();
double TRAINING_DATA[TD_X][TD_Y][TD_Z] = {{{0,0},{0}},
					          	 {{0,1},{1}},
					          	 {{1,0},{1}},
					          	 {{1,1},{0}}}; 

#include "Neuron.cu"

int main(void){
 
 	double result[] = {0, 0, 0, 0}; 
 	// declare and initialize neurons 
	struct neuron neurons[5]; 
	setNeurons(neurons);

	// forward propagation before the training
	for(int i = 0; i < TD_X; i++) {   // TD_X - Traning Data Dimension X 
		forwardProp(TRAINING_DATA[i][0], neurons);
		result[i] = neurons[4].output; // get output
	}
	printResult(result); 

	// training 100 * 100 = 10,000 trainings 
	for(int x = 0; x < 100; x++){
		for(int i = 0; i < NUMB_OF_EPOCHS; i++) {   
			if(i%100 == 0) {
				printf("[epoch %d ]\n", i);
			}
			for(int j = 0; j < TD_X; j++) {  // TD_X - Traning Data Dimension X 
				forwardProp(TRAINING_DATA[j][0], neurons);
				backpropError(TRAINING_DATA[j][1][0], neurons);		
				if(i%100 == 0) printTrainingData(neurons); 
			}
		}
	}
	printf("[done training]\n");

	// forward propagation after the training
	for(int i = 0; i < TD_X; i++) {
		forwardProp(TRAINING_DATA[i][0], neurons);
		result[i] = neurons[4].output; // get output
	}
	printResult(result); 

	return(1);
}