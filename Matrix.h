//This is our header file will contain all our matrix operations and class
#ifndef MATRIX_H
#define MATRIX_H
#endif
#include <iostream>
#include <cmath>
#include <assert.h>
#include <functional>
#include <vector>
using namespace std;
//#pragma once

class Matrix{ //Will allow us to perform linaer alg computations
	//private field members
	public:
	uint32_t _cols; //unit32_t = UNSIGNED 32 BIT INT type
	uint32_t _rows;
	std:: vector<float> _vals; //declaration of float vector which will represent our 2D Matrix

public:
	Matrix():_cols(0),_rows(0),_vals({}){};

	Matrix(uint32_t cols, uint32_t rows):_cols(cols),_rows(rows),_vals({}){  //Initializer list that is part our Constructor so we can define the values of _rows & _cols
		_vals.resize(cols * rows, 0.0f); //Reisize or vector/matrix to contain all elements (rows times cols)
	}

	float& AtElement(uint32_t col, uint32_t row){
		return _vals[row * _cols + col]; //row = row#-1, _cols = total num of cols, col = col#-1
		//Our matrix is 2D :
		// [11][12][13]
		// [21][22][23]
		// [31][32][33]
		//
		// When we create our vector this 2D matrix is compacted into a 1D matrix/vector ->
		//     row 0				row 2				row 3
		// [11][12][13] [21][22][23] [31][32][33]
		//
		// Lets say we want the el in the 2nd row, 3rd col , which lands at index 5
		// We can get i = 5 by doing row * # of cols + col = row 1 * 3 cols + col 2 = 5
		// We can get i = 8, the last el by doing row 2 * 3 cols + col 2 = 8
	}

	Matrix applyFunction(std::function<float(const float&)> func){ 
//NOTE:Functions in C++ are similar to Python, we dont need a name just an input type and name (anonymous functions)
//We then can input wtv into our function and can quickly define it elsewhere
//SYTNAX -> function<returnType(Parameters)> functionName
//DEFINTION -> method([](Param A...){ list of cmds; return Type; })

		Matrix newMatrix(_cols, _rows); //Instantiation of new Matrix obj | has dimensions as original matrix obj we instantiate in constructor

		for(uint32_t y = 0; y < newMatrix._rows; y++){ //parse thru each elem in our 2D array/Matrix
			for(uint32_t x = 0; x < newMatrix._cols; x++){
				newMatrix.AtElement(x,y) = func(AtElement(x,y)); //This will allow us to apply any generic function to each el in a matrix
				//...such as Adding a scalar or muiltplying by scalar
			}
		}

		return newMatrix; //return new Matrix obj
	}


	Matrix muiltply(Matrix& target){ //Function returns a Matrix object after we muiltply it
		// To muiltply to a A by B matrix1 with a C by D matrix2
		// B (num of cols of matrix1) == C (num of rows in matrix2) 
		// And the product will have dimensionsa A by D OR (num of rows in M1 by num of cols in M2)
		assert(_cols == target._rows); //This is an ASSERTION statement used to test an assumption (in this case that the product is defined for our given input Matrix)
		Matrix newMatrix(target._cols, _rows); //instantiation of new Matrix object with proper parameters/dimensions

		for(uint32_t y = 0; y < newMatrix._rows; y++){  //These two for loops parse thru each element in our Matrix Product
			for(uint32_t x = 0; x < newMatrix._cols; x++){ 
					float result = 0.0f;
					for(uint32_t k = 0; k < _cols; k++){ //(el in row 0, col k) * (el in row k, col 0) + (el in row 0, col k+1) *
						result += AtElement(k,y) * target.AtElement(x,k);
					}
				newMatrix.AtElement(x,y) = result;
			}
		}
		return newMatrix; //return our resulting Matrix product
	}


	Matrix muiltplybyScalar(float K){
		Matrix newMatrix(_cols, _rows); //Instantiation of new Matrix obj = oldMatrix * Scalar

		for(uint32_t y = 0; y < newMatrix._rows; y++){ //parse thru each elem in our 2D array/Matrix
			for(uint32_t x = 0; x < newMatrix._cols; x++){
				newMatrix.AtElement(x,y) = AtElement(x,y) * K;  //muitlply each el by our scalar K
			}
		}

		return newMatrix; //return new Matrix obj
	}


	Matrix negative(){ return muiltplybyScalar(-1); } //makes matrix negative


	Matrix muiltplyElement(Matrix& target){ //Dot Product
		Matrix newMatrix(_cols, _rows); //Instantiation of new Matrix obj = oldMatrix * Scalar

		for(uint32_t y = 0; y < newMatrix._rows; y++){ //parse thru each elem in our 2D array/Matrix
			for(uint32_t x = 0; x < newMatrix._cols; x++){
				newMatrix.AtElement(x,y) = AtElement(x,y) * target.AtElement(x,y);  //muitlply each el by our scalar K
			}
		}

		return newMatrix; //return new Matrix obj
	}


	Matrix squareElem(){ //squares each element

		Matrix newMatrix(_cols, _rows); //Instantiation of new Matrix obj = oldMatrix * Scalar
		for(uint32_t y = 0; y < newMatrix._rows; y++){ //parse thru each elem in our 2D array/Matrix
			for(uint32_t x = 0; x < newMatrix._cols; x++){
				newMatrix.AtElement(x,y) = AtElement(x,y) * AtElement(x,y);  //muitlply each el by our scalar K
			}
		}

		return newMatrix; //return new Matrix obj
	}


	Matrix add(Matrix& target){//Function returns additive product of two matrices
		assert(_rows == target._rows && _cols == target._cols); // dimensions of our input matrix and are alr exisiting Matrix must be ==
		Matrix newMatrix(_cols, _rows); //instantiate new Matrix obj
		for(uint32_t y = 0; y < newMatrix._rows; y++){
			for(uint32_t x = 0; x < newMatrix._cols; x++){
				newMatrix.AtElement(x, y) = AtElement(x,y) + target.AtElement(x,y);
			}
		}

		return newMatrix;
	}


	Matrix addbyScalar(float k){
		Matrix newMatrix(_cols, _rows); //instantiation of new Object
		for(uint32_t y = 0; y < newMatrix._rows; y++){
			for(uint32_t x = 0; x < newMatrix._cols; x++){
				newMatrix.AtElement(x,y) = AtElement(x,y) + k; //add scalar to existing element
			}
		}
		return newMatrix;
	}

	Matrix Transpose(){
		//assert(_rows == target._rows && _cols == target._cols); // dimensions of our input matrix and are alr exisiting Matrix must be ==
		Matrix newMatrix(_rows,_cols);
		for(uint32_t y = 0; y < _rows; y++){
			for(uint32_t x = 0; x < _cols; x++){
				newMatrix.AtElement(y,x) = AtElement(x,y);
			}
		}

		return newMatrix;
	}

	static void PrintMatrix(Matrix& matrix){ //Prints our matrix vector
		unsigned count = 0;
		for(auto beg = matrix._vals.begin();  beg != matrix._vals.end(); beg++){
			if(count % matrix._cols == 0){
				cout << endl;
			}
			count++;
			cout << *beg << " ";
		}
	}

};

