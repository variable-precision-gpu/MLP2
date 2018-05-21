#pragma once
#ifndef TENSOR2D_H
#define TENSOR2D_H

#include <stdio.h>

#include "tensor1d.h"

class Tensor2D {
public:
    // TODO: Make me private!
    int sizeX;
    int sizeY;
    float* devData;
    
    Tensor2D(int sizeX, int sizeY, float** hostData);
    Tensor2D(int sizeX, int sizeY, float* devData);
    ~Tensor2D();

    float* getDeviceData();
    float** fetchDataFromDevice();
    
    void add(Tensor2D* tensor);
    void add(Tensor1D* tensor);
    void subtract(Tensor2D* tensor);
    void scale(float factor);
    Tensor2D* multiply(Tensor2D* tensor);
    Tensor2D* multiplyByTransposition(Tensor2D* tensor);
    Tensor2D* transposeAndMultiply(Tensor2D* tensor);
    Tensor1D* meanX();

    void debugPrint();
};

#endif  /* !TENSOR2D_H */
