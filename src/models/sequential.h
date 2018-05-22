#pragma once
#ifndef SEQUENTIAL_MODEL_H
#define SEQUENTIAL_MODEL_H

#include <stdio.h>
#include <cmath>
#include <vector>

#include "../layers/base.h"
#include "../optimizers/base.h"
#include "../loss/base.h"
#include "../tensor/tensor2d.h"
#include "../utils.h"


class SequentialModel {
private:
    Optimizer* optimizer;
    LossFunction* lossFunction;
    std::vector<Layer*> layers;

public:
    SequentialModel(Optimizer* optimizer, LossFunction* lossFunction);

    void addLayer(Layer* layer);
    Tensor2D* forward(Tensor2D* input);
    void backward(Tensor2D* output, Tensor2D* layers);
};

#endif  /* !SEQUENTIAL_MODEL_H */
