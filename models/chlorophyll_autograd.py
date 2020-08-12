# -*- coding: utf-8 -*-
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-tensors-and-autograd
import numpy
import pandas
import torch

if __name__ == '__main__':
    assert torch.cuda.is_available()

    dtype = torch.float
    device = torch.device('cuda')

    training_data = pandas.read_csv('../data/training.csv')
    testing_data = pandas.read_csv('../data/testing.csv')
    validation_data = pandas.read_csv('../data/validation.csv')

    # input size
    D_in = 16
    # hidden size
    H = 100
    # output size
    D_out = 1

    # load training data
    training_array = training_data.to_numpy()
    reflectance = training_array[:, :-1]
    chlorophyll = numpy.expand_dims(training_array[:, -1], axis=1)

    x = torch.tensor(reflectance, device=device, dtype=dtype)
    y = torch.tensor(chlorophyll, device=device, dtype=dtype)

    # Create random Tensors for weights.
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Tensors during the backward pass.
    w1 = numpy.array(())
    w2 = numpy.array(())
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    for t in range(500):
        # Forward pass: compute predicted y using operations on Tensors; these
        # are exactly the same operations we used to compute the forward pass using
        # Tensors, but we do not need to keep references to intermediate values since
        # we are not implementing the backward pass by hand.
        y_pred = x.mm(w1).clamp(min=0).mm(w2)

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss.item())

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call w1.grad and w2.grad will be Tensors holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        # An alternative way is to operate on weight.data and weight.grad.data.
        # Recall that tensor.data gives a tensor that shares the storage with
        # tensor, but doesn't track history.
        # You can also use torch.optim.SGD to achieve this.
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            # Manually zero the gradients after updating weights
            w1.grad.zero_()
            w2.grad.zero_()
