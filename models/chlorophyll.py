# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

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

    wavelengths = list(training_data.columns)[:-1]
    wavelengths[0] = wavelengths[0][2:]
    wavelengths = [float(wavelength) for wavelength in wavelengths]

    # input size
    D_in = 16
    # hidden size
    H = 1000
    # output size
    D_out = 1

    # load training data
    training_array = training_data.to_numpy()
    reflectance = training_array[:, :-1]
    chlorophyll = numpy.expand_dims(training_array[:, -1], axis=1)
    x = torch.tensor(reflectance, device=device, dtype=dtype)
    y = torch.tensor(chlorophyll, device=device, dtype=dtype)

    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
    )
    model.cuda()

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(10000):
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x)

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

    print('done')
