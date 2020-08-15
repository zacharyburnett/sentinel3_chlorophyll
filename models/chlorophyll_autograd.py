# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

from datetime import datetime, timedelta
import math
import os

from matplotlib import pyplot
import numpy
import pandas
import torch

from models.utilities.utilities import get_logger

LOGGER = get_logger('chlorophyll.autograd', os.path.join(__file__, 'logs/chlorophyll_autograd.log'))

TRAINING_CSV_FILENAME = '../data/training.csv'
VALIDATION_CSV_FILENAME = '../data/validation.csv'
INPUT_TESTING_CSV_FILENAME = '../outputs/0_input/testing.csv'
OUTPUT_TESTING_CSV_FILENAME = '../outputs/1_autograd/testing.csv'

if __name__ == '__main__':
    # take advantage of cuda device if it exists
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define data type for tensors
    tensor_data_type = torch.float

    # load data from files
    training_data_frame = pandas.read_csv(TRAINING_CSV_FILENAME)
    validation_data_frame = pandas.read_csv(VALIDATION_CSV_FILENAME)
    testing_data_frame = pandas.read_csv(INPUT_TESTING_CSV_FILENAME)
    columns = training_data_frame.columns

    # covert read data to numpy arrays
    training_data = training_data_frame.to_numpy()
    validation_data = validation_data_frame.to_numpy()
    testing_data = testing_data_frame.to_numpy()

    # size of input per-site (all wavelengths)
    D_in = 16
    # size of output per-site (just chlorophyll)
    D_out = 1
    # size of hidden layer
    H = math.ceil(D_in * 2 / 3 + D_out)

    # training input / output tensors
    training_reflectance_tensor = torch.tensor(training_data[:, :-1], device=device, dtype=tensor_data_type)
    training_chlorophyll_tensor = torch.tensor(numpy.expand_dims(training_data[:, -1], axis=1), device=device, dtype=tensor_data_type)

    # validation input tensor and output data
    validation_reflectance_tensor = torch.tensor(validation_data[:, :-1], device=device, dtype=tensor_data_type)
    validation_chlorophyll_data = numpy.expand_dims(validation_data[:, -1], axis=1)

    # testing input tensor
    testing_reflectance_tensor = torch.tensor(testing_data, device=device, dtype=tensor_data_type)

    # generate random weight tensors
    weight_1 = torch.randn(D_in, H, device=device, dtype=tensor_data_type, requires_grad=True)
    weight_2 = torch.randn(H, D_out, device=device, dtype=tensor_data_type, requires_grad=True)

    start_time = datetime.now()

    # hold times and losses here for plotting later
    times = [datetime.now()]
    losses = [0]

    # iterate over specified passes at defined learning rate
    learning_rate = 1e-8
    passes = 10000

    LOGGER.info(f'starting {passes} optimization passes on autograd model ({D_in}->{H}->{D_out}) w/ {learning_rate} learning rate')

    for pass_index in range(passes):
        # calculate predicted output manually
        predicted_chlorophyll_tensor = training_reflectance_tensor.mm(weight_1).clamp(min=0).mm(weight_2)

        # calculate loss as sum of squared errors
        loss = (predicted_chlorophyll_tensor - training_chlorophyll_tensor).pow(2).sum()

        # collect data point every 100 iterations
        if pass_index % 100 == 99:
            losses.append(loss.item())
            times.append(datetime.now())
            LOGGER.info(f'{pass_index:>6} - {losses[-1]:>12} ({losses[-1] - losses[-2]:>12} difference)')

        # add gradients to weights `.grad`
        loss.backward()

        # apply gradients to weights (don't know how this works but there it is)
        with torch.no_grad():
            weight_1 -= learning_rate * weight_1.grad
            weight_2 -= learning_rate * weight_2.grad

            # clear gradients from weights so they are applied fresh next iteration
            weight_1.grad.zero_()
            weight_2.grad.zero_()

    LOGGER.info(f'optimization took {(datetime.now() - start_time) / timedelta(seconds=1):.2} seconds')

    # plot loss and loss change rate over time
    figure = pyplot.figure()
    figure.suptitle(f'autograd loss w/ {passes} passes @ {learning_rate} learning rate')
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(times, losses, label='loss')
    axis.plot(times, numpy.concatenate(([0], numpy.diff(losses))), label='loss dx')
    axis.legend()
    pyplot.show()

    # compare predicted output with validation input
    validation_chlorophyll_tensor = validation_reflectance_tensor.mm(weight_1).clamp(min=0).mm(weight_2)
    validation_predicted_chlorophyll = validation_chlorophyll_tensor.detach().cpu().numpy()
    validation_differences = validation_chlorophyll_data - validation_predicted_chlorophyll
    validation_rmse = numpy.sqrt(numpy.mean(numpy.square(validation_differences)))
    LOGGER.info(f'RMSE of validation chlorophyll vs predicted chlorophyll: {validation_rmse:.3}')
    LOGGER.info(f'standard deviation of validation chlorophyll:            {numpy.std(validation_chlorophyll_data):.3}')

    # predict output of testing dataset
    testing_chlorophyll_tensor = testing_reflectance_tensor.mm(weight_1).clamp(min=0).mm(weight_2)
    testing_predicted_chlorophyll = testing_chlorophyll_tensor.detach().cpu().numpy()
    testing_data_frame.insert(len(testing_data_frame.columns), 'Chl', testing_predicted_chlorophyll, True)
    testing_data_frame.to_csv(OUTPUT_TESTING_CSV_FILENAME)

    print('done')
