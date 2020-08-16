"""
Optimize PyTorch neural net model to sample set, plot loss over time, log RMSE, and write final predictions.

This script uses a fixed size for the hidden layer, defined as 2/3 the input layer plus output layer.
"""

from datetime import datetime, timedelta
import math
import os

from matplotlib import colors, pyplot
import numpy
import pandas
import torch

from models.utilities.utilities import get_logger

LOGGER = get_logger('chlorophyll.nn', os.path.join(__file__, os.pardir, '../outputs/3_nn_optimH/3_nn_optimH.log'))

TRAINING_CSV_FILENAME = '../data/training.csv'
VALIDATION_CSV_FILENAME = '../data/validation.csv'
INPUT_TESTING_CSV_FILENAME = '../outputs/0_input/testing.csv'
OUTPUT_TESTING_CSV_FILENAME = '../outputs/3_nn_optimH/testing.csv'

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
    input_layer_size = 16
    # size of output per-site (just chlorophyll)
    output_layer_size = 1
    # size of hidden layer
    default_hidden_layer_size = math.ceil(input_layer_size * 2 / 3 + output_layer_size)

    # training input / output tensors
    training_reflectance_tensor = torch.tensor(training_data[:, :-1], device=device, dtype=tensor_data_type)
    training_chlorophyll_tensor = torch.tensor(numpy.expand_dims(training_data[:, -1], axis=1), device=device, dtype=tensor_data_type)

    # validation input tensor and output data
    validation_reflectance_tensor = torch.tensor(validation_data[:, :-1], device=device, dtype=tensor_data_type)
    validation_chlorophyll_data = numpy.expand_dims(validation_data[:, -1], axis=1)

    # testing input tensor
    testing_reflectance_tensor = torch.tensor(testing_data, device=device, dtype=tensor_data_type)

    # define loss function as sum of mean square error
    loss_function = torch.nn.MSELoss(reduction='sum')

    start_time = datetime.now()

    # hold times and RMSEs here for plotting later
    hidden_layer_sizes = [0]
    rmses = [0]
    lowest_rmse = numpy.Inf
    lowest_rmse_hidden_layer_size = None
    lowest_rmse_model = None

    # optimize model over a specified number of iterations
    passes = 10000

    # iterate over a range of hidden layer sizes
    for hidden_layer_size in range(default_hidden_layer_size, input_layer_size * 5):

        # build three-layer model, from input size to hidden layer to output size
        model = torch.nn.Sequential(
            torch.nn.Linear(input_layer_size, hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_size, output_layer_size)
        )
        # ensure model is on CUDA device if available
        if torch.cuda.is_available():
            model.cuda()

        # use Adam algorithm with a liberal learning rate to start
        learning_rate = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        LOGGER.info(f'starting {passes} optimization passes on NN model ({input_layer_size}->{hidden_layer_size}->{output_layer_size}) w/ {learning_rate} learning rate')

        for pass_index in range(passes):
            # predict output using current state of model, and calculate loss
            predicted_chlorophyll_tensor = model(training_reflectance_tensor)
            mse_loss = loss_function(predicted_chlorophyll_tensor, training_chlorophyll_tensor)

            # recalculate gradients of current state
            optimizer.zero_grad()
            mse_loss.backward()

            # iterate to the next step based on current learning rate
            optimizer.step()

        # compare predicted output with validation input
        validation_chlorophyll_tensor = model(validation_reflectance_tensor)
        validation_predicted_chlorophyll = validation_chlorophyll_tensor.detach().cpu().numpy()
        validation_differences = validation_chlorophyll_data - validation_predicted_chlorophyll
        validation_rmse = numpy.sqrt(numpy.mean(numpy.square(validation_differences)))

        # collect data point every 100 iterations
        hidden_layer_size.append(hidden_layer_size)
        rmses.append(validation_rmse)
        LOGGER.info(f'optimization took {(datetime.now() - start_time) / timedelta(seconds=1):.4} seconds')
        LOGGER.info(f'RMSE: {validation_rmse:.3}; hidden layer size: {hidden_layer_size}')

        if validation_rmse < lowest_rmse:
            lowest_rmse = validation_rmse
            lowest_rmse_hidden_layer_size = hidden_layer_size
            lowest_rmse_model = model

        LOGGER.info(f'lowest RMSE: {lowest_rmse:.3}; hidden layer size of lowest RMSE: {lowest_rmse_hidden_layer_size}')

    LOGGER.info(f'standard deviation of validation chlorophyll:            {numpy.std(validation_chlorophyll_data)}')
    LOGGER.info(f'model with lowest RMSE ({lowest_rmse}) had a hidden layer size of {lowest_rmse_hidden_layer_size}')

    # predict output of testing dataset
    testing_chlorophyll_tensor = lowest_rmse_model(testing_reflectance_tensor)
    testing_predicted_chlorophyll = testing_chlorophyll_tensor.detach().cpu().numpy()
    testing_data_frame.insert(len(testing_data_frame.columns), 'Chl', testing_predicted_chlorophyll, True)
    testing_data_frame.to_csv(OUTPUT_TESTING_CSV_FILENAME)

    # remove first dummy entries from times and losses
    hidden_layer_sizes = hidden_layer_sizes[1:]
    rmses = rmses[1:]

    color_map = pyplot.cm.viridis
    color_normalizer = colors.Normalize(vmin=0, vmax=len(rmses))

    # plot loss and loss change rate over time
    figure = pyplot.figure()
    figure.suptitle(f'NN RMSE vs size of (single) hidden layer w/ {10000} passes')
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(hidden_layer_sizes, rmses, color=color_map(color_normalizer(rmses)))
    axis.set_ylabel('RMSE')
    axis.set_xlabel('hidden layer size')
    pyplot.show()
    pyplot.savefig('../notebooks/images/3_nn_optimH.png')

    print('done')
