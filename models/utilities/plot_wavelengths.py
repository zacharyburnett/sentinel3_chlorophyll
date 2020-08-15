import matplotlib

matplotlib.use('Qt5Agg')

from matplotlib import colors, pyplot
import numpy
import pandas

if __name__ == '__main__':
    training_data = pandas.read_csv('../../data/training.csv')
    testing_data = pandas.read_csv('../data/testing.csv')
    validation_data = pandas.read_csv('../../data/validation.csv')

    training_data = training_data.sort_values('Chl')

    wavelengths = list(training_data.columns)[:-1]
    wavelengths[0] = wavelengths[0][2:]
    wavelengths = [float(wavelength) for wavelength in wavelengths]

    color_map = pyplot.cm.rainbow
    color_normalizer = colors.Normalize(vmin=0, vmax=len(training_data))

    chlorophyll_mininmum = training_data['Chl'].min()
    chlorophyll_maxinmum = training_data['Chl'].max()

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1, projection='3d')

    for site_index, site in training_data.iloc[::500, :].iterrows():
        axis.plot(xs=numpy.full([len(wavelengths)], fill_value=chlorophyll_mininmum + ((chlorophyll_maxinmum - chlorophyll_mininmum) * color_normalizer(site_index))),
                  ys=wavelengths,
                  zs=site[:-1],
                  c=color_map(color_normalizer(site_index)))

    axis.set_xlabel('chlorophyll')
    axis.set_ylabel('wavelength (nm)')
    axis.set_zlabel('reflectance')

    pyplot.show()

    print('done')
