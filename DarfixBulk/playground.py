import numpy 
import torch
import time
import darfix
from scipy.signal import medfilt2d

numpy.random.seed(23)
values = numpy.linspace(-0.5, 1, 2)  # Example X-values
data = numpy.random.rand(2, 10, 10)  # Y-values of shape (5, 2048, 2048)
def compute_moments(values, data, smooth: bool = True):
    """
    Compute first, second, third and fourth moment of data on values.

    :param values: 1D array of X-values
    :param data: nD array of Y-values with `len(weight) == len(values)`
    :returns: The four first moments to distribution Y(X)
    """
    if len(values) != len(data):
        raise ValueError("the length of 'values' and 'data' is not equal")

    wsum = numpy.sum(data, axis=0, dtype=numpy.float64)
    values = numpy.asarray(values, dtype=numpy.float64)

    # Moments
    # mean = sum(w * x) / sum(w)
    # var  = sum(w * (x - mean)^2) / sum(w)
    # skew = sum(w * ((x - mean)/sigma)^3) / sum(w)
    # kurt = sum(w * ((x - mean)/sigma)^4) / sum(w) - 3
    #
    # The loops below are there to avoid creating another array
    # in memory with the same shape as `weights`.

    with numpy.errstate(invalid="ignore", divide="ignore"):
        mean = sum(w * x for x, w in zip(values, data))
        mean /= wsum

        var = sum(w * ((x - mean) ** 2) for x, w in zip(values, data))
        var /= wsum
        sigma = numpy.sqrt(var)
        fwhm = darfix.config.FWHM_VAL * sigma

        skew = sum(w * (((x - mean) / sigma) ** 3) for x, w in zip(values, data))
        skew /= wsum

        kurt = sum(w * (((x - mean) / sigma) ** 4) for x, w in zip(values, data))
        kurt /= wsum
        kurt -= 3  # Fisher’s definition

    if smooth:
        mean = medfilt2d(mean)
        fwhm = medfilt2d(fwhm)
        skew = medfilt2d(skew)
        kurt = medfilt2d(kurt)

    return mean, var, skew, kurt
def compute_moments_torch(values, data, smooth: bool = True):
    """
    Compute first, second, third and fourth moment of data on values.

    :param values: 1D array of X-values
    :param data: nD array of Y-values with `len(weight) == len(values)`
    :returns: The four first moments to distribution Y(X)
    """
    if len(values) != len(data):
        raise ValueError("the length of 'values' and 'data' is not equal")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    # Set chunk size based on available memory. You can tune this value.
    chunk_size = 100  # Example value, adjust based on your memory constraints

    mean = torch.zeros(data.shape[1:], dtype=torch.float64)  # Pre-allocate on CPU
    var = torch.zeros(data.shape[1:], dtype=torch.float64)
    skew = torch.zeros(data.shape[1:], dtype=torch.float64)
    kurt = torch.zeros(data.shape[1:], dtype=torch.float64)

    # Loop over chunks of the Y dimension (rows/pixels)
    for start in range(0, data.shape[1], chunk_size):
        end = min(start + chunk_size, data.shape[1])

        # Move a chunk to the GPU
        data_chunk = torch.tensor(data[:, start:end,:], dtype=torch.float64).to(device)

        # Compute the sum of weights (wsum) for this chunk
        wsum_chunk = torch.sum(data_chunk, dim=0)

        # Pre-allocate tensors for this chunk on GPU
        mean_chunk = torch.zeros(data_chunk.shape[1:], dtype=torch.float64).to(device)
        var_chunk = torch.zeros(data_chunk.shape[1:], dtype=torch.float64).to(device)
        skew_chunk = torch.zeros(data_chunk.shape[1:], dtype=torch.float64).to(device)
        kurt_chunk = torch.zeros(data_chunk.shape[1:], dtype=torch.float64).to(device)

        # Compute mean for the chunk
        for i in range(len(values)):
            mean_chunk += values[i] * data_chunk[i]
        mean_chunk /= wsum_chunk
        # Compute variance, skewness, and kurtosis for the chunk
        for i in range(len(values)):
            diff = values[i] - mean_chunk
            var_chunk += data_chunk[i] * (diff ** 2)
        var_chunk /= wsum_chunk

        for i in range(len(values)):
            diff = values[i] - mean_chunk
            skew_chunk += data_chunk[i] * ((diff / torch.sqrt(var_chunk)) ** 3)
            kurt_chunk += data_chunk[i] * ((diff / torch.sqrt(var_chunk)) ** 4)

        # Normalize by wsum

        skew_chunk /= wsum_chunk
        kurt_chunk = kurt_chunk / wsum_chunk - 3

        # Move the results back to CPU and accumulate
        mean[start:end,:] = mean_chunk.cpu()
        var[start:end,:] = var_chunk.cpu()
        skew[start:end,:] = skew_chunk.cpu()
        kurt[start:end,:] = kurt_chunk.cpu()

    mean = mean.cpu().numpy()
    var = var.cpu().numpy()
    skew = skew.cpu().numpy()
    kurt = kurt.cpu().numpy()

    # Compute FWHM
    sigma = numpy.sqrt(var)
    FWHM_VAL = darfix.config.FWHM_VAL  # Adjust as needed
    fwhm = FWHM_VAL * sigma
    if smooth:
        mean = medfilt2d(mean)
        fwhm = medfilt2d(fwhm)
        skew = medfilt2d(skew)
        kurt = medfilt2d(kurt)
    # Print results
    return mean, fwhm, skew, kurt


mean, fwhm, skew, kurt = compute_moments(values, data)
mean_torch , fwhm_torch , skew_torch , kurt_torch = compute_moments_torch(values, data)

# Count the number of elements that are not equal
mean_diff, fwhm_diff, skew_diff, kurt_diff = numpy.isclose(mean, mean_torch), numpy.isclose(fwhm, fwhm_torch), numpy.isclose(skew, skew_torch), numpy.isclose(kurt, kurt_torch)
num_differences_mean = numpy.size(mean) - numpy.sum(mean_diff)

num_differences_fwhm = numpy.size(mean) - numpy.sum(fwhm_diff)

num_differences_3 = numpy.size(mean) - numpy.sum(skew_diff)

num_differences_4 = numpy.size(mean) - numpy.sum(kurt_diff)

print(f"Mean: {num_differences_mean} differences")
print(f"FWHM: {num_differences_fwhm} differences")
print(f"Skew: {num_differences_3} differences")
print(f"Kurtosis: {num_differences_4} differences")