from pathlib import Path
from fastmcp import FastMCP

from utils import read_image, read_image_uint8
from common import PROJECT_ROOT, parse_temp_dir, make_resolvers

mcp = FastMCP()

TEMP_DIR = parse_temp_dir()
_resolve_path, _resolve_output_path = make_resolvers(TEMP_DIR)


@mcp.tool(description='''
Description:
Compute the Coefficient of Variation (CV) for a dataset. 
The CV is defined as the ratio of the standard deviation to the mean 
and is commonly used as a normalized measure of dispersion.

Parameters:
- x (list[float]): Input data values.
- ddof (int, optional): Delta Degrees of Freedom for standard deviation calculation. 
                        * ddof = 0 → population std
                        * ddof = 1 → sample std (default)

Returns:
- cv (float): Coefficient of Variation. Returns NaN if mean == 0.
''')
def coefficient_of_variation(x: list, ddof: int = 1):
    import numpy as np
    x = np.asarray(x)
    mean = np.mean(x)
    std = np.std(x, ddof=ddof)
    if mean == 0:
        return float('nan')
    return float(std / mean)


@mcp.tool(description='''
Description:
Compute the skewness of a dataset, which measures the asymmetry of the probability distribution.

Parameters:
- x (list[float]): Input data values.
- bias (bool, optional): If False, applies bias correction (Fisher-Pearson unbiased estimator).
                         Default = True.

Returns:
- skew (float): Skewness of the dataset.
''')
def skewness(x: list, bias: bool = True):
    import numpy as np
    x = np.asarray(x)
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=0 if bias else 1)
    if std == 0:
        return 0.0
    m3 = np.mean((x - mean)**3)
    skew = m3 / std**3
    if not bias and n > 2:
        skew *= np.sqrt(n * (n - 1)) / (n - 2)
    return float(skew)


@mcp.tool(description='''
Description:
Compute the kurtosis of a dataset, which measures the "tailedness" of the distribution.

Parameters:
- x (list[float]): Input data values.
- bias (bool, optional): If False, applies bias correction (unbiased estimator). Default = True.
- fisher (bool, optional): If True, returns excess kurtosis; if False, regular kurtosis. Default = True.

Returns:
- kurt (float): Kurtosis of the dataset.
''')
def kurtosis(x: list, bias: bool = True, fisher: bool = True):
    import numpy as np
    x = np.asarray(x)
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=0 if bias else 1)
    if std == 0:
        return 0.0
    m4 = np.mean((x - mean)**4)
    kurt = m4 / std**4
    if not bias and n > 3:
        numerator = (n*(n+1)*((x - mean)**4).sum())
        denominator = (n-1)*(n-2)*(n-3)*std**4
        adjustment = numerator / denominator
        excess = 3 * (n-1)**2 / ((n-2)*(n-3))
        kurt = adjustment - excess
    if fisher:
        kurt -= 3
    return float(kurt)


def calc_single_image_mean(file_path: str, uint8: bool = False) -> float:
    import numpy as np
    file_path = _resolve_path(file_path)
    img = read_image_uint8(file_path) if uint8 else read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)
    return float(np.nanmean(flat))


@mcp.tool(description='''
Compute mean value of an batch of images.

Args:
    file_list (list): List of image file paths.
    uint8 (bool): Whether to convert image to uint8 format.

Returns:
    mean (list): List of mean pixel values.
''')
def calc_batch_image_mean(file_list: list[str], uint8: bool = False) -> list[float]:
    return [float(calc_single_image_mean(file_path, uint8)) for file_path in file_list]


def calc_single_image_std(file_path: str, uint8: bool = False) -> float:
    import numpy as np
    file_path = _resolve_path(file_path)
    img = read_image_uint8(file_path) if uint8 else read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)
    return float(np.nanstd(flat, ddof=1))


@mcp.tool(description='''
Description:
Compute the standard deviation (spread of pixel values) for a batch of images.

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- std (list[float]): List of standard deviation values, one for each input image.
''')
def calc_batch_image_std(file_list: list[str], uint8: bool = False) -> list[float]:
    return [float(calc_single_image_std(file_path, uint8)) for file_path in file_list]


def calc_single_image_median(file_path: str, uint8: bool = False) -> float:
    import numpy as np
    file_path = _resolve_path(file_path)
    img = read_image_uint8(file_path) if uint8 else read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)
    return float(np.nanmedian(flat))


@mcp.tool(description='''
Description:
Compute the median pixel value for a batch of images.

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- median (list[float]): List of median pixel values, one for each input image.
''')
def calc_batch_image_median(file_list: list[str], uint8: bool = False) -> list[float]:
    return [float(calc_single_image_median(file_path, uint8)) for file_path in file_list]


def calc_single_image_min(file_path: str, uint8: bool = False) -> float:
    import numpy as np
    file_path = _resolve_path(file_path)
    img = read_image_uint8(file_path) if uint8 else read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)
    return float(np.nanmin(flat))


@mcp.tool(description='''
Description:
Compute the minimum pixel value for a batch of images.

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- min (list[float]): List of minimum pixel values, one for each input image.
''')
def calc_batch_image_min(file_list: list[str], uint8: bool = False) -> list[float]:
    return [float(calc_single_image_min(file_path, uint8)) for file_path in file_list]


def calc_single_image_max(file_path: str, uint8: bool = False) -> float:
    import numpy as np
    file_path = _resolve_path(file_path)
    img = read_image_uint8(file_path) if uint8 else read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)
    return float(np.nanmax(flat))


@mcp.tool(description='''
Description:
Compute the maximum pixel value for a batch of images.

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- max (list[float]): List of maximum pixel values, one for each input image.
''')
def calc_batch_image_max(file_list: list[str], uint8: bool = False) -> list[float]:
    return [float(calc_single_image_max(file_path, uint8)) for file_path in file_list]


def calc_single_image_skewness(file_path: str, uint8: bool = False) -> float:
    import numpy as np
    from scipy.stats import skew
    file_path = _resolve_path(file_path)
    img = read_image_uint8(file_path) if uint8 else read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)
    return float(skew(flat, bias=False))


@mcp.tool(description='''
Description:
Compute the skewness of pixel value distributions for a batch of images. 
Skewness quantifies the asymmetry of the distribution:
- Positive skew → longer right tail
- Negative skew → longer left tail
- Zero skew → symmetric distribution

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- skewness (list[float]): List of skewness values, one for each input image.
''')
def calc_batch_image_skewness(file_list: list[str], uint8: bool = False) -> list[float]:
    return [float(calc_single_image_skewness(file_path, uint8)) for file_path in file_list]


def calc_single_image_kurtosis(file_path: str, uint8: bool = False) -> float:
    import numpy as np
    from scipy.stats import kurtosis
    file_path = _resolve_path(file_path)
    img = read_image_uint8(file_path) if uint8 else read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)
    flat_clean = flat[~np.isnan(flat)]
    if len(flat_clean) == 0:
        raise ValueError("No valid data points for kurtosis calculation")
    return float(kurtosis(flat_clean, fisher=False))


@mcp.tool(description='''
Description:
Compute the kurtosis of pixel value distributions for a batch of images. 
Kurtosis measures the "tailedness" of the distribution relative to a normal distribution.

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- kurtosis (list[float]): List of kurtosis values, one for each input image.
''')
def calc_batch_image_kurtosis(file_list: list[str], uint8: bool = False) -> list[float]:
    return [float(calc_single_image_kurtosis(file_path, uint8)) for file_path in file_list]


def calc_single_image_sum(file_path: str, uint8: bool = False) -> float:
    import numpy as np
    file_path = _resolve_path(file_path)
    img = read_image_uint8(file_path) if uint8 else read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)
    return float(np.nansum(flat))


@mcp.tool(description='''
Description:
Compute the sum of pixel values for a batch of images.

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- sum (list[float]): List of pixel sum values, one for each input image.
''')
def calc_batch_image_sum(file_list: list[str], uint8: bool = False) -> list[float]:
    return [float(calc_single_image_sum(file_path, uint8)) for file_path in file_list]


def calc_single_image_hotspot_percentage(file_path: str, threshold: float, uint8: bool = False) -> float:
    import numpy as np
    file_path = _resolve_path(file_path)
    img = read_image_uint8(file_path) if uint8 else read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)
    valid_pixels = flat[~np.isnan(flat)]
    if len(valid_pixels) == 0:
        return 0.0
    hotspot_pixels = valid_pixels[valid_pixels > threshold]
    return float(len(hotspot_pixels) / len(valid_pixels))


@mcp.tool(description='''
Description:
Compute the hotspot percentage (fraction of pixels above a threshold) for a batch of images.

Parameters:
- file_list (list[str]): List of input image file paths.
- threshold (float): Threshold value for hotspot detection.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- percentage (list[float]): List of hotspot area percentages (0.0–1.0), one for each input image.
''')
def calc_batch_image_hotspot_percentage(file_list: list[str], threshold: float, uint8: bool = False) -> list[float]:
    return [float(calc_single_image_hotspot_percentage(file_path, threshold, uint8)) for file_path in file_list]


def calc_single_image_hotspot_tif(file_path: str, threshold: float, output_path: str, uint8: bool = False) -> str:
    import os
    import numpy as np
    from osgeo import gdal
    file_path = _resolve_path(file_path)
    ds = gdal.Open(file_path)
    if ds is None:
        raise RuntimeError(f"Failed to open image: {file_path}")
    img = read_image_uint8(file_path) if uint8 else read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    img = np.where(np.isinf(img), np.nan, img)
    mask = np.zeros_like(img, dtype=np.float32)
    mask[img < threshold] = 1.0
    mask[img >= threshold] = 0.0
    mask[np.isnan(img)] = np.nan
    output_path = _resolve_output_path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(str(output_path), xsize=mask.shape[1], ysize=mask.shape[0], bands=1, eType=gdal.GDT_Float32)
    out_ds.GetRasterBand(1).WriteArray(mask)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    if ds.GetGeoTransform():
        out_ds.SetGeoTransform(ds.GetGeoTransform())
    if ds.GetProjection():
        out_ds.SetProjection(ds.GetProjection())
    out_ds.FlushCache()
    out_ds = None
    ds = None
    return f'Result saved at {output_path}'


@mcp.tool(description='''
Description:
Create binary hotspot maps for a batch of images, where pixels below a specified 
threshold are set to 1 (hotspot) and others set to 0. The output is saved as 
GeoTIFF files, preserving georeference metadata from the input images.

Parameters:
- file_list (list[str]): List of input image file paths.
- threshold (float): Threshold value for hotspot detection. Pixels below this threshold are marked as hotspots.
- output_path_list (list[str]): List of output file paths for the generated GeoTIFF hotspot maps.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- list[str]: Paths to the saved GeoTIFF images containing the binary hotspot maps.
''')
def calc_batch_image_hotspot_tif(file_list: list[str], threshold: float, output_path_list: list[str], uint8: bool = False) -> list[str]:
    return [calc_single_image_hotspot_tif(file_path, threshold, output_path, uint8) for file_path, output_path in zip(file_list, output_path_list)]


@mcp.tool(description='''
Description:
Compute the absolute difference between two numbers.

Parameters:
- a (float): The first number.
- b (float): The second number.

Returns:
- diff (float): The absolute difference |a - b|.
''')
def difference(a: float, b: float):
    diff = a - b
    return float(abs(diff))


@mcp.tool(description='''
Description:
Perform division between two numbers.

Parameters:
- a (float): The divisor (denominator).
- b (float): The dividend (numerator).

Returns:
- result (float): The result of b ÷ a. Returns +inf if a = 0.
''')
def division(a: float, b: float):
    if a == 0:
        return float('inf')
    return float(b / a)


@mcp.tool(description='''
Description:
Calculate the percentage change between two numbers, useful for comparing relative growth or decline.

Parameters:
- a (float): The original value (denominator).
- b (float): The new value (numerator).

Returns:
- percent (float): The percentage change, computed as ((b - a) / a) * 100.
''')
def percentage_change(a: float, b: float):
    percent = (b - a) / a * 100
    return float(percent)


@mcp.tool(description='''
Description:
Convert temperature from Kelvin to Celsius.

Parameters:
- kelvin (float): Temperature in Kelvin.

Returns:
- celsius (float): Temperature in Celsius, computed as (Kelvin - 273.15).
''')
def kelvin_to_celsius(kelvin: float):
    return kelvin - 273.15


@mcp.tool(description="""
Description:
    Convert temperature from Celsius to Kelvin.

Parameters:
    celsius (float): Temperature in Celsius.

Returns:
    kelvin (float): Temperature in Kelvin, computed as Kelvin = Celsius + 273.15.
""")
def celsius_to_kelvin(celsius: float):
    return celsius + 273.15


@mcp.tool(description='''
Description:
Find the maximum value in a list and return both the maximum value and its index.

Parameters:
- x (list[float]): Input data list.

Returns:
- result (tuple[float, int]): (max_value, max_index)
''')
def max_value_and_index(x: list):
    import numpy as np
    x = np.asarray(x)
    max_index = np.argmax(x)
    max_value = x[max_index]
    return (float(max_value), int(max_index))


@mcp.tool(description='''
Description:
Find the minimum value in a list and return both the minimum value and its index.

Parameters:
- x (list[float]): Input data list.

Returns:
- result (tuple[float, int]): (min_value, min_index)
''')
def min_value_and_index(x: list):
    import numpy as np
    x = np.asarray(x)
    min_index = np.argmin(x)
    min_value = x[min_index]
    return (float(min_value), int(min_index))


@mcp.tool(description="""
Description:
    Multiply two numbers and return their product.

Parameters:
    a (float or int): First number.
    b (float or int): Second number.

Returns:
    result (float or int): The product of a and b.
""")
def multiply(a, b):
    return a * b


@mcp.tool(description="""
Description:
    Return the ceiling (rounded up integer) of a given number.

Parameters:
    n (float): A numeric value.

Returns:
    result (int): The smallest integer greater than or equal to n.
""")
def ceil_number(n: float):
    import math
    return math.ceil(n)


@mcp.tool(description="""
Description:
    Retrieve elements from a list using a list or tuple of indices.

Parameters:
    input_list (list): The source list from which elements will be extracted.
    indexes (list[int] or tuple[int]): Indices specifying which elements to retrieve.

Returns:
    result (list): Elements corresponding to the provided indices.
""")
def get_list_object_via_indexes(input_list, indexes):
    return [input_list[index] for index in indexes]


@mcp.tool(description="""
Description:
    Compute the arithmetic mean (average) of a dataset.

Parameters:
    x (list[float]): Input data array.

Returns:
    mean_value (float): The arithmetic mean of the input values.
""")
def mean(x: list):
    import numpy as np
    x = np.asarray(x)
    return float(np.mean(x))


@mcp.tool(description="""
Description:
    Calculate the average percentage of pixels relative to a given threshold for
    one or more images and a specified band.

Parameters:
    image_paths (str or list[str]): Path or list of image file paths.
    threshold (float, optional): Threshold value. Default = 0.75.
    mode (str, optional): 'above', 'below', 'equal', 'above_equal', 'below_equal'. Default = 'above'.
    band_index (int, optional): Band index to use (0-based). Default = 0.

Returns:
    percentage (float): Average percentage of pixels matching the threshold condition across all images.
""")
def calculate_threshold_ratio(image_paths: str | list[str], threshold: float = 0.75, mode: str = 'above', band_index: int = 0) -> float:
    import numpy as np
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    ratios = []
    for image_path in image_paths:
        image_path = _resolve_path(image_path)
        img = read_image(image_path)
        if img.ndim == 3:
            if img.shape[0] <= 5 and img.shape[0] < img.shape[-1]:
                band = img[band_index]
            else:
                band = img[..., band_index]
        else:
            band = img
        valid_pixels = ~np.isnan(band)
        total_valid_pixels = np.sum(valid_pixels)
        if total_valid_pixels == 0:
            ratios.append(0.0)
            continue
        if mode == 'above':
            matching_pixels = np.sum((band > threshold) & valid_pixels)
        elif mode == 'below':
            matching_pixels = np.sum((band < threshold) & valid_pixels)
        elif mode == 'equal':
            matching_pixels = np.sum((band == threshold) & valid_pixels)
        elif mode == 'above_equal':
            matching_pixels = np.sum((band >= threshold) & valid_pixels)
        elif mode == 'below_equal':
            matching_pixels = np.sum((band <= threshold) & valid_pixels)
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: 'above', 'below', 'equal', 'above_equal', 'below_equal'")
        percentage = (matching_pixels / total_valid_pixels) * 100
        ratios.append(float(percentage))
    return float(np.mean(ratios)) if ratios else 0.0


def calc_single_image_fire_pixels(file_path: str, fire_threshold: float = 0) -> int:
    import numpy as np
    file_path = _resolve_path(file_path)
    img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    fire_pixels = np.sum(img > fire_threshold)
    return int(fire_pixels)


@mcp.tool(description="""
Description:
    Compute the number of fire pixels (FRP > threshold) for a batch of images.

Parameters:
    file_list (list[str]): Paths to input images.
    fire_threshold (float, optional): Minimum FRP value to be considered as fire. Default = 0.

Returns:
    fire_pixels (list[int]): A list of fire pixel counts, one per input image.
""")
def calc_batch_fire_pixels(file_list: list[str], fire_threshold: float = 0) -> list[int]:
    return [calc_single_image_fire_pixels(file_path, fire_threshold) for file_path in file_list]


@mcp.tool(description="""
Description:
    Create a binary map highlighting areas where fire increase exceeds a specified threshold.

Parameters:
    change_image_path (str): Path to the fire change image.
    output_path (str): Relative path for the output raster file.
    threshold (float, optional): Threshold value in MW. Default = 20.0.

Returns:
    result (str): Path to the saved GeoTIFF fire increase map.
""")
def create_fire_increase_map(change_image_path: str, output_path: str, threshold: float = 20.0) -> str:
    import os
    import rasterio
    import numpy as np
    change_image_path = _resolve_path(change_image_path)
    change_img = read_image(change_image_path)
    fire_increase_map = (change_img >= threshold).astype(np.uint8)
    with rasterio.open(change_image_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.uint8, compress='lzw', nodata=255)
        os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
        with rasterio.open(_resolve_output_path(output_path), 'w', **profile) as dst:
            dst.write(fire_increase_map, 1)
    return f'Result saved at {_resolve_output_path(output_path)}'


@mcp.tool(description="""
Description:
    Identify fire-prone areas from a hotspot map based on a given percentile threshold.

Parameters:
    file_path (str): Path to the input hotspot map file.
    output_path (str): Relative path for the output raster file.
    threshold_percentile (float, optional): Percentile threshold. Default = 75.
    uint8 (bool, optional): Whether to use uint8 format when reading the input. Default = False.

Returns:
    result (tuple[str, float]): (path to saved GeoTIFF, threshold value used)
""")
def identify_fire_prone_areas(file_path: str, output_path: str, threshold_percentile: float = 75, uint8: bool = False) -> tuple[str, float]:
    import os
    import rasterio
    import numpy as np
    file_path = _resolve_path(file_path)
    hotspot_map = read_image_uint8(file_path) if uint8 else read_image(file_path)
    if hotspot_map.size == 0:
        raise ValueError("Input hotspot map is empty")
    valid_pixels = hotspot_map[hotspot_map > 0]
    if len(valid_pixels) == 0:
        threshold_value = 0.0
        fire_prone_areas = np.zeros_like(hotspot_map, dtype=np.uint8)
    else:
        threshold_value = np.percentile(valid_pixels, threshold_percentile)
        fire_prone_areas = (hotspot_map >= threshold_value).astype(np.uint8)
    os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
    with rasterio.open(file_path) as src:
        output_profile = src.profile.copy()
        output_profile.update(dtype=rasterio.uint8, compress='lzw', nodata=255)
        with rasterio.open(_resolve_output_path(output_path), 'w', **output_profile) as dst:
            dst.write(fire_prone_areas, 1)
    return f'Result saved at {_resolve_output_path(output_path)}', float(threshold_value)


@mcp.tool(description="""
Description:
    Calculate the N-th percentile value of pixel values in a raster image.

Parameters:
    image_path (str): Path to the input raster (.tif) file.
    percentile (int or float): Percentile to calculate (range 1–100).

Returns:
    value (int or float): The pixel value at the specified percentile.
""")
def get_percentile_value_from_image(image_path, percentile):
    import rasterio
    import numpy as np
    if not (1 <= percentile <= 100):
        raise ValueError("Percentile must be between 1 and 100.")
    image_path = _resolve_path(image_path)
    with rasterio.open(image_path) as src:
        image = src.read(1)
        dtype = image.dtype
        image = image.astype(np.float32)
    image = image[np.isfinite(image)]
    if image.size == 0:
        raise ValueError("No valid pixel values found in the image.")
    result = np.percentile(image, percentile)
    if np.issubdtype(dtype, np.integer):
        return int(round(result))
    elif np.issubdtype(dtype, np.floating):
        return float(result)
    else:
        raise TypeError(f"Unsupported raster data type: {dtype}")


@mcp.tool(description="""
Description:
    Calculate the mean of pixel-wise division between two images 
    or between two bands of the same image.

Parameters:
    image_path1 (str): Path to the first image.
    image_path2 (str, optional): Path to the second image. If None, band1/band2 of image_path1 are used.
    band1 (int, optional): Band index for numerator. Default = 1.
    band2 (int, optional): Band index for denominator. Default = 2.

Returns:
    result (float): The mean of the valid pixel-wise division results.
""")
def image_division_mean(image_path1, image_path2=None, band1=1, band2=2):
    import rasterio
    import numpy as np
    image_path1 = _resolve_path(image_path1)
    if image_path2 is not None:
        image_path2 = _resolve_path(image_path2)
    if image_path2 is None:
        with rasterio.open(image_path1) as src:
            array1 = src.read(band1).astype(np.float32)
            array2 = src.read(band2).astype(np.float32)
    else:
        with rasterio.open(image_path1) as src1, rasterio.open(image_path2) as src2:
            array1 = src1.read(1).astype(np.float32)
            array2 = src2.read(1).astype(np.float32)
    mask = (array2 != 0) & (~np.isnan(array1)) & (~np.isnan(array2))
    ratio = np.full_like(array1, np.nan, dtype=np.float32)
    ratio[mask] = array1[mask] / array2[mask]
    return float(np.nanmean(ratio))


@mcp.tool(description="""
Description:
    Calculate the percentage of pixels that simultaneously satisfy 
    threshold conditions in two raster images.

Parameters:
    path1 (str): Path to the first raster image (e.g., NDVI).
    threshold1 (float): Threshold value for the first image.
    path2 (str): Path to the second raster image (e.g., TVDI).
    threshold2 (float): Threshold value for the second image.

Returns:
    percentage (float): Percentage of pixels that satisfy both conditions.
""")
def calculate_intersection_percentage(path1, threshold1, path2, threshold2):
    import rasterio
    import numpy as np
    path1 = _resolve_path(path1)
    path2 = _resolve_path(path2)
    with rasterio.open(path1) as src1:
        data1 = src1.read(1).astype(np.float32)
        mask1 = src1.read_masks(1) > 0
    with rasterio.open(path2) as src2:
        data2 = src2.read(1).astype(np.float32)
        mask2 = src2.read_masks(1) > 0
    valid_mask = mask1 & mask2
    valid_data1 = np.where(valid_mask, data1, np.nan)
    valid_data2 = np.where(valid_mask, data2, np.nan)
    condition1 = valid_data1 > threshold1
    condition2 = valid_data2 > threshold2
    intersection_mask = condition1 & condition2
    total_valid_pixels = np.count_nonzero(~np.isnan(valid_data1) & ~np.isnan(valid_data2))
    intersection_pixels = np.count_nonzero(intersection_mask)
    if total_valid_pixels == 0:
        return 0.0
    return (intersection_pixels / total_valid_pixels) * 100


@mcp.tool(description="""
Description:
    Compute the average of mean pixel values across a batch of images.

Parameters:
    file_list (list[str]): List of image file paths.
    uint8 (bool, optional): Whether to convert images to uint8 format. Default = False.

Returns:
    mean_of_means (float): The average of the mean pixel values across all images.
""")
def calc_batch_image_mean_mean(file_list: list[str], uint8: bool = False) -> float:
    import numpy as np
    means = [float(calc_single_image_mean(file_path, uint8)) for file_path in file_list]
    return float(np.mean(means))


@mcp.tool(description="""
Description:
    Compute the mean pixel values of a batch of images and return the maximum mean.

Parameters:
    file_list (list[str]): Paths to input images.
    uint8 (bool, optional): Whether to treat image as uint8. Default = False.

Returns:
    max_mean (float): The maximum mean pixel value among all images.
""")
def calc_batch_image_mean_max(file_list: list[str], uint8: bool = False) -> float:
    means = [float(calc_single_image_mean(file_path, uint8)) for file_path in file_list]
    return max(means)


@mcp.tool(description="""
Description:
    Compute batch-wise statistics across multiple images: mean of means,
    max of maxes, min of mins.

Parameters:
    file_list (list[str]): List of image file paths.
    uint8 (bool, optional): Whether to convert the data to uint8 range. Default = False.

Returns:
    result (tuple[float, float, float]): (mean of means, max of maxs, min of mins)
""")
def calc_batch_image_mean_max_min(file_list: list[str], uint8: bool = False) -> tuple[float, float, float]:
    import rasterio
    import numpy as np
    means, maxs, mins = [], [], []
    for file_path in file_list:
        file_path = _resolve_path(file_path)
        with rasterio.open(file_path) as src:
            img = src.read(1).astype(np.float32)
            img = img[np.isfinite(img)]
            if uint8:
                img = np.clip(img, 0, 1) * 255
            means.append(np.mean(img))
            maxs.append(np.max(img))
            mins.append(np.min(img))
    return float(np.mean(means)), float(np.max(maxs)), float(np.min(mins))


@mcp.tool(description="""
Description:
    Calculate the percentage or count of images whose mean pixel values 
    (in a specified band) are above or below a given threshold.

Parameters:
    file_list (list[str]): List of image file paths.
    threshold (float): Threshold value for comparison.
    above (bool, optional): If True, count images with mean > threshold. Default = True.
    uint8 (bool, optional): If True, rescale image data to 0–255 range. Default = False.
    band_index (int, optional): Index of the band to read. Default = 0.
    return_type (str, optional): "ratio" (percentage) or "count". Default = "ratio".

Returns:
    float | int: Percentage (0–100) or count of images satisfying the condition.
""")
def calc_batch_image_mean_threshold(
    file_list: list[str],
    threshold: float,
    above: bool = True,
    uint8: bool = False,
    band_index: int = 0,
    return_type: str = "ratio"
) -> float | int:
    import rasterio
    import numpy as np
    valid_means = []
    for file_path in file_list:
        try:
            file_path = _resolve_path(file_path)
            with rasterio.open(file_path) as src:
                if band_index >= src.count:
                    print(f"Warning: {file_path} does not contain band {band_index + 1}. Skipped.")
                    continue
                data = src.read(band_index + 1).astype(np.float32)
                data = np.where(data == src.nodata, np.nan, data)
                if uint8:
                    dmin, dmax = np.nanmin(data), np.nanmax(data)
                    if dmax > dmin:
                        data = (data - dmin) / (dmax - dmin) * 255
                    else:
                        data[:] = 0
                mean_val = np.nanmean(data)
                if not np.isnan(mean_val):
                    valid_means.append(mean_val)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    if not valid_means:
        return 0 if return_type == "count" else 0.0
    valid_means = np.array(valid_means)
    count = np.sum(valid_means > threshold) if above else np.sum(valid_means < threshold)
    if return_type == "count":
        return int(count)
    elif return_type == "ratio":
        return float(count / len(valid_means) * 100.0)
    else:
        raise ValueError("return_type must be 'ratio' or 'count'")


@mcp.tool(description="""
Description:
    Calculate the percentage of pixels that simultaneously satisfy multiple band threshold conditions.

Parameters:
    image_path (str): Path to the multi-band image file.
    band_conditions (list[tuple[int, float, str]]): (band_index, threshold_value, "above"/"below")

Returns:
    float: Percentage of pixels satisfying all conditions (intersection).
""")
def calculate_multi_band_threshold_ratio(image_path: str, band_conditions: list) -> float:
    import rasterio
    import numpy as np
    image_path = _resolve_path(image_path)
    with rasterio.open(image_path) as src:
        bands = []
        for band_index, _, _ in band_conditions:
            band = src.read(band_index + 1).astype(np.float32)
            band[band == src.nodata] = np.nan
            bands.append(band)
    combined_mask = np.ones_like(bands[0], dtype=bool)
    for band, (band_index, threshold, compare_type) in zip(bands, band_conditions):
        valid = ~np.isnan(band)
        if compare_type.lower() == "above":
            mask = (band > threshold) & valid
        elif compare_type.lower() == "below":
            mask = (band < threshold) & valid
        else:
            raise ValueError(f"Invalid compare_type '{compare_type}', must be 'above' or 'below'")
        combined_mask &= mask
    total_valid_pixels = float(np.sum(~np.isnan(bands[0])))
    satisfying_pixels = float(np.sum(combined_mask))
    if total_valid_pixels == 0:
        return 0.0
    return (satisfying_pixels / total_valid_pixels) * 100


@mcp.tool(description="""
Description:
    Count the number of pixels that simultaneously satisfy multiple band threshold conditions.

Parameters:
    image_path (str): Path to the multi-band image file.
    band_conditions (list[tuple[int, float, str]]): (band_index, threshold_value, "above"/"below")

Returns:
    int: Number of pixels satisfying all threshold conditions (intersection).
""")
def count_pixels_satisfying_conditions(image_path: str, band_conditions: list) -> int:
    import rasterio
    import numpy as np
    image_path = _resolve_path(image_path)
    with rasterio.open(image_path) as src:
        bands = []
        for band_index, _, _ in band_conditions:
            band = src.read(band_index + 1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                band[band == nodata] = np.nan
            bands.append(band)
    combined_mask = np.ones_like(bands[0], dtype=bool)
    for band, (band_index, threshold, compare_type) in zip(bands, band_conditions):
        valid = ~np.isnan(band)
        if compare_type.lower() == "above":
            mask = (band > threshold) & valid
        elif compare_type.lower() == "below":
            mask = (band < threshold) & valid
        else:
            raise ValueError(f"Invalid compare_type '{compare_type}', must be 'above' or 'below'")
        combined_mask &= mask
    return int(np.sum(combined_mask))


@mcp.tool(description="""
Count how many images have a percentage of pixels above or below a threshold 
that exceeds a specified ratio.

Parameters:
    image_paths (str or list): Path(s) to image file(s).
    value_threshold (float): Pixel value threshold.
    ratio_threshold (float): Percentage threshold for comparison.
    mode (str): 'above' or 'below'. Default 'above'.
    verbose (bool): If True, prints detailed ratio results per image.

Returns:
    int: Number of images whose pixel ratio exceeds the ratio_threshold.
""")
def count_images_exceeding_threshold_ratio(
    image_paths: str | list[str],
    value_threshold: float = 0.7,
    ratio_threshold: float = 20.0,
    mode: str = 'above',
    verbose: bool = True
) -> int:
    import rasterio
    import numpy as np
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    count_exceeding = 0
    for path in image_paths:
        path = _resolve_path(path)
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                band[band == nodata] = np.nan
        valid_mask = ~np.isnan(band)
        total_pixels = np.sum(valid_mask)
        if total_pixels == 0:
            ratio = 0.0
        else:
            if mode == 'below':
                selected_mask = (band < value_threshold) & valid_mask
            else:
                selected_mask = (band > value_threshold) & valid_mask
            ratio = (np.sum(selected_mask) / total_pixels) * 100
        if ratio > ratio_threshold:
            count_exceeding += 1
            status = "match"
        else:
            status = "no match"
        if verbose:
            print(f"{path}: {ratio:.2f}% {'<' if mode == 'below' else '>'} {value_threshold} → {status} (threshold: {ratio_threshold}%)")
    if verbose:
        print(f"\nTotal images exceeding threshold ratio: {count_exceeding} / {len(image_paths)}")
    return count_exceeding


@mcp.tool(description="""
Calculate the average percentage of pixels exceeding a value threshold,
considering only images where the ratio is greater than a specified ratio threshold.

Parameters:
    image_paths (str or list): Path(s) to image file(s).
    value_threshold (float): Pixel value threshold.
    ratio_threshold (float): Minimum percentage threshold for inclusion.
    mode (str): 'above' or 'below'. Default 'above'.
    verbose (bool): If True, prints detailed ratio results per image.

Returns:
    float: Average percentage of qualifying images. Returns 0.0 if none qualify.
""")
def average_ratio_exceeding_threshold(
    image_paths: str | list[str],
    value_threshold: float = 0.7,
    ratio_threshold: float = 20.0,
    mode: str = 'above',
    verbose: bool = True
) -> float:
    import rasterio
    import numpy as np
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    ratios = []
    for path in image_paths:
        try:
            path = _resolve_path(path)
            with rasterio.open(path) as src:
                band = src.read(1).astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    band[band == nodata] = np.nan
        except Exception as e:
            if verbose:
                print(f"Error processing {path}: {e}")
            continue
        valid_mask = ~np.isnan(band)
        total_pixels = np.sum(valid_mask)
        if total_pixels == 0:
            ratio = 0.0
        else:
            if mode == 'below':
                selected_mask = (band < value_threshold) & valid_mask
            else:
                selected_mask = (band > value_threshold) & valid_mask
            ratio = (np.sum(selected_mask) / total_pixels) * 100
        if ratio > ratio_threshold:
            ratios.append(ratio)
            status = "match"
        else:
            status = "no match"
        if verbose:
            print(f"{path}: {ratio:.2f}% {'<' if mode == 'below' else '>'} {value_threshold} → {status} (threshold: {ratio_threshold}%)")
    avg = float(np.mean(ratios)) if ratios else 0.0
    if verbose:
        print(f"\nAverage ratio of qualifying images: {avg:.2f}% ({len(ratios)} out of {len(image_paths)} images)")
    return avg


@mcp.tool(description="""
Count how many images have a mean pixel value above or below
a multiple of the overall mean pixel value across all images.

Parameters:
    image_paths (str or list): Path(s) to image file(s).
    mean_multiplier (float): Multiplier applied to the overall mean.
    mode (str): 'above' or 'below'. Default 'above'.
    verbose (bool): If True, prints detailed mean and threshold comparisons per image.

Returns:
    int: Number of images satisfying the condition.
""")
def count_images_exceeding_mean_multiplier(
    image_paths: str | list[str],
    mean_multiplier: float = 1.1,
    mode: str = 'above',
    verbose: bool = True
) -> int:
    import rasterio
    import numpy as np
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    image_means = []
    for path in image_paths:
        path = _resolve_path(path)
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                band[band == nodata] = np.nan
        image_means.append(np.nanmean(band))
    overall_mean = np.nanmean(image_means)
    threshold = mean_multiplier * overall_mean
    if verbose:
        print(f"\nOverall mean across all images: {overall_mean:.4f}")
        print(f"Threshold for comparison ({mode}): {threshold:.4f}\n")
    count = 0
    for path, img_mean in zip(image_paths, image_means):
        condition = img_mean < threshold if mode == 'below' else img_mean > threshold
        op = "<" if mode == 'below' else ">"
        if condition:
            count += 1
            status = "match"
        else:
            status = "no match"
        if verbose:
            print(f"{path}: mean = {img_mean:.4f} {op} {threshold:.4f} → {status}")
    if verbose:
        print(f"\nTotal images satisfying condition: {count} / {len(image_paths)}")
    return count


@mcp.tool(description="""
Calculate the mean value of a target band over pixels where a condition band
satisfies a threshold.

Parameters:
    image_path (str): Path to the multi-band raster image.
    condition_band_index (int): Zero-based index of the band used for thresholding.
    condition_threshold (float): Threshold value to apply on the condition band.
    condition_mode (str, default='above'): 'above' (>=) or 'below' (<).
    target_band_index (int, default=0): Band for which the mean is calculated.

Returns:
    float: Mean value of the target band over selected pixels.
""")
def calculate_band_mean_by_condition(
    image_path: str,
    condition_band_index: int,
    condition_threshold: float,
    condition_mode: str = 'above',
    target_band_index: int = 0
) -> float:
    import rasterio
    import numpy as np
    image_path = _resolve_path(image_path)
    with rasterio.open(image_path) as src:
        condition_band = src.read(condition_band_index + 1).astype(np.float32)
        target_band = src.read(target_band_index + 1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            condition_band[condition_band == nodata] = np.nan
            target_band[target_band == nodata] = np.nan
    if condition_mode == 'below':
        mask = (condition_band < condition_threshold) & (~np.isnan(condition_band)) & (~np.isnan(target_band))
    else:
        mask = (condition_band >= condition_threshold) & (~np.isnan(condition_band)) & (~np.isnan(target_band))
    selected_values = target_band[mask]
    return float(np.nanmean(selected_values))


@mcp.tool(description="""
Calculate the mean value of corresponding raster pixels in path2 
where the raster values in path1 exceed the given threshold. Requires
filenames in both path1 and path2 to contain a YYYY_MM_DD_HHMM timestamp
so files can be paired; if none match, no computation happens and NaN is
returned — treat that as a data/input problem, not a real answer of "NaN".

Parameters:
    path1 (str or list[str]): Path(s) to the first set of raster files (e.g., LST).
    path2 (str or list[str]): Path(s) to the second set of raster files (e.g., TVDI).
    threshold (float): Threshold for values in path1 (e.g., LST in Kelvin).

Returns:
    float: Mean value of path2 pixels that meet the threshold condition in path1.
           Returns NaN if no matched file pairs or no valid data is found.
""")
def calc_threshold_value_mean(
    path1: str | list[str],
    path2: str | list[str],
    threshold: float = 300.0
) -> float:
    import re
    import numpy as np
    files1 = [Path(_resolve_path(path1))] if isinstance(path1, (str, Path)) else [Path(_resolve_path(p)) for p in path1]
    files2 = [Path(_resolve_path(path2))] if isinstance(path2, (str, Path)) else [Path(_resolve_path(p)) for p in path2]
    pattern = re.compile(r"\d{4}_\d{2}_\d{2}_\d{4}")
    dict1 = {pattern.search(f.name).group(): f for f in files1 if pattern.search(f.name)}
    dict2 = {pattern.search(f.name).group(): f for f in files2 if pattern.search(f.name)}
    matched_keys = set(dict1.keys()) & set(dict2.keys())
    if not matched_keys:
        print("No matched file pairs found.")
        return np.nan
    import rasterio
    all_vals = []
    for key in sorted(matched_keys):
        file1, file2 = dict1[key], dict2[key]
        with rasterio.open(file1) as ds1, rasterio.open(file2) as ds2:
            data1 = ds1.read(1).astype(np.float32)
            data2 = ds2.read(1).astype(np.float32)
            mask = (data1 > threshold) & (data1 > 0) & (data2 >= 0) & (data2 <= 1)
            if np.any(mask):
                all_vals.extend(data2[mask].flatten())
    if not all_vals:
        print("No valid values found for path1 > threshold.")
        return np.nan
    return float(np.mean(all_vals))


@mcp.tool(description="""
Calculate average of multiple tif files and save result to same directory.

Parameters:
    file_list (list[str]): List of tif file paths.
    output_path (str): Relative path for the output raster file.
    uint8 (bool): Convert to uint8 format, default False.

Returns:
    output_path (str): Full path of output file.
""")
def calculate_tif_average(file_list: list[str], output_path: str, uint8: bool = False) -> str:
    import os
    from osgeo import gdal
    import numpy as np
    file_list = [_resolve_path(f) for f in file_list]
    output_path = _resolve_output_path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    ds = gdal.Open(file_list[0])
    bands = ds.RasterCount
    rows = ds.RasterYSize
    cols = ds.RasterXSize
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    if bands == 1:
        first_img = ds.GetRasterBand(1).ReadAsArray()
    else:
        first_img = np.stack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)], axis=0)
        first_img = np.transpose(first_img, (1, 2, 0))
    ds = None
    sum_img = np.zeros_like(first_img, dtype=np.float64)
    count = len(file_list)
    sum_img = sum_img + first_img
    for file_path in file_list[1:]:
        ds = gdal.Open(file_path)
        if bands == 1:
            img = ds.GetRasterBand(1).ReadAsArray()
        else:
            img = np.stack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)], axis=0)
            img = np.transpose(img, (1, 2, 0))
        ds = None
        sum_img = sum_img + img
    avg_img = sum_img / count
    if uint8:
        if len(avg_img.shape) == 2:
            min_val, max_val = np.min(avg_img), np.max(avg_img)
            avg_img = (avg_img - min_val) / (max_val - min_val) * 255
            avg_img = avg_img.astype(np.uint8)
        else:
            for band in range(avg_img.shape[2]):
                band_data = avg_img[:, :, band]
                min_val, max_val = np.min(band_data), np.max(band_data)
                band_data = (band_data - min_val) / (max_val - min_val) * 255
                avg_img[:, :, band] = band_data.astype(np.uint8)
    driver = gdal.GetDriverByName('GTiff')
    data_type = gdal.GDT_Byte if uint8 else gdal.GDT_Float32
    if len(avg_img.shape) == 2:
        out_ds = driver.Create(str(output_path), cols, rows, 1, data_type)
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        out_ds.GetRasterBand(1).WriteArray(avg_img)
    else:
        out_ds = driver.Create(str(output_path), cols, rows, bands, data_type)
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        for i in range(bands):
            out_ds.GetRasterBand(i + 1).WriteArray(avg_img[:, :, i])
    out_ds = None
    return f'Result saved at {output_path}'


@mcp.tool(description="""
Calculate difference between two tif files (image_b - image_a) and save result.

Parameters:
    image_a_path (str): Path to first image (will be subtracted from).
    image_b_path (str): Path to second image (will subtract from).
    output_path (str): Relative path for the output raster file.
    uint8 (bool): Convert to uint8 format, default False.

Returns:
    output_path (str): Full path of output file.
""")
def calculate_tif_difference(image_a_path: str, image_b_path: str, output_path: str, uint8: bool = False) -> str:
    import os
    from osgeo import gdal
    import numpy as np
    image_a_path = _resolve_path(image_a_path)
    image_b_path = _resolve_path(image_b_path)
    output_path = _resolve_output_path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    ds_a = gdal.Open(image_a_path)
    if ds_a is None:
        raise RuntimeError(f"Failed to open {image_a_path}")
    bands_a, rows_a, cols_a = ds_a.RasterCount, ds_a.RasterYSize, ds_a.RasterXSize
    geotransform = ds_a.GetGeoTransform()
    projection = ds_a.GetProjection()
    if bands_a == 1:
        img_a = ds_a.GetRasterBand(1).ReadAsArray()
    else:
        img_a = np.stack([ds_a.GetRasterBand(i + 1).ReadAsArray() for i in range(bands_a)], axis=0)
        img_a = np.transpose(img_a, (1, 2, 0))
    ds_a = None
    ds_b = gdal.Open(image_b_path)
    if ds_b is None:
        raise RuntimeError(f"Failed to open {image_b_path}")
    bands_b, rows_b, cols_b = ds_b.RasterCount, ds_b.RasterYSize, ds_b.RasterXSize
    if bands_b == 1:
        img_b = ds_b.GetRasterBand(1).ReadAsArray()
    else:
        img_b = np.stack([ds_b.GetRasterBand(i + 1).ReadAsArray() for i in range(bands_b)], axis=0)
        img_b = np.transpose(img_b, (1, 2, 0))
    ds_b = None
    if rows_a != rows_b or cols_a != cols_b or bands_a != bands_b:
        raise ValueError(f"Images must have same dimensions. Image A: {rows_a}x{cols_a}x{bands_a}, Image B: {rows_b}x{cols_b}x{bands_b}")
    diff_img = img_b.astype(np.float64) - img_a.astype(np.float64)
    if uint8:
        if len(diff_img.shape) == 2:
            min_val, max_val = np.min(diff_img), np.max(diff_img)
            if max_val > min_val:
                diff_img = (diff_img - min_val) / (max_val - min_val) * 255
                diff_img = diff_img.astype(np.uint8)
            else:
                diff_img = np.zeros_like(diff_img, dtype=np.uint8)
        else:
            for band in range(diff_img.shape[2]):
                band_data = diff_img[:, :, band]
                min_val, max_val = np.min(band_data), np.max(band_data)
                if max_val > min_val:
                    band_data = (band_data - min_val) / (max_val - min_val) * 255
                    diff_img[:, :, band] = band_data.astype(np.uint8)
                else:
                    diff_img[:, :, band] = 0
    driver = gdal.GetDriverByName('GTiff')
    data_type = gdal.GDT_Byte if uint8 else gdal.GDT_Float32
    if len(diff_img.shape) == 2:
        out_ds = driver.Create(str(output_path), cols_a, rows_a, 1, data_type)
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        out_ds.GetRasterBand(1).WriteArray(diff_img)
    else:
        out_ds = driver.Create(str(output_path), cols_a, rows_a, bands_a, data_type)
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        for i in range(bands_a):
            out_ds.GetRasterBand(i + 1).WriteArray(diff_img[:, :, i])
    out_ds = None
    return f"Result saved at {output_path}"


@mcp.tool(description="""
Subtract two images and save result.

Parameters:
    img1_path (str): Path to first image.
    img2_path (str): Path to second image.
    output_path (str): Relative path for the output raster file.

Returns:
    str: Path to output file.
""")
def subtract(img1_path: str, img2_path: str, output_path: str) -> str:
    import os
    import numpy as np
    import rasterio
    img1_path = _resolve_path(img1_path)
    img2_path = _resolve_path(img2_path)
    img1 = read_image(img1_path)
    img2 = read_image(img2_path)
    result = img1.astype(np.float32) - img2.astype(np.float32)
    output_path = _resolve_output_path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    with rasterio.open(img1_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, compress='lzw')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(result, 1)
    return f'Result saved at {output_path}'


@mcp.tool(description="""
Description:
This function calculates the area of non-zero pixels in the input image and returns the result.

Parameters:
    input_image_path (str): Path to the input image file (TIFF, PNG, JPG, etc.).
    gsd (float): Ground sample distance in meters per pixel; if None, returns raw non-zero pixel count.
Returns:
    area (int): The area of non-zero pixels in the input image.
""")
def calculate_area(input_image_path, gsd):
    import numpy as np
    input_image_path = _resolve_path(input_image_path)
    image = read_image(input_image_path)
    if gsd is None:
        return float(np.sum(image != 0))
    return float(np.sum(image != 0) * gsd * gsd)


@mcp.tool()
def grayscale_to_colormap(image_path: str, save_name: str, cmap_name: str = 'viridis', preserve_geo: bool = False):
    """
    Apply a colormap to a grayscale image and save as a color image.

    Parameters:
        image_path (str): Path to input grayscale image (e.g. .tif).
        save_name (str): Filename for save color image (.png, .jpg, or .tif).
        cmap_name (str): Name of a matplotlib colormap, e.g. 'viridis', 'RdBu', etc.
        preserve_geo (bool): If True, preserves georeferencing.
    """
    import os
    from osgeo import gdal
    import numpy as np
    import matplotlib.cm as cm
    import cv2
    image_path = _resolve_path(image_path)
    ds = gdal.Open(image_path)
    if ds is None:
        raise RuntimeError(f"Failed to open image: {image_path}")
    gray = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    gray = np.nan_to_num(gray, nan=0.0)
    norm = (gray - np.min(gray)) / (np.max(gray) - np.min(gray) + 1e-8)
    cmap = cm.get_cmap(cmap_name)
    color_img = cmap(norm)[:, :, :3]
    color_img_uint8 = (color_img * 255).astype(np.uint8)
    save_path = _resolve_output_path(save_name)
    if preserve_geo:
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            os.path.splitext(str(save_path))[0] + '.tif',
            xsize=color_img_uint8.shape[1],
            ysize=color_img_uint8.shape[0],
            bands=3,
            eType=gdal.GDT_Byte
        )
        for i in range(3):
            out_ds.GetRasterBand(i + 1).WriteArray(color_img_uint8[:, :, i])
            gt = ds.GetGeoTransform()
            prj = ds.GetProjection()
            if gt:
                out_ds.SetGeoTransform(gt)
            if prj:
                out_ds.SetProjection(prj)
        out_ds.FlushCache()
        out_ds = None
    else:
        bgr_img = cv2.cvtColor(color_img_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), bgr_img)
    return f'Result saved at {save_path}'


@mcp.tool(description="""
Apply Landsat 8 surface reflectance (SR_B*) radiometric correction.

Parameters:
    input_band_path (str): Path to the input reflectance band file.
    output_path (str): Relative path for the output raster file.

Returns:
    str: Path to the saved corrected reflectance file.
""")
def radiometric_correction_sr(input_band_path, output_path):
    import os
    import rasterio
    import numpy as np
    input_band_path = _resolve_path(input_band_path)
    with rasterio.open(input_band_path) as band_src:
        band_array = band_src.read(1)
        band_profile = band_src.profile
    band_array = np.array(band_array, dtype=np.float32)
    corrected_band = band_array * 0.0000275 + (-0.2)
    corrected_profile = band_profile.copy()
    corrected_profile.update(dtype=rasterio.float32, nodata=np.nan)
    os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
    with rasterio.open(_resolve_output_path(output_path), 'w', **corrected_profile) as dst:
        dst.write(corrected_band.astype(rasterio.float32), 1)
    return f'Result saved at {_resolve_output_path(output_path)}'


@mcp.tool(description="""
Apply cloud/shadow mask to a single Landsat 8 surface reflectance band using QA_PIXEL band.

Parameters:
    sr_band_path (str): Path to surface reflectance band (e.g., SR_B3 or SR_B5).
    qa_pixel_path (str): Path to QA_PIXEL band.
    output_path (str): Relative path for the output raster file.

Returns:
    str: Path to the saved masked raster file.
""")
def apply_cloud_mask(sr_band_path, qa_pixel_path, output_path):
    import os
    import rasterio
    import numpy as np
    sr_band_path = _resolve_path(sr_band_path)
    qa_pixel_path = _resolve_path(qa_pixel_path)
    cloud_mask_bits = 1 + 2 + 4 + 8 + 16
    os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
    with rasterio.open(sr_band_path) as band_src:
        band = band_src.read(1).astype(np.float32)
        profile = band_src.profile
    with rasterio.open(qa_pixel_path) as qa_src:
        qa = qa_src.read(1)
    mask = (qa & cloud_mask_bits) == 0
    band[~mask] = np.nan
    output_profile = profile.copy()
    output_profile.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(_resolve_output_path(output_path), 'w', **output_profile) as dst:
        dst.write(band.astype(rasterio.float32), 1)
    return f'Result saved at {_resolve_output_path(output_path)}'


if __name__ == "__main__":
    mcp.run()