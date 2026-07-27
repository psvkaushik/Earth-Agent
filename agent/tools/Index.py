from pathlib import Path
from fastmcp import FastMCP

from utils import read_image, read_image_uint8
from common import PROJECT_ROOT, parse_temp_dir, make_resolvers

mcp = FastMCP()

TEMP_DIR = parse_temp_dir()
_resolve_path, _resolve_output_path = make_resolvers(TEMP_DIR)


def calculate_ndvi(input_nir_path, input_red_path, output_path):
    """
    Calculate NDVI = (NIR - Red) / (NIR + Red) and save the result.

    Parameters:
        input_nir_path (str): Path to the NIR band raster file.
        input_red_path (str): Path to the Red band raster file.
        output_path (str): relative path for the output raster, e.g. "question17/ndvi_2022-01-16.tif"

    Returns:
        str: Path to the saved NDVI file.
    """
    import os
    import rasterio
    import numpy as np

    input_nir_path = _resolve_path(input_nir_path)
    input_red_path = _resolve_path(input_red_path)
    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)
        nir_profile = nir_src.profile
    with rasterio.open(input_red_path) as red_src:
        red_band = red_src.read(1)

    nir_band = np.array(nir_band, dtype=np.float32)
    red_band = np.array(red_band, dtype=np.float32)
    denominator = nir_band + red_band + 1e-6
    ndvi = (nir_band - red_band) / denominator

    ndvi_profile = nir_profile.copy()
    ndvi_profile.update(dtype=rasterio.float32, nodata=-9999, compress='lzw')

    os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
    with rasterio.open(_resolve_output_path(output_path), 'w', **ndvi_profile) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)

    return f'Result saved at {_resolve_output_path(output_path)}'


@mcp.tool(description="""
Batch-calculate NDVI = (NIR - Red) / (NIR + Red) from multiple pairs of NIR/Red raster files.

Parameters:
    input_nir_paths (list[str]): Paths to NIR band raster files.
    input_red_paths (list[str]): Paths to Red band raster files.
    output_paths (list[str]): Relative output paths, one per pair (e.g. "question17/ndvi_2022-01-16.tif").

Returns:
    list[str]: Result messages (saved paths), one per pair.
""")
def calculate_batch_ndvi(input_nir_paths: list[str], input_red_paths: list[str], output_paths: list[str]) -> list[str]:
    return [calculate_ndvi(nir, red, out) for nir, red, out in zip(input_nir_paths, input_red_paths, output_paths)]


def calculate_ndwi(input_nir_path, input_swir_path, output_path):
    """
    Calculate NDWI = (NIR - SWIR) / (NIR + SWIR) and save the result.

    Parameters:
        input_nir_path (str): Path to the NIR band raster file.
        input_swir_path (str): Path to the SWIR band raster file.
        output_path (str): relative path for the output raster, e.g. "question17/ndwi_2022-01-16.tif"

    Returns:
        str: Path to the saved NDWI file.
    """
    import os
    import rasterio
    import numpy as np

    input_nir_path = _resolve_path(input_nir_path)
    input_swir_path = _resolve_path(input_swir_path)
    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)
        nir_profile = nir_src.profile
    with rasterio.open(input_swir_path) as swir_src:
        swir_band = swir_src.read(1)

    nir_band = np.array(nir_band, dtype=np.float32)
    swir_band = np.array(swir_band, dtype=np.float32)
    denominator = nir_band + swir_band + 1e-6
    ndwi = (nir_band - swir_band) / denominator

    ndwi_profile = nir_profile.copy()
    ndwi_profile.update(dtype=rasterio.float32, nodata=-9999, compress='lzw')

    os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
    with rasterio.open(_resolve_output_path(output_path), 'w', **ndwi_profile) as dst:
        dst.write(ndwi.astype(rasterio.float32), 1)

    return f'Result saved at {_resolve_output_path(output_path)}'


@mcp.tool(description="""
Batch-calculate NDWI = (NIR - SWIR) / (NIR + SWIR) from multiple pairs of NIR/SWIR raster files.

Parameters:
    input_nir_paths (list[str]): Paths to NIR band raster files.
    input_swir_paths (list[str]): Paths to SWIR band raster files.
    output_paths (list[str]): Relative output paths, one per pair (e.g. "question17/ndwi_2022-01-16.tif").

Returns:
    list[str]: Result messages (saved paths), one per pair.
""")
def calculate_batch_ndwi(input_nir_paths: list[str], input_swir_paths: list[str], output_paths: list[str]) -> list[str]:
    return [calculate_ndwi(nir, swir, out) for nir, swir, out in zip(input_nir_paths, input_swir_paths, output_paths)]


def calculate_ndbi(input_swir_path, input_nir_path, output_path):
    """
    Calculate NDBI = (SWIR - NIR) / (SWIR + NIR) and save the result.

    Parameters:
        input_swir_path (str): Path to the SWIR band raster file.
        input_nir_path (str): Path to the NIR band raster file.
        output_path (str): relative path for the output raster, e.g. "question17/ndbi_2022-01-16.tif"

    Returns:
        str: Path to the saved NDBI file.
    """
    import os
    import rasterio
    import numpy as np

    input_swir_path = _resolve_path(input_swir_path)
    input_nir_path = _resolve_path(input_nir_path)
    with rasterio.open(input_swir_path) as swir_src:
        swir_band = swir_src.read(1)
        swir_profile = swir_src.profile
    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)

    swir_band = np.array(swir_band, dtype=np.float32)
    nir_band = np.array(nir_band, dtype=np.float32)
    denominator = swir_band + nir_band + 1e-6
    ndbi = (swir_band - nir_band) / denominator

    ndbi_profile = swir_profile.copy()
    ndbi_profile.update(dtype=rasterio.float32, nodata=-9999, compress='lzw')

    os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
    with rasterio.open(_resolve_output_path(output_path), 'w', **ndbi_profile) as dst:
        dst.write(ndbi.astype(rasterio.float32), 1)

    return f'Result saved at {_resolve_output_path(output_path)}'


@mcp.tool(description="""
Batch-calculate NDBI = (SWIR - NIR) / (SWIR + NIR) from multiple pairs of SWIR/NIR raster files.

Parameters:
    input_swir_paths (list[str]): Paths to SWIR band raster files.
    input_nir_paths (list[str]): Paths to NIR band raster files.
    output_paths (list[str]): Relative output paths, one per pair (e.g. "question17/ndbi_2022-01-16.tif").

Returns:
    list[str]: Result messages (saved paths), one per pair.
""")
def calculate_batch_ndbi(input_swir_paths: list[str], input_nir_paths: list[str], output_paths: list[str]) -> list[str]:
    return [calculate_ndbi(swir, nir, out) for swir, nir, out in zip(input_swir_paths, input_nir_paths, output_paths)]


def calculate_evi(input_nir_path, input_red_path, input_blue_path, output_path, G: float = 2.5, C1: float = 6, C2: float = 7.5, L: float = 1):
    """
    Calculate EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L) and save the result.

    Parameters:
        input_nir_path (str): Path to the NIR band raster file.
        input_red_path (str): Path to the Red band raster file.
        input_blue_path (str): Path to the Blue band raster file.
        output_path (str): relative path for the output raster, e.g. "question17/evi_2022-01-16.tif"
        G, C1, C2, L (float, optional): EVI coefficients (defaults: 2.5, 6, 7.5, 1).

    Returns:
        str: Path to the saved EVI file.
    """
    import os
    import rasterio
    import numpy as np

    input_nir_path = _resolve_path(input_nir_path)
    input_red_path = _resolve_path(input_red_path)
    input_blue_path = _resolve_path(input_blue_path)
    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)
        nir_profile = nir_src.profile
    with rasterio.open(input_red_path) as red_src:
        red_band = red_src.read(1)
    with rasterio.open(input_blue_path) as blue_src:
        blue_band = blue_src.read(1)

    nir_band = np.array(nir_band, dtype=np.float32)
    red_band = np.array(red_band, dtype=np.float32)
    blue_band = np.array(blue_band, dtype=np.float32)
    denominator = nir_band + C1 * red_band - C2 * blue_band + L + 1e-6
    evi = G * (nir_band - red_band) / denominator

    evi_profile = nir_profile.copy()
    evi_profile.update(dtype=rasterio.float32, nodata=-9999, compress='lzw')

    os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
    with rasterio.open(_resolve_output_path(output_path), 'w', **evi_profile) as dst:
        dst.write(evi.astype(rasterio.float32), 1)

    return f'Result saved at {_resolve_output_path(output_path)}'


@mcp.tool(description="""
Batch-calculate EVI from multiple sets of NIR/Red/Blue raster files.

Parameters:
    input_nir_paths (list[str]): Paths to NIR band raster files.
    input_red_paths (list[str]): Paths to Red band raster files.
    input_blue_paths (list[str]): Paths to Blue band raster files.
    output_paths (list[str]): Relative output paths, one per set (e.g. "question17/evi_2022-01-16.tif").
    G, C1, C2, L (float, optional): EVI coefficients (defaults: 2.5, 6, 7.5, 1).

Returns:
    list[str]: Result messages (saved paths), one per set.
""")
def calculate_batch_evi(
    input_nir_paths: list[str], input_red_paths: list[str], input_blue_paths: list[str],
    output_paths: list[str], G: float = 2.5, C1: float = 6, C2: float = 7.5, L: float = 1
) -> list[str]:
    return [
        calculate_evi(nir, red, blue, out, G=G, C1=C1, C2=C2, L=L)
        for nir, red, blue, out in zip(input_nir_paths, input_red_paths, input_blue_paths, output_paths)
    ]


def calculate_nbr(input_nir_path, input_swir_path, output_path):
    """
    Calculate NBR = (NIR - SWIR) / (NIR + SWIR) and save the result.

    Parameters:
        input_nir_path (str): Path to the NIR band raster file.
        input_swir_path (str): Path to the SWIR band raster file.
        output_path (str): relative path for the output raster, e.g. "question17/nbr_2022-01-16.tif"

    Returns:
        str: Path to the saved NBR file.
    """
    import os
    import rasterio
    import numpy as np

    input_nir_path = _resolve_path(input_nir_path)
    input_swir_path = _resolve_path(input_swir_path)
    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)
        nir_profile = nir_src.profile
    with rasterio.open(input_swir_path) as swir_src:
        swir_band = swir_src.read(1)

    nir_band = np.array(nir_band, dtype=np.float32)
    swir_band = np.array(swir_band, dtype=np.float32)
    denominator = nir_band + swir_band + 1e-6
    nbr = (nir_band - swir_band) / denominator

    nbr_profile = nir_profile.copy()
    nbr_profile.update(dtype=rasterio.float32, nodata=-9999, compress='lzw')

    os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
    with rasterio.open(_resolve_output_path(output_path), 'w', **nbr_profile) as dst:
        dst.write(nbr.astype(rasterio.float32), 1)

    return f'Result saved at {_resolve_output_path(output_path)}'


@mcp.tool(description="""
Batch-calculate NBR = (NIR - SWIR) / (NIR + SWIR) from multiple pairs of NIR/SWIR raster files.
Useful for burn-severity analysis (typically dNBR = pre-fire NBR - post-fire NBR).

Parameters:
    input_nir_paths (list[str]): Paths to NIR band raster files.
    input_swir_paths (list[str]): Paths to SWIR band raster files.
    output_paths (list[str]): Relative output paths, one per pair (e.g. "question17/nbr_2022-01-16.tif").

Returns:
    list[str]: Result messages (saved paths), one per pair.
""")
def calculate_batch_nbr(input_nir_paths: list[str], input_swir_paths: list[str], output_paths: list[str]) -> list[str]:
    return [calculate_nbr(nir, swir, out) for nir, swir, out in zip(input_nir_paths, input_swir_paths, output_paths)]


def calculate_fvc(input_nir_path, input_red_path, output_path, ndvi_min=0.1, ndvi_max=0.9):
    """
    Calculate Fractional Vegetation Cover: FVC = clip(((NDVI - ndvi_min) / (ndvi_max - ndvi_min)) * 100, 0, 100).

    Parameters:
        input_nir_path (str): Path to the NIR band raster file.
        input_red_path (str): Path to the Red band raster file.
        output_path (str): relative path for the output raster, e.g. "question17/fvc_2022-01-16.tif"
        ndvi_min (float): NDVI value treated as 0% cover (default 0.1).
        ndvi_max (float): NDVI value treated as 100% cover (default 0.9).

    Returns:
        str: Path to the saved FVC file.
    """
    import os
    import rasterio
    import numpy as np

    input_nir_path = _resolve_path(input_nir_path)
    input_red_path = _resolve_path(input_red_path)
    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)
        nir_profile = nir_src.profile
    with rasterio.open(input_red_path) as red_src:
        red_band = red_src.read(1)

    nir_band = np.array(nir_band, dtype=np.float32)
    red_band = np.array(red_band, dtype=np.float32)
    denominator = nir_band + red_band + 1e-6
    ndvi = (nir_band - red_band) / denominator

    fvc = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min)) * 100
    fvc = np.clip(fvc, 0, 100)

    fvc_profile = nir_profile.copy()
    fvc_profile.update(dtype=rasterio.float32, nodata=-9999, compress='lzw')

    os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
    with rasterio.open(_resolve_output_path(output_path), 'w', **fvc_profile) as dst:
        dst.write(fvc.astype(rasterio.float32), 1)

    return f'Result saved at {_resolve_output_path(output_path)}'


@mcp.tool(description="""
Batch-calculate Fractional Vegetation Cover (FVC, 0-100%) from multiple pairs of NIR/Red raster files.

Parameters:
    input_nir_paths (list[str]): Paths to NIR band raster files.
    input_red_paths (list[str]): Paths to Red band raster files.
    output_paths (list[str]): Relative output paths, one per pair (e.g. "question17/fvc_2022-01-16.tif").
    ndvi_min (float, optional): NDVI treated as 0% cover. Default 0.1.
    ndvi_max (float, optional): NDVI treated as 100% cover. Default 0.9.

Returns:
    list[str]: Result messages (saved paths), one per pair.
""")
def calculate_batch_fvc(
    input_nir_paths: list[str], input_red_paths: list[str], output_paths: list[str],
    ndvi_min: float = 0.1, ndvi_max: float = 0.9
) -> list[str]:
    return [
        calculate_fvc(nir, red, out, ndvi_min=ndvi_min, ndvi_max=ndvi_max)
        for nir, red, out in zip(input_nir_paths, input_red_paths, output_paths)
    ]


def calculate_wri(input_green_path, input_red_path, input_nir_path, input_swir_path, output_path):
    """
    Calculate WRI = (Green + Red) / (NIR + SWIR) and save the result.

    Parameters:
        input_green_path (str): Path to the Green band raster file.
        input_red_path (str): Path to the Red band raster file.
        input_nir_path (str): Path to the NIR band raster file.
        input_swir_path (str): Path to the SWIR band raster file.
        output_path (str): relative path for the output raster, e.g. "question17/wri_2022-01-16.tif"

    Returns:
        str: Path to the saved WRI file.
    """
    import os
    import rasterio
    import numpy as np

    input_green_path = _resolve_path(input_green_path)
    input_red_path = _resolve_path(input_red_path)
    input_nir_path = _resolve_path(input_nir_path)
    input_swir_path = _resolve_path(input_swir_path)
    with rasterio.open(input_green_path) as green_src:
        green_band = green_src.read(1)
        green_profile = green_src.profile
    with rasterio.open(input_red_path) as red_src:
        red_band = red_src.read(1)
    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)
    with rasterio.open(input_swir_path) as swir_src:
        swir_band = swir_src.read(1)

    green_band = np.array(green_band, dtype=np.float32)
    red_band = np.array(red_band, dtype=np.float32)
    nir_band = np.array(nir_band, dtype=np.float32)
    swir_band = np.array(swir_band, dtype=np.float32)
    denominator = nir_band + swir_band + 1e-6
    wri = (green_band + red_band) / denominator

    wri_profile = green_profile.copy()
    wri_profile.update(dtype=rasterio.float32, nodata=-9999, compress='lzw')

    os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
    with rasterio.open(_resolve_output_path(output_path), 'w', **wri_profile) as dst:
        dst.write(wri.astype(rasterio.float32), 1)

    return f'Result saved at {_resolve_output_path(output_path)}'


@mcp.tool(description="""
Batch-calculate WRI = (Green + Red) / (NIR + SWIR) from multiple sets of Green/Red/NIR/SWIR raster files.
Values > 1 indicate open water.

Parameters:
    input_green_paths (list[str]): Paths to Green band raster files.
    input_red_paths (list[str]): Paths to Red band raster files.
    input_nir_paths (list[str]): Paths to NIR band raster files.
    input_swir_paths (list[str]): Paths to SWIR band raster files.
    output_paths (list[str]): Relative output paths, one per set (e.g. "question17/wri_2022-01-16.tif").

Returns:
    list[str]: Result messages (saved paths), one per set.
""")
def calculate_batch_wri(
    input_green_paths: list[str], input_red_paths: list[str],
    input_nir_paths: list[str], input_swir_paths: list[str], output_paths: list[str]
) -> list[str]:
    return [
        calculate_wri(green, red, nir, swir, out)
        for green, red, nir, swir, out in zip(input_green_paths, input_red_paths, input_nir_paths, input_swir_paths, output_paths)
    ]


def calculate_ndti(input_red_path, input_green_path, output_path):
    """
    Calculate NDTI (turbidity) = (Red - Green) / (Red + Green) and save the result.

    Parameters:
        input_red_path (str): Path to the Red band raster file.
        input_green_path (str): Path to the Green band raster file.
        output_path (str): relative path for the output raster, e.g. "question17/ndti_2022-01-16.tif"

    Returns:
        str: Path to the saved NDTI file.
    """
    import os
    import rasterio
    import numpy as np

    input_red_path = _resolve_path(input_red_path)
    input_green_path = _resolve_path(input_green_path)
    with rasterio.open(input_red_path) as red_src:
        red_band = red_src.read(1)
        red_profile = red_src.profile
    with rasterio.open(input_green_path) as green_src:
        green_band = green_src.read(1)

    red_band = np.array(red_band, dtype=np.float32)
    green_band = np.array(green_band, dtype=np.float32)
    denominator = red_band + green_band + 1e-6
    ndti = (red_band - green_band) / denominator

    ndti_profile = red_profile.copy()
    ndti_profile.update(dtype=rasterio.float32, nodata=-9999, compress='lzw')

    os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
    with rasterio.open(_resolve_output_path(output_path), 'w', **ndti_profile) as dst:
        dst.write(ndti.astype(rasterio.float32), 1)

    return f'Result saved at {_resolve_output_path(output_path)}'


@mcp.tool(description="""
Batch-calculate NDTI (Normalized Difference Turbidity Index) = (Red - Green) / (Red + Green)
from multiple pairs of Red/Green raster files.

Parameters:
    input_red_paths (list[str]): Paths to Red band raster files.
    input_green_paths (list[str]): Paths to Green band raster files.
    output_paths (list[str]): Relative output paths, one per pair (e.g. "question17/ndti_2022-01-16.tif").

Returns:
    list[str]: Result messages (saved paths), one per pair.
""")
def calculate_batch_ndti(input_red_paths: list[str], input_green_paths: list[str], output_paths: list[str]) -> list[str]:
    return [calculate_ndti(red, green, out) for red, green, out in zip(input_red_paths, input_green_paths, output_paths)]


def calculate_frp(input_frp_path, output_path, fire_threshold=0):
    """
    Build a binary fire mask (0/255) from an FRP raster: pixels with FRP > fire_threshold are marked as fire.

    Parameters:
        input_frp_path (str): Path to the FRP raster file.
        output_path (str): relative path for the output raster, e.g. "question17/frp_2022-01-16.tif"
        fire_threshold (float): Minimum FRP value to be considered as fire (default 0).

    Returns:
        str: Path to the saved fire mask file.
    """
    import os
    import rasterio
    import numpy as np

    input_frp_path = _resolve_path(input_frp_path)
    with rasterio.open(input_frp_path) as frp_src:
        frp_band = frp_src.read(1)
        frp_profile = frp_src.profile

    frp_band = np.array(frp_band, dtype=np.float32)
    fire_mask = (frp_band > fire_threshold).astype(np.uint8)
    output_data = fire_mask * 255

    fire_profile = frp_profile.copy()
    fire_profile.update(dtype=rasterio.uint8, nodata=0)

    os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
    with rasterio.open(_resolve_output_path(output_path), 'w', **fire_profile) as dst:
        dst.write(output_data, 1)

    return f'Result saved at {_resolve_output_path(output_path)}'


@mcp.tool(description="""
Batch-build binary fire masks (0/255, FRP > fire_threshold) from multiple FRP raster files.
Note: this returns a mask raster path, not a pixel count — use
calc_batch_fire_pixels (statistics tools) if a count is what's needed instead.

Parameters:
    input_frp_paths (list[str]): Paths to FRP raster files.
    output_paths (list[str]): Relative output paths, one per file (e.g. "question17/frp_2022-01-16.tif").
    fire_threshold (float, optional): Minimum FRP value to be considered as fire. Default 0.

Returns:
    list[str]: Result messages (saved paths), one per file.
""")
def calculate_batch_frp(input_frp_paths: list[str], output_paths: list[str], fire_threshold: float = 0) -> list[str]:
    return [calculate_frp(frp, out, fire_threshold=fire_threshold) for frp, out in zip(input_frp_paths, output_paths)]


def calculate_ndsi(input_green_path: str, input_swir_path: str, output_path: str) -> str:
    """
    Calculate NDSI (snow) = (Green - SWIR) / (Green + SWIR) for MODIS surface reflectance
    (scaled by 0.0001), with mismatched-resolution resampling and out-of-range masking.

    Parameters:
        input_green_path (str): Path to the Green band raster file.
        input_swir_path (str): Path to the SWIR band raster file.
        output_path (str): relative path for the output raster, e.g. "question17/ndsi_2022-01-16.tif"

    Returns:
        str: Path to the saved NDSI file.
    """
    import os
    import rasterio
    import numpy as np
    from scipy.ndimage import zoom

    input_green_path = _resolve_path(input_green_path)
    input_swir_path = _resolve_path(input_swir_path)
    with rasterio.open(input_green_path) as green_src:
        green_band = green_src.read(1)
        green_profile = green_src.profile
    with rasterio.open(input_swir_path) as swir_src:
        swir_band = swir_src.read(1)
        swir_profile = swir_src.profile

    green_band = np.array(green_band, dtype=np.float32)
    swir_band = np.array(swir_band, dtype=np.float32)

    scale_factor = 0.0001
    green_band = green_band * scale_factor
    swir_band = swir_band * scale_factor

    if green_band.shape != swir_band.shape:
        target_height = min(green_band.shape[0], swir_band.shape[0])
        target_width = min(green_band.shape[1], swir_band.shape[1])
        target_shape = (target_height, target_width)
        if green_band.shape != target_shape:
            zoom_factors = (target_height / green_band.shape[0], target_width / green_band.shape[1])
            green_band = zoom(green_band, zoom_factors, order=1)
        if swir_band.shape != target_shape:
            zoom_factors = (target_height / swir_band.shape[0], target_width / swir_band.shape[1])
            swir_band = zoom(swir_band, zoom_factors, order=1)

    green_band = np.where((green_band < 0) | (green_band > 1), np.nan, green_band)
    swir_band = np.where((swir_band < 0) | (swir_band > 1), np.nan, swir_band)

    denominator = green_band + swir_band + 1e-6
    denominator = np.where((np.isnan(green_band)) | (np.isnan(swir_band)), np.nan, denominator)

    ndsi = (green_band - swir_band) / denominator
    ndsi = np.clip(ndsi, -1, 1)

    output_profile = swir_profile.copy() if swir_band.shape[0] <= green_band.shape[0] else green_profile.copy()
    output_profile.update(
        dtype=rasterio.float32, nodata=-9999, compress='lzw',
        height=ndsi.shape[0], width=ndsi.shape[1]
    )

    os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
    with rasterio.open(_resolve_output_path(output_path), 'w', **output_profile) as dst:
        dst.write(ndsi.astype(rasterio.float32), 1)

    return f'Result saved at {_resolve_output_path(output_path)}'


@mcp.tool(description="""
Batch-calculate NDSI (snow index) for multiple pairs of Green/SWIR band images. Handles
MODIS scaling (x0.0001) and mismatched Green/SWIR resolutions automatically.

Parameters:
    green_file_list (list[str]): Paths to Green band raster files.
    swir_file_list (list[str]): Paths to SWIR band raster files (same length as green_file_list).
    output_path_list (list[str]): Relative output paths, one per pair (e.g. "question17/ndsi_2022-01-16.tif").

Returns:
    list[str]: Result messages (saved paths), one per pair.
""")
def calculate_batch_ndsi(green_file_list: list[str], swir_file_list: list[str], output_path_list: list[str]) -> list[str]:
    if len(green_file_list) != len(swir_file_list):
        raise ValueError("Number of Green and SWIR files must be equal")
    return [
        calculate_ndsi(green, swir, out)
        for green, swir, out in zip(green_file_list, swir_file_list, output_path_list)
    ]


@mcp.tool(description="""
Calculate the fraction of pixels in a binary snow/ice-loss map that are marked as extreme loss.

Parameters:
    binary_map_path (str): Path to a binary raster where pixel value 1.0 marks
        extreme snow/ice loss.

Returns:
    float: Fraction of extreme-loss pixels among all valid pixels (0.0-1.0).

Example:
    >>> calc_extreme_snow_loss_percentage_from_binary_map("snow_loss_binary.tif")
    0.27
""")
def calc_extreme_snow_loss_percentage_from_binary_map(binary_map_path: str) -> float:
    import numpy as np

    binary_map_path = _resolve_path(binary_map_path)
    img = read_image(binary_map_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)
    valid_pixels = flat[~np.isnan(flat)]
    if len(valid_pixels) == 0:
        return 0.0
    extreme_loss_pixels = valid_pixels[valid_pixels == 1.0]
    return float(len(extreme_loss_pixels) / len(valid_pixels))


@mcp.tool(description='''
Compute TVDI (Temperature Vegetation Dryness Index) from NDVI and LST rasters via a
trapezoidal dry/wet-edge regression (bins NDVI into 100 bins, fits max/min LST per bin,
normalizes each pixel's LST between the fitted dry and wet edges). Needs at least 100
valid pixels and 2+ populated NDVI bins to fit; otherwise returns an all-NaN raster
(check the tool's printed warning, not just the returned path, in that case).

Parameters:
    ndvi_path (str): Path to NDVI GeoTIFF (e.g., MODIS NDVI scaled by 0.0001).
    lst_path (str): Path to LST GeoTIFF (e.g., MODIS LST scaled by 0.02).
    output_path (str): relative path for the output raster, e.g. "question17/tvdi_2022-01-16.tif"

Returns:
    str: Path to the saved TVDI GeoTIFF.
''')
def compute_tvdi(ndvi_path: str, lst_path: str, output_path: str) -> str:
    import os
    import rasterio
    import numpy as np
    from scipy.stats import linregress

    ndvi_path = _resolve_path(ndvi_path)
    lst_path = _resolve_path(lst_path)
    with rasterio.open(ndvi_path) as src_ndvi:
        ndvi = src_ndvi.read(1).astype(np.float32) * 0.0001
        profile = src_ndvi.profile
    with rasterio.open(lst_path) as src_lst:
        lst = src_lst.read(1).astype(np.float32) * 0.02

    valid_mask = (ndvi >= 0) & (ndvi <= 1) & (lst > 0)

    def _save_nan_result():
        tvdi_nan = np.full_like(ndvi, np.nan, dtype=np.float32)
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
        with rasterio.open(_resolve_output_path(output_path), 'w', **profile) as dst:
            dst.write(tvdi_nan, 1)
        return f'Result saved at {_resolve_output_path(output_path)}'

    if not np.any(valid_mask):
        print(f"Warning: No valid data points in {output_path}")
        return _save_nan_result()

    ndvi_valid = ndvi[valid_mask]
    lst_valid = lst[valid_mask]

    if len(ndvi_valid) < 100:
        print(f"Warning: Too few valid data points ({len(ndvi_valid)}) in {output_path}")
        return _save_nan_result()

    n_bins = 100
    bins = np.linspace(ndvi_valid.min(), ndvi_valid.max(), n_bins + 1)
    ndvi_bin_centers, lst_max_vals, lst_min_vals = [], [], []
    for i in range(n_bins):
        bin_mask = (ndvi_valid >= bins[i]) & (ndvi_valid < bins[i + 1])
        if np.any(bin_mask):
            ndvi_bin_centers.append((bins[i] + bins[i + 1]) / 2)
            lst_max_vals.append(np.max(lst_valid[bin_mask]))
            lst_min_vals.append(np.min(lst_valid[bin_mask]))

    if len(ndvi_bin_centers) < 2:
        print(f"Warning: Not enough data bins for regression in {output_path}")
        return _save_nan_result()

    ndvi_bin_centers = np.array(ndvi_bin_centers)
    lst_max_vals = np.array(lst_max_vals)
    lst_min_vals = np.array(lst_min_vals)

    slope_max, intercept_max, _, _, _ = linregress(ndvi_bin_centers, lst_max_vals)
    slope_min, intercept_min, _, _, _ = linregress(ndvi_bin_centers, lst_min_vals)

    lst_max = ndvi * slope_max + intercept_max
    lst_min = ndvi * slope_min + intercept_min

    denominator = lst_max - lst_min
    denominator[denominator == 0] = 1e-6
    tvdi = (lst - lst_min) / denominator
    tvdi = np.clip(tvdi, 0, 1).astype(np.float32)
    tvdi[~valid_mask] = np.nan

    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    os.makedirs((_resolve_output_path(output_path)).parent, exist_ok=True)
    with rasterio.open(_resolve_output_path(output_path), 'w', **profile) as dst:
        dst.write(tvdi, 1)

    return f'Result saved at {_resolve_output_path(output_path)}'


if __name__ == "__main__":
    mcp.run()