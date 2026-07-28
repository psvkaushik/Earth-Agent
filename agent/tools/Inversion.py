from pathlib import Path
from fastmcp import FastMCP

from utils import read_image, read_image_uint8
from common import PROJECT_ROOT, parse_temp_dir, make_resolvers

mcp = FastMCP()

TEMP_DIR = parse_temp_dir()
_resolve_path, _resolve_output_path = make_resolvers(TEMP_DIR)


@mcp.tool(description='''
Compute Precipitable Water Vapor (PWV) image from local MODIS surface reflectance band files
using the band ratio method.

This method uses surface reflectance bands:
- sur_refl_b02 (0.865 μm), sur_refl_b05 (1.240 μm): atmospheric window bands
- sur_refl_b17, sur_refl_b18, sur_refl_b19: water vapor absorption bands

Parameters:
    sur_refl_b02_path (str): File path to band sur_refl_b02 (0.865 um) GeoTIFF.
    sur_refl_b05_path (str): File path to band sur_refl_b05 (1.240 um) GeoTIFF.
    sur_refl_b17_path (str): File path to band sur_refl_b17 GeoTIFF.
    sur_refl_b18_path (str): File path to band sur_refl_b18 GeoTIFF.
    sur_refl_b19_path (str): File path to band sur_refl_b19 GeoTIFF.
    output_path (str): relative path for the output raster file, e.g. "question17/pwv_2022-01-16.tif"

Returns:
    str: Path to the saved PWV GeoTIFF (4 bands: PWV, T17, T18, T19).
''')
def band_ratio(
    sur_refl_b02_path: str,
    sur_refl_b05_path: str,
    sur_refl_b17_path: str,
    sur_refl_b18_path: str,
    sur_refl_b19_path: str,
    output_path: str
) -> str:
    import os
    import rasterio
    import numpy as np

    sur_refl_b02_path = _resolve_path(sur_refl_b02_path)
    sur_refl_b05_path = _resolve_path(sur_refl_b05_path)
    sur_refl_b17_path = _resolve_path(sur_refl_b17_path)
    sur_refl_b18_path = _resolve_path(sur_refl_b18_path)
    sur_refl_b19_path = _resolve_path(sur_refl_b19_path)

    with rasterio.open(sur_refl_b02_path) as src02, \
         rasterio.open(sur_refl_b05_path) as src05, \
         rasterio.open(sur_refl_b17_path) as src17, \
         rasterio.open(sur_refl_b18_path) as src18, \
         rasterio.open(sur_refl_b19_path) as src19:

        b02 = src02.read(1).astype(np.float32)
        b05 = src05.read(1).astype(np.float32)
        b17 = src17.read(1).astype(np.float32)
        b18 = src18.read(1).astype(np.float32)
        b19 = src19.read(1).astype(np.float32)

        profile = src02.profile

    λ2, λ5 = 0.865, 1.240
    λ17, λ18, λ19 = 0.905, 0.936, 0.940

    a = (b05 - b02) / (λ5 - λ2)
    b = b02 - a * λ2
    rho17 = a * λ17 + b
    rho18 = a * λ18 + b
    rho19 = a * λ19 + b

    T17 = np.divide(b17, rho17, out=np.zeros_like(b17), where=rho17 != 0)
    T18 = np.divide(b18, rho18, out=np.zeros_like(b18), where=rho18 != 0)
    T19 = np.divide(b19, rho19, out=np.zeros_like(b19), where=rho19 != 0)

    k = 0.03
    with np.errstate(divide='ignore', invalid='ignore'):
        PWV = -np.log(T18) / k
        PWV[np.isnan(PWV)] = 0
        PWV[PWV < 0] = 0

    out_data = np.stack([PWV, T17, T18, T19], axis=0).astype(np.float32)
    profile.update(dtype=rasterio.float32, count=4, compress='lzw')

    resolved_output = _resolve_output_path(output_path)
    os.makedirs(resolved_output.parent, exist_ok=True)
    with rasterio.open(resolved_output, 'w', **profile) as dst:
        dst.write(out_data)

    return f'Result saved at {resolved_output}'


@mcp.tool(description='''
Estimate Land Surface Temperature (LST) using the Single-Channel method, with NDVI-based
emissivity estimation from RED and NIR bands.

Parameters:
    bt_path (str): Brightness Temperature GeoTIFF (Kelvin).
    red_path (str): Red band GeoTIFF (e.g., Landsat 8 Band 4).
    nir_path (str): NIR band GeoTIFF (e.g., Landsat 8 Band 5).
    output_path (str): relative path for the output raster file, e.g. "question17/lst_2022-01-16.tif"

Returns:
    str: Path to saved LST GeoTIFF.
''')
def lst_single_channel(
    bt_path: str,
    red_path: str,
    nir_path: str,
    output_path: str
) -> str:
    import os
    import rasterio
    import numpy as np

    def read_band(path):
        path = _resolve_path(path)
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            profile = src.profile
            band[band < 0] = np.nan
        return band, profile

    bt, profile = read_band(bt_path)
    red, _ = read_band(red_path)
    nir, _ = read_band(nir_path)

    ndvi = (nir - red) / (nir + red + 1e-6)

    emissivity = np.where(
        ndvi > 0.7, 0.99,
        np.where(
            ndvi < 0.2, 0.96,
            0.97 + 0.003 * ndvi
        )
    )

    wavelength = 10.9
    c2 = 1.43877e4

    lst = bt / (1 + (wavelength * bt / c2) * np.log(emissivity))

    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    resolved_output = _resolve_output_path(output_path)
    os.makedirs(resolved_output.parent, exist_ok=True)
    with rasterio.open(resolved_output, 'w', **profile) as dst:
        dst.write(lst.astype(np.float32), 1)

    return f'Result saved at {resolved_output}'


@mcp.tool(description='''
Estimate Land Surface Temperature (LST) using the multi-channel algorithm.

Parameters:
    band31_path (str): Path to local GeoTIFF file for thermal band 31 (~11 μm).
    band32_path (str): Path to local GeoTIFF file for thermal band 32 (~12 μm).
    output_path (str): Relative path for the output raster file, e.g. "question17/lst_2022-01-16.tif"

Returns:
    str: Local file path of the exported LST image.
''')
def lst_multi_channel(
    band31_path: str,
    band32_path: str,
    output_path: str
) -> str:
    import os
    import rasterio
    import numpy as np

    band31_path = _resolve_path(band31_path)
    band32_path = _resolve_path(band32_path)

    with rasterio.open(band31_path) as src31:
        band31 = src31.read(1).astype(np.float32)
        profile = src31.profile

    with rasterio.open(band32_path) as src32:
        band32 = src32.read(1).astype(np.float32)

    a = 1.022
    b = 0.47
    c = 0.43

    lst = a * band31 + b * (band31 - band32) + c

    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    resolved_output = _resolve_output_path(output_path)
    os.makedirs(resolved_output.parent, exist_ok=True)
    with rasterio.open(resolved_output, 'w', **profile) as dst:
        dst.write(lst.astype(np.float32), 1)

    return f'Result saved at {resolved_output}'


@mcp.tool(description='''
Estimate Land Surface Temperature (LST) or Precipitable Water Vapor (PWV) using the
split-window algorithm.

Parameters:
    band31_path (str): Path to thermal band 31 GeoTIFF.
    band32_path (str): Path to thermal band 32 GeoTIFF.
    emissivity31_path (str): Path to emissivity band 31 GeoTIFF.
    emissivity32_path (str): Path to emissivity band 32 GeoTIFF.
    parameter (str): "LST" or "PWV" to specify output.
    output_path (str): Relative path for the output raster file, e.g. "question17/lst_2022-01-16.tif"

Returns:
    str: Path to exported output GeoTIFF.
''')
def split_window(
    band31_path: str,
    band32_path: str,
    emissivity31_path: str,
    emissivity32_path: str,
    parameter: str,
    output_path: str
) -> str:
    import os
    import rasterio
    import numpy as np

    band31_path = _resolve_path(band31_path)
    band32_path = _resolve_path(band32_path)
    emissivity31_path = _resolve_path(emissivity31_path)
    emissivity32_path = _resolve_path(emissivity32_path)

    with rasterio.open(band31_path) as src31:
        band31 = src31.read(1).astype(np.float32)
        profile = src31.profile

    with rasterio.open(band32_path) as src32:
        band32 = src32.read(1).astype(np.float32)

    with rasterio.open(emissivity31_path) as src_e31:
        e31 = src_e31.read(1).astype(np.float32)
        e31 = e31 * 0.002 + 0.49

    with rasterio.open(emissivity32_path) as src_e32:
        e32 = src_e32.read(1).astype(np.float32)
        e32 = e32 * 0.002 + 0.49

    delta_T = band31 - band32
    eps_mean = (e31 + e32) / 2
    delta_eps = e31 - e32
    eps_mean = np.clip(eps_mean, 0.8, 1.0)

    if parameter.upper() == "LST":
        C0, C1, C2, C3, C4 = 0.268, 1.378, 0.183, 54.3, -2.238

        t31_c = band31 - 273.15

        term1 = C0
        term2 = C1 * t31_c
        term3 = C2 * (t31_c ** 2) / 1000
        term4 = (C3 + C4 * delta_T) * (1 - eps_mean)
        term5 = (C3 + C4 * delta_T) * delta_eps

        lst = term1 + term2 + term3 + term4 + term5
        lst = lst + 273.15

        lst = np.where((lst < 200) | (lst > 350), np.nan, lst)

        output = lst.astype(np.float32)

    elif parameter.upper() == "PWV":
        pwv = delta_T / (band31 * eps_mean) * 100
        output = pwv.astype(np.float32)

    else:
        raise ValueError("Parameter must be either 'LST' or 'PWV'")

    profile.update(dtype=rasterio.float32, count=1, compress="lzw")
    resolved_output = _resolve_output_path(output_path)
    os.makedirs(resolved_output.parent, exist_ok=True)

    with rasterio.open(resolved_output, "w", **profile) as dst:
        dst.write(output, 1)

    return f"Result saved at {resolved_output}"


@mcp.tool(description='''
Estimate Land Surface Temperature (LST) using an enhanced Temperature Emissivity Separation
(TES) algorithm with empirical emissivity estimation.

Parameters:
    tir_band_paths (list[str]): List of Thermal Infrared (TIR) GeoTIFF file paths (e.g., ASTER Bands 10–14).
    representative_band_index (int): Index of the TIR band to use as reference brightness temperature.
    output_path (str): relative path for the output raster file, e.g. "question17/lst_2022-01-16.tif"

Returns:
    str: Path to the output GeoTIFF containing three bands: LST, emissivity, and emissivity variation.
''')
def temperature_emissivity_separation(
    tir_band_paths: list[str],
    representative_band_index: int,
    output_path: str
) -> str:
    import os
    import rasterio
    import numpy as np

    tir_band_paths = [_resolve_path(p) for p in tir_band_paths]

    c2 = 1.43877e4
    wavelength = 10.6

    with rasterio.open(tir_band_paths[representative_band_index]) as src:
        rep_band = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        valid_mask = (rep_band > 0) & (rep_band < 1000)

    bands_data = []
    for path in tir_band_paths:
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            band[~valid_mask] = np.nan
            bands_data.append(band)

    bands_stack = np.stack(bands_data, axis=0)

    masked_stack = np.ma.masked_invalid(bands_stack)
    band_max = np.ma.max(masked_stack, axis=0).filled(np.nan)
    band_min = np.ma.min(masked_stack, axis=0).filled(np.nan)
    delta_epsilon = band_max - band_min

    emissivity = 0.982 - 0.072 * delta_epsilon
    emissivity = np.clip(emissivity, 0.85, 0.999)

    Tb = bands_stack[representative_band_index]
    Tb[~valid_mask] = np.nan
    valid_calc = (Tb > 0) & (emissivity > 0) & (~np.isnan(Tb)) & (~np.isnan(emissivity))
    lst = np.full_like(Tb, np.nan)
    lst[valid_calc] = Tb[valid_calc] / (1 + (wavelength * Tb[valid_calc] / c2) * np.log(emissivity[valid_calc]))

    out_stack = np.stack([lst, emissivity, delta_epsilon], axis=0).astype(np.float32)
    profile.update(dtype=rasterio.float32, count=3, compress='lzw', nodata=np.nan)

    resolved_output = _resolve_output_path(output_path)
    os.makedirs(resolved_output.parent, exist_ok=True)

    with rasterio.open(resolved_output, 'w', **profile) as dst:
        dst.write(out_stack)
        dst.set_band_description(1, "LST (K)")
        dst.set_band_description(2, "Emissivity (ε)")
        dst.set_band_description(3, "Emissivity Variation (Δε)")

    return f'Result saved at {resolved_output}'


@mcp.tool(description='''
Estimate land surface temperature (LST) from local MODIS Day and Night brightness temperatures
using a single-channel correction method.

Parameters:
    BT_day_path (str): Path to local Brightness Temperature Day GeoTIFF.
    BT_night_path (str): Path to local Brightness Temperature Night GeoTIFF.
    Emis_day_path (str): Path to local Emissivity Day GeoTIFF.
    Emis_night_path (str): Path to local Emissivity Night GeoTIFF.
    output_path (str): relative path for the output raster file, e.g. "question17/lst_2022-01-16.tif"

Returns:
    str: Path to the exported GeoTIFF containing six bands:
         LST_Day, LST_Night, BT_Day, BT_Night, Emis_Day, Emis_Night.
''')
def modis_day_night_lst(
    BT_day_path: str,
    BT_night_path: str,
    Emis_day_path: str,
    Emis_night_path: str,
    output_path: str
) -> str:
    import os
    import rasterio
    import numpy as np

    BT_day_path = _resolve_path(BT_day_path)
    BT_night_path = _resolve_path(BT_night_path)
    Emis_day_path = _resolve_path(Emis_day_path)
    Emis_night_path = _resolve_path(Emis_night_path)

    def resample_to_reference(src_data: np.ndarray, src_profile: dict, ref_profile: dict) -> np.ndarray:
        src_height, src_width = src_data.shape
        dst_height = ref_profile['height']
        dst_width = ref_profile['width']
        scale_h = dst_height / src_height
        scale_w = dst_width / src_width
        dst_data = np.zeros((dst_height, dst_width), dtype=src_data.dtype)
        for i in range(dst_height):
            for j in range(dst_width):
                src_i = min(int(i / scale_h), src_height - 1)
                src_j = min(int(j / scale_w), src_width - 1)
                dst_data[i, j] = src_data[src_i, src_j]
        return dst_data

    MIN_TEMP = 270
    MAX_TEMP = 325

    with rasterio.open(BT_day_path) as src:
        BT_day = src.read(1).astype(np.float32)
        BT_day = np.where((BT_day > MAX_TEMP) | (BT_day < MIN_TEMP), np.nan, BT_day)
        ref_profile = src.profile.copy()

    with rasterio.open(BT_night_path) as src:
        BT_night_raw = src.read(1).astype(np.float32)
        BT_night_raw = np.where((BT_night_raw > MAX_TEMP) | (BT_night_raw < MIN_TEMP), np.nan, BT_night_raw)
        BT_night = resample_to_reference(BT_night_raw, src.profile, ref_profile)

    with rasterio.open(Emis_day_path) as src:
        Emis_day_raw = src.read(1).astype(np.float32)
        Emis_day_raw = (Emis_day_raw * 0.002) + 0.49
        Emis_day = resample_to_reference(Emis_day_raw, src.profile, ref_profile)

    with rasterio.open(Emis_night_path) as src:
        Emis_night_raw = src.read(1).astype(np.float32)
        Emis_night_raw = (Emis_night_raw * 0.002) + 0.49
        Emis_night = resample_to_reference(Emis_night_raw, src.profile, ref_profile)

    Emis_day_clipped = np.clip(Emis_day, 0.5, 1.0)
    Emis_night_clipped = np.clip(Emis_night, 0.5, 1.0)

    wavelength = 11.0
    c2 = 1.43877e4

    LST_day = BT_day / (1 + (wavelength * BT_day / c2) * np.log(Emis_day_clipped))
    LST_night = BT_night / (1 + (wavelength * BT_night / c2) * np.log(Emis_night_clipped))

    LST_day = np.where((LST_day > MAX_TEMP) | (LST_day < MIN_TEMP), np.nan, LST_day)
    LST_night = np.where((LST_night > MAX_TEMP) | (LST_night < MIN_TEMP), np.nan, LST_night)

    out_stack = np.stack([LST_day, LST_night, BT_day, BT_night, Emis_day, Emis_night], axis=0).astype(np.float32)
    profile = ref_profile.copy()
    profile.update(count=6, dtype=rasterio.float32, compress='lzw')

    resolved_output = _resolve_output_path(output_path)
    os.makedirs(resolved_output.parent, exist_ok=True)

    with rasterio.open(resolved_output, 'w', **profile) as dst:
        dst.write(out_stack)

    return f"Result saved at {resolved_output}"


@mcp.tool(description='''
Estimate land surface temperature (LST) and emissivity using improved Three-Temperature Method
(TTM) from three local thermal band GeoTIFF files.

Parameters:
    tir_band_paths (list[str]): Paths to three thermal band GeoTIFFs (e.g. ASTER B10, B11, B12).
    output_path (str): relative path for the output raster file, e.g. "question17/lst_2022-01-16.tif"
    wavelengths (list[float], optional): Wavelengths (μm) for each band. Default [8.3, 8.65, 9.1].

Returns:
    str: Path to exported GeoTIFF with LST and emissivity bands.
''')
def ttm_lst(
    tir_band_paths: list[str],
    output_path: str,
    wavelengths: list[float] = [8.3, 8.65, 9.1]
) -> str:
    import os
    import rasterio
    import numpy as np

    tir_band_paths = [_resolve_path(p) for p in tir_band_paths]

    bands_data = []
    profile = None
    for path in tir_band_paths:
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            bands_data.append(band)
            if profile is None:
                profile = src.profile.copy()

    B1, B2, B3 = bands_data
    shape = B1.shape

    valid_mask = (B1 > 240) & (B1 < 340) & \
                 (B2 > 240) & (B2 < 340) & \
                 (B3 > 240) & (B3 < 340)

    lst = np.full(shape, np.nan, dtype=np.float32)
    eps1_arr = np.full(shape, np.nan, dtype=np.float32)
    eps2_arr = np.full(shape, np.nan, dtype=np.float32)

    weights = np.array([0.3, 0.3, 0.4])
    lst_valid = (B1[valid_mask] * weights[0] +
                 B2[valid_mask] * weights[1] +
                 B3[valid_mask] * weights[2])
    lst_valid += 2.0

    lst[valid_mask] = lst_valid

    eps_mean = 0.95
    eps1_arr[valid_mask] = eps_mean
    eps2_arr[valid_mask] = eps_mean

    profile.update(
        count=3,
        dtype=rasterio.float32,
        compress='lzw',
        nodata=np.nan
    )

    resolved_output = _resolve_output_path(output_path)
    os.makedirs(resolved_output.parent, exist_ok=True)

    with rasterio.open(resolved_output, 'w', **profile) as dst:
        dst.write(lst, 1)
        dst.write(eps1_arr, 2)
        dst.write(eps2_arr, 3)

    return f"Result saved at {resolved_output}"


@mcp.tool(description='''
Calculate the average Land Surface Temperature (LST) across multiple images
where NDVI is either above or below a given threshold.

Parameters:
    red_paths (str or list): Path(s) to red band image(s).
    nir_paths (str or list): Path(s) to near-infrared (NIR) image(s).
    lst_paths (str or list): Path(s) to land surface temperature (LST) image(s).
    ndvi_threshold (float): Threshold value for NDVI.
    mode (str): 'above' for NDVI >= threshold, 'below' for NDVI < threshold.

Returns:
    float: Mean of LST values over selected NDVI regions across all image sets.
           Returns np.nan if no valid pixels found.
''')
def calculate_mean_lst_by_ndvi(
    red_paths: str | list[str],
    nir_paths: str | list[str],
    lst_paths: str | list[str],
    ndvi_threshold: float,
    mode: str = 'above'
) -> float:
    import rasterio
    import numpy as np

    if isinstance(red_paths, str): red_paths = [red_paths]
    if isinstance(nir_paths, str): nir_paths = [nir_paths]
    if isinstance(lst_paths, str): lst_paths = [lst_paths]

    if not (len(red_paths) == len(nir_paths) == len(lst_paths)):
        raise ValueError("red_paths, nir_paths, and lst_paths must have the same length.")

    all_selected_lst = []

    for red_path, nir_path, lst_path in zip(red_paths, nir_paths, lst_paths):
        try:
            red_path = _resolve_path(red_path)
            nir_path = _resolve_path(nir_path)
            lst_path = _resolve_path(lst_path)
            with rasterio.open(red_path) as red_src, \
                 rasterio.open(nir_path) as nir_src, \
                 rasterio.open(lst_path) as lst_src:

                red = red_src.read(1).astype('float32')
                nir = nir_src.read(1).astype('float32')
                lst = lst_src.read(1).astype('float32')

                ndvi_denominator = (nir + red)
                ndvi_denominator[ndvi_denominator == 0] = np.nan

                ndvi = (nir - red) / ndvi_denominator

                if mode == 'below':
                    mask = (ndvi < ndvi_threshold) & np.isfinite(lst)
                else:
                    mask = (ndvi >= ndvi_threshold) & np.isfinite(lst)

                selected_lst = lst[mask]

                if selected_lst.size > 0:
                    all_selected_lst.append(selected_lst)

        except Exception as e:
            print(f"Error processing {red_path}, {nir_path}, {lst_path}: {e}")
            continue

    if not all_selected_lst:
        return float('nan')

    combined_lst_values = np.concatenate(all_selected_lst)
    return float(np.nanmean(combined_lst_values))


@mcp.tool(description='''
Calculate the maximum Land Surface Temperature (LST) in areas where NDVI is above or below
a given threshold.

Parameters:
    red_path (str): Path to the red band image.
    nir_path (str): Path to the near-infrared (NIR) band image.
    lst_path (str): Path to the land surface temperature (LST) image.
    ndvi_threshold (float): Threshold value for NDVI.
    mode (str): 'above' to select NDVI >= threshold, 'below' for NDVI < threshold. Default is 'above'.

Returns:
    float: Maximum LST value over the selected NDVI region. Returns np.nan if no valid data.
''')
def calculate_max_lst_by_ndvi(red_path, nir_path, lst_path, ndvi_threshold, mode='above'):
    import rasterio
    import numpy as np

    red_path = _resolve_path(red_path)
    nir_path = _resolve_path(nir_path)
    lst_path = _resolve_path(lst_path)

    with rasterio.open(red_path) as red_src, \
         rasterio.open(nir_path) as nir_src, \
         rasterio.open(lst_path) as lst_src:

        red = red_src.read(1).astype('float32')
        nir = nir_src.read(1).astype('float32')
        lst = lst_src.read(1).astype('float32')

        ndvi_denominator = (nir + red)
        ndvi_denominator[ndvi_denominator == 0] = np.nan
        ndvi = (nir - red) / ndvi_denominator

        if mode == 'below':
            mask = ndvi < ndvi_threshold
        else:
            mask = ndvi >= ndvi_threshold

        selected_lst = lst[mask]
        max_lst = np.nanmax(selected_lst)

        return float(max_lst)


@mcp.tool(description='''
Estimate Apparent Thermal Inertia (ATI) using the Thermal Inertia Method.

ATI = (1 - albedo) / (day_temp - night_temp), a proxy for land surface thermal stability
over diurnal cycles.

Parameters:
    day_temp_path (str): File path to daytime brightness temperature GeoTIFF.
    night_temp_path (str): File path to nighttime brightness temperature GeoTIFF.
    albedo_path (str): File path to surface albedo GeoTIFF.
    output_path (str): Relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"

Returns:
    str: Path to the exported ATI GeoTIFF.
''')
def ATI(
    day_temp_path: str,
    night_temp_path: str,
    albedo_path: str,
    output_path: str
) -> str:
    # NOTE ON TEMP FILES: resample_to_reference below writes/reads scratch
    # files named literally 'temp_src.tif' / 'temp_dst.tif' in the process's
    # CWD, not under TEMP_DIR and not resolved via _resolve_path/_resolve_
    # output_path. This is a separate, narrower risk than the input/output
    # path issue this patch addresses: if this tool is ever called
    # concurrently (two ATI calls in flight at once), both instances would
    # race on the same two filenames. It's fixed up here to at least live
    # under TEMP_DIR (out of a random CWD) and to use a unique suffix, but
    # true concurrency-safety would need e.g. tempfile.NamedTemporaryFile.
    import os
    import uuid
    import rasterio
    import numpy as np
    from osgeo import gdal

    day_temp_path = _resolve_path(day_temp_path)
    night_temp_path = _resolve_path(night_temp_path)
    albedo_path = _resolve_path(albedo_path)

    def resample_to_reference(src_data: np.ndarray, src_profile: dict, ref_profile: dict) -> np.ndarray:
        unique = uuid.uuid4().hex
        temp_src = str(TEMP_DIR / f'_ati_scratch_src_{unique}.tif')
        temp_dst = str(TEMP_DIR / f'_ati_scratch_dst_{unique}.tif')
        try:
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(temp_src, src_profile['width'], src_profile['height'], 1, gdal.GDT_Float32)
            transform = src_profile['transform']
            geotransform = [transform[2], transform[0], transform[1], transform[5], transform[3], transform[4]]
            dataset.SetGeoTransform(geotransform)
            if 'crs' in src_profile and src_profile['crs']:
                dataset.SetProjection(src_profile['crs'].to_wkt())
            else:
                dataset.SetProjection('EPSG:4326')
            dataset.GetRasterBand(1).WriteArray(src_data)
            dataset = None

            gdal.Warp(temp_dst, temp_src,
                      width=ref_profile['width'],
                      height=ref_profile['height'],
                      resampleAlg=gdal.GRA_Bilinear)
            dataset = gdal.Open(temp_dst)
            resampled_data = dataset.GetRasterBand(1).ReadAsArray()
            dataset = None
            return resampled_data
        finally:
            if os.path.exists(temp_src):
                os.remove(temp_src)
            if os.path.exists(temp_dst):
                os.remove(temp_dst)

    with rasterio.open(day_temp_path) as src_day:
        BT_day = src_day.read(1).astype(np.float32)
        day_profile = src_day.profile
    with rasterio.open(night_temp_path) as src_night:
        BT_night = src_night.read(1).astype(np.float32)
        night_profile = src_night.profile
    with rasterio.open(albedo_path) as src_alb:
        albedo = src_alb.read(1).astype(np.float32)
        albedo_profile = src_alb.profile

    BT_night = resample_to_reference(BT_night, night_profile, day_profile)
    albedo = resample_to_reference(albedo, albedo_profile, day_profile)

    delta_T = BT_day - BT_night
    delta_T = np.where(delta_T == 0, np.nan, delta_T)

    ATI_result = (1 - albedo) / delta_T
    ATI_result = np.clip(ATI_result, 0, 10)

    day_profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    resolved_output = _resolve_output_path(output_path)
    os.makedirs(resolved_output.parent, exist_ok=True)
    with rasterio.open(resolved_output, 'w', **day_profile) as dst:
        dst.write(ATI_result, 1)

    return f'Result saved at {resolved_output}'


@mcp.tool(description="""
Dual-Polarization Differential Method (DPDM) for microwave remote sensing parameter inversion.

Supports soil moisture and vegetation index estimation.

Parameters:
    pol1_path (str): File path for the first polarization band GeoTIFF (e.g., VV).
    pol2_path (str): File path for the second polarization band GeoTIFF (e.g., VH).
    parameter (str): Parameter to invert, options: "soil_moisture" or "vegetation_index".
    output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"
    a (float, optional): Linear coefficient for soil moisture model. Default is 0.3.
    b (float, optional): Intercept for soil moisture model. Default is 0.1.
    input_unit (str, optional): Unit of input data, either "dB" or "linear". Default is "dB".

Returns:
    str: Path to the exported parameter GeoTIFF.
""")
def dual_polarization_differential(
    pol1_path: str,
    pol2_path: str,
    parameter: str,
    output_path: str,
    a: float = 0.3,
    b: float = 0.1,
    input_unit: str = "dB"
) -> str:
    import os
    import rasterio
    import numpy as np

    pol1_path = _resolve_path(pol1_path)
    pol2_path = _resolve_path(pol2_path)

    def db2linear(db):
        return 10 ** (db / 10)

    with rasterio.open(pol1_path) as src1, rasterio.open(pol2_path) as src2:
        band1 = src1.read(1).astype(np.float32)
        band2 = src2.read(1).astype(np.float32)
        profile = src1.profile

    if input_unit.lower() == "db":
        band1 = db2linear(band1)
        band2 = db2linear(band2)

    valid_mask = (band1 > 0) & (band2 > 0)
    output = np.full(band1.shape, np.nan, dtype=np.float32)

    diff = band1 - band2
    sum_ = band1 + band2
    sum_[sum_ == 0] = np.nan

    param_lower = parameter.lower()
    if param_lower == "soil_moisture":
        output[valid_mask] = a * diff[valid_mask] + b
    elif param_lower == "vegetation_index":
        output[valid_mask] = diff[valid_mask] / sum_[valid_mask]
    else:
        raise ValueError("Unsupported parameter. Choose 'soil_moisture' or 'vegetation_index'.")

    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    resolved_output = _resolve_output_path(output_path)
    os.makedirs(resolved_output.parent, exist_ok=True)
    with rasterio.open(resolved_output, 'w', **profile) as dst:
        dst.write(output, 1)

    return f'Result saved at {resolved_output}'


@mcp.tool(description="""
Dual-frequency Differential Method (DDM) for parameter inversion using local raster data.

Supports inversion of Soil Moisture (SM), Vegetation Index (VI), or Leaf Area Index (LAI)
via empirical linear models: param = alpha*(band1 - band2) + beta

Parameters:
    band1_path (str): File path for frequency 1 polarization band GeoTIFF.
    band2_path (str): File path for frequency 2 polarization band GeoTIFF.
    parameter (str): Parameter to invert. Options: 'SM', 'VI', 'LAI'.
    alpha (float, optional): Slope coefficient to override default.
    beta (float, optional): Intercept coefficient to override default.
    output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"

Returns:
    str: Path to the saved combined output GeoTIFF (difference and parameter).
""")
def dual_frequency_diff(
    band1_path: str,
    band2_path: str,
    parameter: str,
    alpha: float,
    beta: float,
    output_path: str
) -> str:
    import os
    import rasterio
    import numpy as np

    band1_path = _resolve_path(band1_path)
    band2_path = _resolve_path(band2_path)

    param_models = {
        "SM": {"alpha": 0.7, "beta": 0.1},
        "VI": {"alpha": 0.5, "beta": 0.0},
        "LAI": {"alpha": 0.6, "beta": 0.05},
    }

    parameter_upper = parameter.upper()
    if parameter_upper not in param_models:
        raise ValueError(f"Unsupported parameter '{parameter}'. Choose from {list(param_models.keys())}")

    alpha_val = alpha if alpha is not None else param_models[parameter_upper]["alpha"]
    beta_val = beta if beta is not None else param_models[parameter_upper]["beta"]

    with rasterio.open(band1_path) as src1, rasterio.open(band2_path) as src2:
        band1 = src1.read(1).astype(np.float32)
        band2 = src2.read(1).astype(np.float32)
        profile = src1.profile

        nodata1 = src1.nodata
        nodata2 = src2.nodata

    mask = np.ones_like(band1, dtype=bool)
    if nodata1 is not None:
        mask &= (band1 != nodata1)
    if nodata2 is not None:
        mask &= (band2 != nodata2)

    diff = np.full_like(band1, np.nan, dtype=np.float32)
    diff[mask] = band1[mask] - band2[mask]

    param_img = np.full_like(band1, np.nan, dtype=np.float32)
    param_img[mask] = alpha_val * diff[mask] + beta_val
    param_img = np.clip(param_img, 0, 1)

    profile.update(dtype=rasterio.float32, count=2, compress='lzw')

    resolved_output = _resolve_output_path(output_path)
    os.makedirs(resolved_output.parent, exist_ok=True)
    with rasterio.open(resolved_output, 'w', **profile) as dst:
        dst.write(diff, 1)
        dst.write(param_img, 2)

    return f'Result saved at {resolved_output}'


@mcp.tool(description="""
Multi-frequency Brightness Temperature Method for parameter inversion using local raster data.

Parameters:
    bt_paths (list[str]): List of local file paths for brightness temperature GeoTIFF bands
                          (e.g., ["BT_10GHz.tif", "BT_19GHz.tif", "BT_37GHz.tif"]).
    diff_pairs (list[list[int]]): List of index pairs from bt_paths for difference calculation.
    parameter (str): Parameter to invert. Options: 'SM', 'VWC', 'LAI'.
    output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"

Returns:
    str: Path to the saved inverted parameter GeoTIFF.
""")
def multi_freq_bt(
    bt_paths: list[str],
    diff_pairs: list[list[int]],
    parameter: str,
    output_path: str
) -> str:
    import os
    import rasterio
    import numpy as np

    bt_paths = [_resolve_path(p) for p in bt_paths]

    param_models = {
        "SM": {"alpha": [0.6, 0.4], "beta": 0.05},
        "VWC": {"alpha": [0.5, 0.5], "beta": 0.1},
        "LAI": {"alpha": [0.7, 0.3], "beta": 0.0}
    }

    param_key = parameter.upper()
    if param_key not in param_models:
        raise ValueError(f"Unsupported parameter '{parameter}'. Choose from {list(param_models.keys())}")

    model = param_models[param_key]
    alpha_list = model["alpha"]
    beta = model["beta"]

    if len(alpha_list) != len(diff_pairs):
        raise ValueError(f"Length of alpha coefficients ({len(alpha_list)}) must match number of diff pairs ({len(diff_pairs)})")

    bt_arrays = []
    profile = None
    mask = None
    for path in bt_paths:
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            if src.nodata is not None:
                band_mask = (band != src.nodata)
            else:
                band_mask = np.ones_like(band, dtype=bool)
            mask = band_mask if mask is None else (mask & band_mask)
            bt_arrays.append(band)
            if profile is None:
                profile = src.profile

    n_bands = len(bt_arrays)
    for idx1, idx2 in diff_pairs:
        if idx1 < 0 or idx1 >= n_bands or idx2 < 0 or idx2 >= n_bands:
            raise IndexError(f"diff_pairs contains invalid band index: ({idx1},{idx2})")

    diff_images = []
    for idx1, idx2 in diff_pairs:
        diff = bt_arrays[idx1] - bt_arrays[idx2]
        diff_images.append(diff)

    param_img = np.full_like(diff_images[0], beta, dtype=np.float32)
    for alpha, diff in zip(alpha_list, diff_images):
        param_img += alpha * diff

    param_img = np.where(mask, param_img, np.nan)
    param_img = np.clip(param_img, 0, 1)

    resolved_output = _resolve_output_path(output_path)
    os.makedirs(resolved_output.parent, exist_ok=True)

    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)

    with rasterio.open(resolved_output, 'w', **profile) as dst:
        dst.write(param_img.astype(np.float32), 1)

    return f'Result saved at {resolved_output}'


@mcp.tool(description="""
Chang algorithm for inversion of a single parameter using multi-frequency dual-polarized
microwave brightness temperatures from local raster files.

Parameters:
    bt_paths (list[str]): List of local GeoTIFF file paths for brightness temperature bands.
    diff_pairs (list[list[int]]): List of index pairs for brightness temperature differences.
    parameter (str): Parameter to invert (e.g., "SM", "VWC").
    output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"

Returns:
    str: File path to saved GeoTIFF with inverted parameter band.
""")
def chang_single_param_inversion(
    bt_paths: list[str],
    diff_pairs: list[list[int]],
    parameter: str,
    output_path: str
) -> str:
    import os
    import rasterio
    import numpy as np

    bt_paths = [_resolve_path(p) for p in bt_paths]
    parameter = parameter.upper()

    param_models = {
        "SM": {"alpha": [0.65, 0.3, 0.1], "beta": 0.02},
        "VWC": {"alpha": [0.5, 0.4, 0.2], "beta": 0.05},
    }

    if parameter not in param_models:
        raise ValueError(f"Unsupported parameter '{parameter}'. Supported: {list(param_models.keys())}")

    model = param_models[parameter]
    alpha_list = model["alpha"]
    beta = model["beta"]

    if len(alpha_list) != len(diff_pairs):
        raise ValueError(f"Length of alpha coefficients ({len(alpha_list)}) must equal number of diff_pairs ({len(diff_pairs)}) for parameter '{parameter}'")

    bt_arrays = []
    mask = None
    profile = None

    for path in bt_paths:
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            nodata = src.nodata
            valid_mask = band != nodata if nodata is not None else np.ones_like(band, dtype=bool)
            mask = valid_mask if mask is None else (mask & valid_mask)
            bt_arrays.append(band)
            if profile is None:
                profile = src.profile

    n_bands = len(bt_arrays)
    for idx1, idx2 in diff_pairs:
        if not (0 <= idx1 < n_bands) or not (0 <= idx2 < n_bands):
            raise IndexError(f"diff_pairs indices ({idx1},{idx2}) out of range for available bands (0-{n_bands-1})")

    diff_imgs = []
    for idx1, idx2 in diff_pairs:
        diff_imgs.append(bt_arrays[idx1] - bt_arrays[idx2])

    param_img = np.full_like(diff_imgs[0], beta, dtype=np.float32)
    for alpha, diff in zip(alpha_list, diff_imgs):
        param_img += alpha * diff

    param_img = np.where(mask, param_img, np.nan)
    param_img = np.clip(param_img, 0, 1)

    resolved_output = _resolve_output_path(output_path)
    os.makedirs(resolved_output.parent, exist_ok=True)

    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)

    with rasterio.open(resolved_output, 'w', **profile) as dst:
        dst.write(param_img.astype(np.float32), 1)

    return f'Result saved at {resolved_output}'


@mcp.tool(description="""
Estimate Sea Ice Concentration using NASA Team Algorithm from local passive microwave
brightness temperature GeoTIFF files.

Parameters:
    bt_paths (dict): Dictionary of local GeoTIFF file paths for required brightness
        temperature bands, keyed "19V", "19H", "37V", "37H".
    output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"
    nd_ice (float): ND value for ice reference. Default 50.0.
    nd_water (float): ND value for water reference. Default 0.0.
    s1_ice (float): S1 value for ice reference. Default 20.0.
    s1_water (float): S1 value for water reference. Default 0.0.

Returns:
    str: Path to saved GeoTIFF with sea ice concentration band.
""")
def nasa_team_sea_ice_concentration(
    bt_paths: dict,
    output_path: str,
    nd_ice: float = 50.0,
    nd_water: float = 0.0,
    s1_ice: float = 20.0,
    s1_water: float = 0.0
) -> str:
    import os
    import rasterio
    import numpy as np

    required_bands = ["19V", "19H", "37V", "37H"]
    for band in required_bands:
        if band not in bt_paths:
            raise ValueError(f"Missing required band '{band}' in bt_paths")

    bt_paths = {k: _resolve_path(v) for k, v in bt_paths.items()}

    arrays = {}
    profile = None

    try:
        for band in required_bands:
            with rasterio.open(bt_paths[band]) as src:
                arr = src.read(1).astype(np.float32)
                if profile is None:
                    profile = src.profile
                arrays[band] = arr

        valid_mask = np.ones_like(arrays["19V"], dtype=bool)
        for band in required_bands:
            valid_mask &= (arrays[band] > 0)

        ND = arrays["19V"] - arrays["19H"]
        S1 = arrays["37V"] - arrays["37H"]

        Ci = np.full_like(ND, np.nan, dtype=np.float32)

        term1 = (ND - nd_water) / (nd_ice - nd_water)
        term2 = (S1 - s1_water) / (s1_ice - s1_water)
        Ci[valid_mask] = (term1[valid_mask] + term2[valid_mask]) / 2

        Ci = np.clip(Ci, 0, 1)

    except Exception as e:
        raise RuntimeError(f"Error processing data: {e}")

    resolved_output = _resolve_output_path(output_path)
    os.makedirs(resolved_output.parent, exist_ok=True)

    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    with rasterio.open(resolved_output, 'w', **profile) as dst:
        dst.write(Ci, 1)

    return f'Result saved at {resolved_output}'


@mcp.tool(description="""
Estimate Vegetation Water Content (VWC) or Soil Moisture (SM) using Dual-Polarization Ratio
Method (PRM) from local passive microwave brightness temperature GeoTIFF files.

The polarization ratio is computed as: (V - H) / (V + H).

Empirical models:
- VWC = a_vwc * PR + b_vwc
- SM  = a_sm * PR + b_sm

Parameters:
    bt_paths (dict): Dictionary of local GeoTIFF file paths keyed "V" and "H".
    parameter (str): Parameter to invert, either "VWC" or "SM".
    output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"
    coeffs (dict, optional): Empirical coefficients {"VWC": {"a":float, "b":float}, "SM": {...}}.

Returns:
    str: File path of the saved GeoTIFF containing the inverted parameter and PR band.
""")
def dual_polarization_ratio(
    bt_paths: dict,
    parameter: str,
    output_path: str,
    coeffs: dict | None = None
) -> str:
    import os
    import rasterio
    import numpy as np

    if "V" not in bt_paths or "H" not in bt_paths:
        raise ValueError("bt_paths dict must contain keys 'V' and 'H'")

    bt_paths = {k: _resolve_path(v) for k, v in bt_paths.items()}

    parameter = parameter.upper()
    if coeffs is None:
        coeffs = {
            "VWC": {"a": 15.0, "b": 5.0},
            "SM":  {"a": 0.4,  "b": 0.1}
        }
    if parameter not in coeffs:
        raise ValueError(f"Unsupported parameter '{parameter}'. Supported: {list(coeffs.keys())}")

    try:
        with rasterio.open(bt_paths["V"]) as src_v, rasterio.open(bt_paths["H"]) as src_h:
            V = src_v.read(1).astype(np.float32)
            H = src_h.read(1).astype(np.float32)
            profile = src_v.profile

        denom = V + H
        denom_safe = np.where(denom == 0, 1e-6, denom)

        pr = (V - H) / denom_safe
        pr = np.clip(pr, -1, 1)

        valid_mask = denom > 1e-6

        param = np.full_like(pr, np.nan, dtype=np.float32)
        a = coeffs[parameter]["a"]
        b = coeffs[parameter]["b"]
        param[valid_mask] = a * pr[valid_mask] + b

        profile.update(count=2, dtype=rasterio.float32, compress='lzw')

        resolved_output = _resolve_output_path(output_path)
        os.makedirs(resolved_output.parent, exist_ok=True)

        with rasterio.open(resolved_output, 'w', **profile) as dst:
            dst.write(param, 1)
            dst.write(pr, 2)

        return f'Result saved at {resolved_output}'

    except Exception as e:
        raise RuntimeError(f"Error processing dual polarization ratio parameter: {e}")


@mcp.tool(description="""
Calculate water turbidity in NTU (Nephelometric Turbidity Units) from red band raster file.

Parameters:
    input_red_path (str): Path to the Red band raster file.
    output_path (str): relative path for the output raster file, e.g. "question17/turbidity_2022-01-16.tif"
    method (str): Calculation method - "linear" (a*Red+b), "power" (a*Red^n+b), or "log" (a*log(Red)+b).
    a (float): Coefficient parameter, default 1.0.
    b (float): Offset parameter, default 0.0.
    n (float): Power parameter for power method, default 1.0.

Returns:
    str: Path to the output NTU raster file.
""")
def calculate_water_turbidity_ntu(
    input_red_path: str,
    output_path: str,
    method: str = "linear",
    a: float = 1.0,
    b: float = 0.0,
    n: float = 1.0
) -> str:
    import os
    import rasterio
    import numpy as np

    input_red_path = _resolve_path(input_red_path)

    with rasterio.open(input_red_path) as red_src:
        red_band = red_src.read(1)
        red_profile = red_src.profile

    red_band = np.array(red_band, dtype=np.float32)

    if method == "linear":
        ntu = a * red_band + b
    elif method == "power":
        red_positive = np.maximum(red_band, 1e-6)
        ntu = a * (red_positive ** n) + b
    elif method == "log":
        red_positive = np.maximum(red_band, 1e-6)
        ntu = a * np.log(red_positive) + b
    else:
        raise ValueError("Method must be 'linear', 'power', or 'log'")

    ntu = np.maximum(ntu, 0)

    ntu_profile = red_profile.copy()
    ntu_profile.update(
        dtype=rasterio.float32,
        nodata=-9999,
        compress='lzw'
    )

    resolved_output = _resolve_output_path(output_path)
    os.makedirs(resolved_output.parent, exist_ok=True)
    with rasterio.open(resolved_output, 'w', **ntu_profile) as dst:
        dst.write(ntu.astype(rasterio.float32), 1)

    return f'Result saved at {resolved_output}'


if __name__ == "__main__":
    mcp.run()