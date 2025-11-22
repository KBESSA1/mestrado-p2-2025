# Schema — complete_S2_allgeom_clim.csv

**Keys**: Sample, Sub_Sample, Date_std
**Targets**: CP, TDN_based_ADF
**Coords**: Lat, Lon

## Bands
B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12

## Geometries
2DAP5x5, BUF30m

## Stats
MED, IQR

Pattern: <Band>__<Geom>_<Stat>

## QC/Metadata
valid_px_ratio__2DAP5x5, valid_px_ratio__BUF30m, SCL_clear_prop__2DAP5x5, SCL_cloud_prop__2DAP5x5, SCL_shadow_prop__2DAP5x5, SCL_veg_prop__2DAP5x5, SCL_clear_prop__BUF30m, SCL_cloud_prop__BUF30m, SCL_shadow_prop__BUF30m, SCL_veg_prop__BUF30m, n_scenes_used, win_days_used, min_abs_delta_day, tile_id, crs, processing_baseline, sun_zenith, sun_azimuth, cloud_cover_scene, source_s2

## Climate columns (expected; placeholders added if missing)
precip_sum_d5/d7/d10, tavg_mean_d5/d7/d10, tmax_mean_d5/d7/d10, tmin_mean_d5/d7/d10, rh_mean_d5/d7/d10, rainy_days_d5/d7/d10, insolation_h_d5/d7/d10