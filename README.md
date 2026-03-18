# uav-coverage-benchmark

Benchmark repository for base-station coverage approximation methods with a fair shared train/test split.

## Implemented methods
- `0_convex_hull`
- `1_alpha_shape`
- `2_kriging`
- `3_idw`
- `4_ml`
- `5_gpr`

All methods are trained/fitted on the exact same training set and evaluated on the exact same test set.
Each method has dedicated tunable parameters in `configs/default.yaml`:
- minimum support
- outlier filter (`none | percentile | mad | dbscan`)
- method-specific hyperparameters (alpha, variogram settings, RF settings).
- one or more signal thresholds (`preprocess.signal_threshold_dbm`), e.g. `[-105,-110,-115,-120]`.

## Dataset
The repository expects the copied connectivity dataset under:
- `data/connectivity/dataset`
- `data/connectivity/dataset-old/{bike,car,train,walk}`

## Quick start
```bash
/home/fra/uavenv/bin/python run_experiments.py --config configs/default.yaml
```

Optional fast smoke run:
```bash
/home/fra/uavenv/bin/python run_experiments.py --config configs/default.yaml --max-cells 50
```

## Outputs
- `results/metrics/metrics_summary_<method>.csv`: macro metrics and runtime per method
- `results/metrics/split_info.csv`: split statistics (time split and sample counts)

## Optional dependencies
- `xgboost` is only required if you enable `methods.ml.model: xgb` in the config.
