# Experiment Notes

The local dataset contains 14,122 training images, 3,099 validation images, and 4,306 test images. Each split has the same number of label files as images. Matching counts do not establish annotation correctness. Duplicate-image and cross-split leakage checks have not yet been performed.

## Preserved Records

`reports/test_run-3.csv` and `reports/test_run-4.csv` are byte-for-byte copies of the original training CSV files. Each contains a single epoch row. The original prediction script used the best.pt checkpoint from test_run-3, so test_run-4 metrics must not be attributed to that checkpoint.

Shared settings: RT-DETR-L, epochs=1, imgsz=640, batch=2, workers=0, device=0, seed=0, pretrained=True, optimizer=auto. The original args.yaml files contain machine-specific paths; only relevant experiment settings are documented publicly. Original dataset configurations and weights remain local.

## Class Mapping Discrepancy

Two YAML configurations were found in the original project: one used `0=fire, 1=smoke`, while the configuration actually used by the training script specified `0=smoke, 1=fire`. The published CLI follows the training configuration. Spot checks found class 0 consistent with smoke in AoF03796 and class 1 consistent with fire in AoF04076. These checks do not replace a complete annotation audit.

## Interpretation Limits

These records demonstrate that the training pipeline ran and produced checkpoints and validation results. Differences between runs cannot be directly attributed to algorithmic improvements. FPS is not reported because no reliable timing protocol was used. Test-set accuracy is not reported because no new test-set evaluation was performed.
