# MRIQC Guidelines

MRIQC is maintained independently by NiPreps and should be run outside this package.

1. Arrange scans in BIDS or a BIDS-like structure accepted by MRIQC.
2. Generate a Docker command:

```bash
uncs-interp mriqc-command --bids-dir /path/to/bids --output-dir outputs/mriqc
```

3. Run the printed command.
4. Convert the resulting group-level table to a compact CSV:

```bash
uncs-interp select-mriqc \
  --input-csv outputs/mriqc/group_T1w.tsv \
  --output-csv outputs/mriqc_selected.csv
```

5. Provide the selected file to feature extraction:

```bash
uncs-interp extract-features ... --mriqc-csv outputs/mriqc_selected.csv
```

The selected IQMs are `cjv`, `efc`, `icvs_gm`, `icvs_wm`, `inu_med`, `inu_range`, `rpve_gm`, `rpve_wm`, `snr_total`, `snr_gm`, `snr_wm`, `wm2max`, and `fwhm_avg`.
