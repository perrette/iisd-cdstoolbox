#!/bin/bash

suffix='-v2'
figures="--png-timeseries --png-region"   # comment out if figures are not needed

# With bias correction (the default)
for location in Welkenraedt Johannesburg Uvinza; do
    for asset in energy buildings irrigation nature roads wastewater; do
        echo $location - $asset
        cmd="python download.py --asset $asset --model bnu_esm csiro_mk3_6_0 ipsl_cm5a_mr mpi_esm_mr --experiment rcp_8_5 rcp_4_5 --location $location $figures -o iisd_final$suffix"
        echo $cmd
        eval $cmd
    done
done

# Without bias correction
for location in Welkenraedt Johannesburg Uvinza; do
    for asset in energy buildings irrigation nature roads wastewater; do
        echo $location - $asset
        cmd="python download.py --asset $asset --model bnu_esm csiro_mk3_6_0 ipsl_cm5a_mr mpi_esm_mr --experiment rcp_8_5 rcp_4_5 --location $location $figures -o iisd_final-no-bias-correction$suffix --no-bias-correction"
        echo $cmd
        eval $cmd
    done
done

# Create archive
cmd="zip iisd_final$suffix.zip iisd_final$suffix iisd_final-no-bias-correction$suffix -r"
echo $cmd
eval $cmd
