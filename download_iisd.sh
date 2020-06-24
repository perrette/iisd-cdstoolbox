#!/bin/bash

suffix='-v2'
figures="--png-timeseries --png-region"

for location in Welkenraedt Johannesburg Uvinza; do
    for asset in energy buildings irrigation nature roads wastewater; do
        echo $location - $asset
        # With bias correction
        cmd="python download.py --asset $asset --model bnu_esm csiro_mk3_6_0 ipsl_cm5a_mr mpi_esm_mr --experiment rcp_8_5 rcp_4_5 --location $location $figures -o iisd_final$suffix"
        echo $cmd
        eval $cmd
    done
done

for location in Welkenraedt Johannesburg Uvinza; do
    for asset in energy buildings irrigation nature roads wastewater; do
        echo $location - $asset
        # Without bias correction
        cmd="python download.py --asset $asset --model bnu_esm csiro_mk3_6_0 ipsl_cm5a_mr mpi_esm_mr --experiment rcp_8_5 rcp_4_5 --location $location $figures -o iisd_final-no-bias-correction$suffix --no-bias-correction"
        echo $cmd
        eval $cmd
    done
done
