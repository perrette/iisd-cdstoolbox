#!/bin/bash

for location in Welkenraedt Johannesburg Uvinza; do
    for asset in energy buildings irrigation nature roads wastewater; do
        echo $location - $asset
        cmd="python download.py --asset $asset --model bnu_esm csiro_mk3_6_0 ipsl_cm5a_mr mpi_esm_mr --experiment rcp_8_5 rcp_4_5 --location $location --no-bias-correction -o iisd_final"
        echo $cmd
        eval $cmd
        cmd="python download.py --asset $asset --model bnu_esm csiro_mk3_6_0 ipsl_cm5a_mr mpi_esm_mr --experiment rcp_8_5 rcp_4_5 --location $location --png-timeseries --png-region -o iisd_final"
        echo $cmd
        eval $cmd
    done
done
