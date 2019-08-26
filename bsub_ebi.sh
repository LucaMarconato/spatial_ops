#!/bin/bash
ssh ebi-login 'bash -lc "bsub -q research-rh74 -P gpu -gpu - -M 7000 \"cd deployed/spatial_ops; source ~/.bashrc; conda activate spatial_ops-dev-requirements; snakemake\""'
#ssh ebi-login 'bsub -q research-rh74 -P gpu -oo my_stdout -eo my_stderr -gpu - -M 7000 bash run.sh'
