rule all:
    input:
         # "snakemake/csv_eda"
         # "snakemake/test_vae"
         "snakemake/ome_eda"
    shell:
         # the rules for which the output files are removed basically act as .PHONY targets
         "rm snakemake/*"

rule clean:
    shell:
         "rm snakemake/*"

rule csv_eda:
    output:
          "snakemake/csv_eda"
    shell:
         """
         python3 -m sandbox.csv_eda
         # can dependencies be specified without having to deal with files?
         mkdir -p snakemake; touch snakemake/csv_eda
         """

rule ome_eda:
    output:
          "snakemake/ome_eda"
    shell:
         """
         python3 -m sandbox.ome_eda
         mkdir -p snakemake; touch snakemake/ome_eda
         """

rule test_vae:
    output:
          "snakemake/test_vae"
    shell:
         """
         python3 -m sandbox.vae
         mkdir -p snakemake; touch snakemake/test_vae
         """
