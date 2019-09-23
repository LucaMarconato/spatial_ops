rule all:
    input:
         # "snakemake/csv_eda"
         # "snakemake/vae"
         # "snakemake/ome_viewer"
         "snakemake/vae_viewer"
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

rule ome_viewer:
    output:
          "snakemake/ome_viewer"
    shell:
         """
         python3 -m spatial_ops.gui.ome_viewer
         mkdir -p snakemake; touch snakemake/ome_viewer
         """

rule vae_viewer:
    output:
          "snakemake/vae_viewer"
    shell:
         """
         python3 -m spatial_ops.gui.vae_viewer
         mkdir -p snakemake; touch snakemake/vae_viewer
         """
rule vae:
    output:
          "snakemake/vae"
    shell:
         """
         python3 -m sandbox.vae
         mkdir -p snakemake; touch snakemake/vae
         """
