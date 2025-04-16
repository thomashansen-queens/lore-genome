"""
This pipeline quickly queries and summarizes protein data from NCBI.

It is essentially a python wrapper around the NCBI Datasets API, made using
OpenAPI to generate the client code.

Specify parameters in config.yaml before running.

Refer to the README.md for usage instructions.
"""
import logging

from pipeline.pipeline import GenomePipeline
from pipeline.config import load_config, PipelineConfig

def main():
    """Main entry point."""
    config_dict = load_config('config.yaml')
    config = PipelineConfig(**config_dict)
    logging.info("Configuration loaded successfully!")
    pipeline = GenomePipeline(config)
    pipeline.run()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    main()
