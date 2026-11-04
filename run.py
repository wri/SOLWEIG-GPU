"""
Run the SOLWEIG-GPU pipeline (processing.py) with the settings of a config file.

    python run.py                   # config.toml next to this file
    python run.py other_city.toml   # any other config file
"""
import argparse
import os

from processing import load_config, run_pipeline

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SOLWEIG-GPU pipeline with the settings of a config file.")
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG,
                        help="TOML config file (default: config.toml next to run.py)")
    run_pipeline(load_config(parser.parse_args().config))
