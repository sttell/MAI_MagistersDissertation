import argparse
from experimenters.context import Context
from experimenters.constants import EXPERIMENTERS_MAP, EXPERIMENTERS_UNIQUE_NAMES_STRING

# Command line arguments
parser = argparse.ArgumentParser(
    prog='Experimenter',
    description='Program for experiments running'
)
parser.add_argument(
    'experiment_name',
    help=f"Type of experiment. One of: {EXPERIMENTERS_UNIQUE_NAMES_STRING}"
)
parser.add_argument(
    '--dataset_path',
    default="./data",
    help=f"Path to directory with dataset."
)
parser.add_argument(
    '--config_path',
    default="./experiments/configs",
    help=f"Path to directory with configs."
)
args = parser.parse_args()


# Validate experiment name
experiment_name = args.experiment_name
if not experiment_name in EXPERIMENTERS_MAP.keys():
    raise ValueError("Unknown experiment name. Please, see --help for more info about experiment_name values.")

# Create context
ctx = Context(args)
ctx.validate()

# Run experiment
exp_state = EXPERIMENTERS_MAP[experiment_name](ctx)
exit_code = exp_state.run()
exit(exit_code)