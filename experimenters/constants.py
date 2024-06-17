from experimenters.sc import CNNExperiment, CMCExperiment, SCExperiment

def _exp_to_kv(exp):
    return exp.get_name(), exp


# Possible Experimenter classes
EXPERIMENT_CLASSES = {
    CNNExperiment,
    CMCExperiment,
    SCExperiment
}

# Mapping --->    Experiment name -> Experiment class
EXPERIMENTERS_MAP = {exp.get_name(): exp for exp in EXPERIMENT_CLASSES}

# Extract unique experimenter names
EXPERIMENTERS_UNIQUE_NAMES_STRING = ', '.join(EXPERIMENTERS_MAP.keys())
