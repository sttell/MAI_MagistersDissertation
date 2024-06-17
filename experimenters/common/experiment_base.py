from experimenters.context import Context
from experimenters.common.model import evaluate

class ExperimentBase:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    def _train_models(self, models, train_data, test_data):
        print("Experiment start...")
        best_models = dict()
        for model_name, model in models.items():
            print(f"\tStart train process for {model_name} model.")
            model.train(train_data, test_data)
            print(f"\tEnd of train process.")
            print(f"\tBest model metrics on test:", model.best_result["metrics"])
            print(f"\tBest model params         :", model.best_result["params"])
            best_models[model_name] = {
                "model": model.best_model,
                "test_metrics": model.best_result["metrics"]
            }
        return best_models

    def _evaluate_models(self, best_models, valid_data):
        for _, model_data in best_models.items():
            best_models["valid_metrics"] = evaluate(model_data["model"], valid_data)

        final_model_name = ""
        final_model_data = {"valid_metrics": {"f1": 0.0}}
        for model_name, model_data in best_models.items():
            if model_data["valid_metrics"]["f1"] > final_model_data["valid_metrics"]["f1"]:
                final_model_data = model_data
                final_model_name = model_name
        print("Final model evaluation:")
        print(f"\tBest model - {final_model_name}.")
        print(f"\tEvaluation metrics on validation data:")
        print(f"\t\tF1        --- {final_model_data['valid_metrics']['f1']}")
        print(f"\t\tAccuracy  --- {final_model_data['valid_metrics']['accuracy']}")
        print(f"\t\tPrecision --- {final_model_data['valid_metrics']['precision']}")
        print(f"\t\tRecall    --- {final_model_data['valid_metrics']['recall']}")

    def run(self):
        print("Running base")

    @staticmethod
    def get_name():
        return "Base Experiment"