import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference_lensless")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # Setup data_loader instances
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # Build model architecture, then print to console
    model = instantiate(config.model).to(device)
    print(model)

    # Get metrics
    metrics = instantiate(config.metrics)

    # Save_path for model predictions
    save_path = Path(config.inferencer.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    # skip_model_load=True for models without checkpoints (e.g. ADMM-100)
    skip_model_load = config.inferencer.get("skip_model_load", False)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=skip_model_load,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
