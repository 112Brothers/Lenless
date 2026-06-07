import torch
from torchvision.utils import save_image
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer for lensless image reconstruction.
    Saves reconstructions as PNG files.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # Define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # Path definition
        self.save_path = save_path

        # Define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # Init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        # Save reconstructions as PNG
        batch_size = batch["reconstruction"].shape[0]
        for i in range(batch_size):
            recon = batch["reconstruction"][i].detach().cpu()
            if "image_id" in batch:
                img_id = batch["image_id"][i]
            else:
                img_id = f"{batch_idx * batch_size + i:06d}"
            save_path = self.save_path / part / f"{img_id}.png"
            save_path.parent.mkdir(exist_ok=True, parents=True)
            save_image(recon, save_path)

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions.
        """
        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        if self.evaluation_metrics is not None:
            return self.evaluation_metrics.result()
        return {}
