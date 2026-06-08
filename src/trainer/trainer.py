from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class for lensless image reconstruction.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            if self.optimizer is not None:
                self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train and self.optimizer is not None:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Update metrics for each loss
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.
        """
        if mode == "train":
            # Log sample images: lensless | ground truth | reconstruction
            # Only log if tensors have 3 channels (C, H, W)
            if batch["lensless"][0].shape[0] == 3:
                self.writer.add_image(
                    "lensless", batch["lensless"][0].cpu()
                )
            if "lensed" in batch and batch["lensed"][0].shape[0] == 3:
                self.writer.add_image(
                    "ground_truth", batch["lensed"][0].cpu()
                )
            if batch["reconstruction"][0].shape[0] == 3:
                self.writer.add_image(
                    "reconstruction", batch["reconstruction"][0].detach().cpu()
                )
        else:
            # Log a few samples during evaluation
            for i in range(min(3, batch["reconstruction"].shape[0])):
                if batch["reconstruction"][i].shape[0] == 3:
                    self.writer.add_image(
                        f"sample_{i}_reconstruction",
                        batch["reconstruction"][i].detach().cpu()
                    )
                if "lensed" in batch and batch["lensed"][i].shape[0] == 3:
                    self.writer.add_image(
                        f"sample_{i}_ground_truth",
                        batch["lensed"][i].cpu()
                    )
