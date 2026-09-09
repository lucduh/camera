import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from donut_camera.attention import apply_attention
from donut_camera.data import DonutDataset, load_samples
from donut_camera.model import autocast, load_bundle
from donut_camera.records import write_record
from donut_camera.tasks import TaskSpec
from donut_camera.timing import synchronize


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    train_json: Path
    validation_json: Path
    output: Path
    run_name: str
    height: int
    width: int
    max_length: int
    batch_size: int
    workers: int
    epochs: int
    limit_train_batches: int | None
    limit_validation_batches: int | None
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    grad_clip: float
    precision: str
    seed: int
    device: str
    attention_backend: str = "baseline"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, *, batch_size: int, workers: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )


@torch.no_grad()
def validation_loss(
    model,
    loader,
    *,
    device: str,
    precision: str,
    limit_batches: int | None = None,
) -> tuple[float, int, int, float]:
    model.eval()
    losses = []
    documents = 0
    synchronize()
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast(device, precision):
            losses.append(model(pixel_values=pixels, labels=labels).loss.item())
        documents += pixels.shape[0]
    synchronize()
    seconds = time.perf_counter() - started
    mean_loss = sum(losses) / len(losses) if losses else float("nan")
    return mean_loss, documents, len(losses), seconds


def train(task: TaskSpec, config: TrainingConfig) -> Path:
    seed_everything(config.seed)
    run_directory = config.output / config.run_name
    bundle = load_bundle(
        config.model,
        task=task,
        device=config.device,
        dtype="fp32",
        training=True,
    )
    apply_attention(bundle.model, config.attention_backend)
    bundle.set_resolution(config.height, config.width)

    train_data = DonutDataset(
        load_samples(config.train_json), bundle.processor, task, config.max_length
    )
    validation_data = DonutDataset(
        load_samples(config.validation_json),
        bundle.processor,
        task,
        config.max_length,
    )
    train_loader = make_loader(
        train_data,
        batch_size=config.batch_size,
        workers=config.workers,
        shuffle=True,
        seed=config.seed,
    )
    validation_loader = make_loader(
        validation_data,
        batch_size=config.batch_size,
        workers=config.workers,
        shuffle=False,
        seed=config.seed,
    )

    model = bundle.model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    if config.limit_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, config.limit_train_batches)
    total_steps = steps_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    cuda_device = (
        torch.device(config.device)
        if torch.device(config.device).type == "cuda"
        else None
    )
    history = []
    best_loss = float("inf")
    best_epoch = None
    overall_peak_mb = 0.0
    optimizer.zero_grad(set_to_none=True)
    synchronize()
    loop_started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        synchronize()
        start_allocated = (
            torch.cuda.memory_allocated(cuda_device)
            if cuda_device is not None
            else None
        )
        if cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device)
        started = time.perf_counter()
        loss_sum = 0.0
        documents = 0
        steps = 0
        for batch_index, batch in enumerate(
            tqdm(train_loader, desc=f"epoch {epoch}/{config.epochs}")
        ):
            if (
                config.limit_train_batches is not None
                and batch_index >= config.limit_train_batches
            ):
                break
            pixels = batch["pixel_values"].to(config.device, non_blocking=True)
            labels = batch["labels"].to(config.device, non_blocking=True)
            with autocast(config.device, config.precision):
                loss = model(pixel_values=pixels, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item()
            documents += pixels.shape[0]
            steps += 1
        synchronize()
        epoch_seconds = time.perf_counter() - started
        val_loss, validation_documents, validation_batches, validation_seconds = (
            validation_loss(
                model,
                validation_loader,
                device=config.device,
                precision=config.precision,
                limit_batches=config.limit_validation_batches,
            )
        )
        peak_allocated = (
            torch.cuda.max_memory_allocated(cuda_device)
            if cuda_device is not None
            else None
        )
        peak_mb = peak_allocated / 1024**2 if peak_allocated is not None else None
        if peak_mb is not None:
            overall_peak_mb = max(overall_peak_mb, peak_mb)
        row = {
            "epoch": epoch,
            "documents": documents,
            "steps": steps,
            "train_loss": loss_sum / steps if steps else float("nan"),
            "validation_loss": val_loss,
            "training_seconds": epoch_seconds,
            "validation_seconds": validation_seconds,
            "epoch_seconds": epoch_seconds + validation_seconds,
            "training_documents_per_second": documents / epoch_seconds,
            "validation_documents": validation_documents,
            "validation_batches": validation_batches,
            "peak_allocated_mb": peak_mb,
            "incremental_peak_mb": (
                (peak_allocated - start_allocated) / 1024**2
                if peak_allocated is not None and start_allocated is not None
                else None
            ),
        }
        history.append(row)
        print(row)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            bundle.save(run_directory / "best")

    synchronize()
    loop_wall_seconds = time.perf_counter() - loop_started
    bundle.save(run_directory / "last")
    write_record(
        run_directory / "train.json",
        config={"task": task.name, **asdict(config)},
        measurements={
            "best_validation_loss": best_loss,
            "best_epoch": best_epoch,
            "fine_tuning_seconds": sum(row["epoch_seconds"] for row in history),
            "loop_wall_seconds_including_best_checkpoints": loop_wall_seconds,
            "peak_allocated_mb": overall_peak_mb if cuda_device is not None else None,
            "epochs": history,
        },
    )
    return run_directory
