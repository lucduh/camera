import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

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
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    grad_clip: float
    precision: str
    seed: int
    device: str


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
def validation_loss(model, loader, *, device: str, precision: str) -> float:
    model.eval()
    losses = []
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast(device, precision):
            losses.append(model(pixel_values=pixels, labels=labels).loss.item())
    return sum(losses) / len(losses) if losses else float("nan")


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
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    best_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, config.epochs + 1):
        model.train()
        synchronize()
        started = time.perf_counter()
        loss_sum = 0.0
        documents = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{config.epochs}"):
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
        synchronize()
        epoch_seconds = time.perf_counter() - started
        val_loss = validation_loss(
            model,
            validation_loader,
            device=config.device,
            precision=config.precision,
        )
        row = {
            "epoch": epoch,
            "documents": documents,
            "steps": len(train_loader),
            "train_loss": loss_sum / len(train_loader),
            "validation_loss": val_loss,
            "wall_seconds": epoch_seconds,
            "documents_per_second": documents / epoch_seconds,
        }
        history.append(row)
        print(row)
        if val_loss < best_loss:
            best_loss = val_loss
            bundle.save(run_directory / "best")

    bundle.save(run_directory / "last")
    write_record(
        run_directory / "train.json",
        config={"task": task.name, **asdict(config)},
        measurements={"best_validation_loss": best_loss, "epochs": history},
    )
    return run_directory
