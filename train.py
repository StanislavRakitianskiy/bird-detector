"""Навчання (fine-tune) YOLO26n для пошуку птахів (один клас — bird).

Що робить скрипт:
1. Перевіряє, що датасет існує, і робить копію data/bird.yaml з АБСОЛЮТНИМ шляхом (див. resolve_data_yaml).
2. Донавчає модель (за замовчуванням стартові ваги yolo26n.pt) на датасеті з prepare_dataset.py.
3. Копіює найкращі ваги в weights/best.pt, а графіки навчання — в examples/training/ (для README).
4. Окремо перевіряє збережені ваги на val і пише метрики та параметри запуску в weights/metrics.json.

Запуск без аргументів (`python train.py`) працює з усіма значеннями за замовчуванням.
"""

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import YAML

log = logging.getLogger("train")

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# Які графіки копіюємо в README. Точні назви залежать від версії Ultralytics, тому шукаємо через glob.
PLOT_PATTERNS = ("results.png", "confusion_matrix.png", "*_curve.png")


def setup_logging() -> None:
    """Налаштовує формат логів (одна точка налаштування для всього скрипта)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Читає аргументи командного рядка."""
    parser = argparse.ArgumentParser(
        description="Навчання YOLO26n для детекції птахів: ваги й метрики потрапляють у weights/.",
        add_help=False,  # свій -h нижче, щоб довідка була українською
    )
    parser.add_argument("-h", "--help", action="help", help="показати цю довідку і вийти")
    parser.add_argument("--data", type=Path, default=Path("data/bird.yaml"),
                        help="конфіг датасету (відносно кореня проєкту), за замовчуванням data/bird.yaml")
    parser.add_argument("--model", default="yolo26n.pt",
                        help="стартові ваги, за замовчуванням yolo26n.pt (шукаємо в корені проєкту)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="максимальна кількість епох, за замовчуванням 50")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="розмір картинки при навчанні, за замовчуванням 640")
    parser.add_argument("--batch", type=int, default=8,
                        help="розмір батча; якщо не вистачає пам'яті GPU — зменшуйте, за замовчуванням 8")
    parser.add_argument("--device", default=None,
                        help="пристрій: 0 (перший GPU), mps або cpu; за замовчуванням вибирається автоматично "
                             "(cuda, потім mps, потім cpu)")
    parser.add_argument("--patience", type=int, default=15,
                        help="зупинити навчання, якщо якість не росте стільки епох поспіль, за замовчуванням 15")
    parser.add_argument("--seed", type=int, default=42,
                        help="зерно генератора випадкових чисел, за замовчуванням 42")
    parser.add_argument("--workers", type=int, default=4,
                        help="кількість процесів, що читають дані, за замовчуванням 4")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="яку частку train брати для навчання (0..1], для швидких тестів; "
                             "val завжди береться повністю, за замовчуванням 1.0")
    parser.add_argument("--name", default="bird",
                        help="назва запуску: результати лягають у runs/<name>, за замовчуванням bird")
    parser.add_argument("--no-amp", action="store_true",
                        help="вимкнути змішану точність (AMP); потрібно, якщо на GTX 16xx у loss з'являється nan")
    parser.add_argument("--out-dir", type=Path, default=Path("weights"),
                        help="куди покласти best.pt і metrics.json (відносно кореня проєкту), "
                             "за замовчуванням weights")
    parser.add_argument("--plots-dir", type=Path, default=Path("examples/training"),
                        help="куди скопіювати графіки навчання для README (відносно кореня проєкту), "
                             "за замовчуванням examples/training")
    args = parser.parse_args()
    if min(args.epochs, args.imgsz, args.batch, args.patience) < 1:
        parser.error("--epochs, --imgsz, --batch і --patience мають бути не менше 1")
    if args.workers < 0:
        parser.error("--workers не може бути від'ємним")
    if not 0 < args.fraction <= 1:
        parser.error("--fraction має бути в межах (0, 1]")
    return args


# --------------------------------------------------------------------------- пристрій і датасет

def pick_device(requested: str | None) -> str:
    """Повертає пристрій для навчання: те, що просили, або автоматично cuda -> mps -> cpu."""
    if requested:
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_label(device: str) -> str:
    """Людська назва пристрою для логів: для GPU додає назву відеокарти."""
    first = device.split(",")[0].replace("cuda:", "").replace("cuda", "0")
    if first.isdigit() and torch.cuda.is_available():
        return f"{device} ({torch.cuda.get_device_name(int(first))})"
    return device


def has_images(folder: Path) -> bool:
    """True, якщо папка існує і в ній є хоч одне фото."""
    return folder.is_dir() and any(p.suffix.lower() in IMAGE_SUFFIXES for p in folder.iterdir())


def resolve_data_yaml(data_arg: Path) -> Path:
    """Перевіряє датасет і повертає шлях до копії yaml з АБСОЛЮТНИМ `path`.

    Домовленість з prepare_dataset.py: у data/bird.yaml `path` відносний до кореня проєкту
    (щоб у git не було шляхів з нашого комп'ютера). Але Ultralytics рахує відносний `path`
    від своєї папки datasets_dir, а не від yaml, тому тут ми робимо шлях абсолютним
    і кладемо копію в runs/ (поза git). Сам data/bird.yaml не змінюється.
    """
    yaml_path = PROJECT_ROOT / data_arg  # якщо --data абсолютний, Path поверне саме його
    if not yaml_path.is_file():
        raise SystemExit(f"Не знайдено конфіг датасету {yaml_path}. Спочатку запустіть python prepare_dataset.py")
    data = YAML.load(yaml_path)
    dataset_dir = (PROJECT_ROOT / data.get("path", "")).resolve()  # якщо path уже абсолютний — лишиться він
    for key in ("train", "val"):
        folder = dataset_dir / data.get(key, "")
        if not has_images(folder):
            raise SystemExit(f"У {folder} немає фото. Спочатку запустіть python prepare_dataset.py")
    data["path"] = dataset_dir.as_posix()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RUNS_DIR / f"{yaml_path.stem}_abs.yaml"
    YAML.save(dest, data)
    log.info("Датасет: %s (конфіг з абсолютним шляхом: %s)", dataset_dir, dest)
    return dest


# --------------------------------------------------------------------------- навчання

def train_model(args: argparse.Namespace, data_path: Path) -> dict[str, Any]:
    """Запускає навчання. Повертає словник з підсумками запуску (папка, best.pt, епохи, час, AMP)."""
    local_weights = PROJECT_ROOT / args.model
    model = YOLO(str(local_weights) if local_weights.exists() else args.model)  # інакше Ultralytics завантажить сам
    started = time.perf_counter()
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        seed=args.seed,
        workers=args.workers,
        fraction=args.fraction,
        name=args.name,
        amp=not args.no_amp,
        deterministic=True,
        # project — абсолютний шлях: відносний Ultralytics рахує від свого runs_dir і додає ще підпапку detect/
        project=str(RUNS_DIR),
        exist_ok=False,  # якщо runs/<name> уже є, створиться <name>2, <name>3... — старі результати не затираються
        plots=True,
        # optimizer="auto" — свідомо НЕ чіпаємо оптимізатор і lr0: Ultralytics сам обирає їх за кількістю
        # ітерацій (batch/64 * epochs). Менше 10 000 → AdamW з lr0=0.002 (для 1 класу), більше → MuSGD з lr0=0.01.
        # Це добре підібрані значення для донавчання, а ручний підбір lr — окрема велика задача.
        optimizer="auto",
    )
    trainer = model.trainer
    return {
        "run_dir": Path(trainer.save_dir),
        "best_path": Path(trainer.best),  # шлях бере сам trainer, нічого не хардкодимо
        "epochs_done": trainer.epoch + 1,  # епохи нумеруються з 0; при ранній зупинці їх менше, ніж --epochs
        "best_epoch": trainer.stopper.best_epoch,  # Ultralytics уже рахує її з 1
        "train_seconds": time.perf_counter() - started,  # разом із валідацією після кожної епохи
        "amp_used": bool(trainer.amp),  # Ultralytics міг сам вимкнути AMP, якщо його перевірка не пройшла
    }


# --------------------------------------------------------------------------- ваги, графіки, метрики

def export_best_weights(best_path: Path, out_dir: Path) -> Path:
    """Копіює найкращі ваги запуску в out_dir/best.pt і повертає шлях до копії."""
    if not best_path.is_file():
        raise SystemExit(f"Навчання не створило {best_path} — дивіться повідомлення про помилку вище")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "best.pt"
    shutil.copy2(best_path, dest)
    log.info("Найкращі ваги збережено: %s", dest)
    return dest


def copy_training_plots(run_dir: Path, plots_dir: Path) -> None:
    """Копіює графіки навчання з папки запуску в plots_dir. Якщо чогось немає — лише попередження."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in PLOT_PATTERNS:
        found = sorted(run_dir.glob(pattern))
        if not found:
            log.warning("Графік %s не знайдено в %s — пропускаю", pattern, run_dir)
        for plot in found:
            shutil.copy2(plot, plots_dir / plot.name)
            copied += 1
    log.info("Скопійовано графіків: %d -> %s", copied, plots_dir)


def load_dataset_stats(data_path: Path) -> dict[str, Any] | None:
    """Читає stats.json, який лежить у папці датасету (створюється prepare_dataset.py)."""
    stats_path = Path(YAML.load(data_path)["path"]) / "stats.json"
    if not stats_path.is_file():
        return None
    return json.loads(stats_path.read_text(encoding="utf-8"))


def validate_weights(weights: Path, data_path: Path, args: argparse.Namespace, run_dir: Path) -> dict[str, float]:
    """Окремо перевіряє збережені ваги на val і повертає precision, recall, mAP50, mAP50-95."""
    box = YOLO(str(weights)).val(
        data=str(data_path),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        plots=False,  # графіки вже є в папці навчання
        project=str(run_dir.parent),
        name=f"{run_dir.name}_val",
    ).box
    return {
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
    }


def evaluate_and_save_metrics(
    weights: Path, data_path: Path, args: argparse.Namespace, train_info: dict[str, Any], out_dir: Path
) -> dict[str, float]:
    """Рахує метрики збережених ваг на val, пише out_dir/metrics.json і виводить підсумок у лог."""
    metrics = validate_weights(weights, data_path, args, train_info["run_dir"])
    seconds = round(train_info["train_seconds"])
    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "evaluated_on": "val",
        "metrics": metrics,
        "epochs_done": train_info["epochs_done"],
        "best_epoch": train_info["best_epoch"],
        "train_time_sec": seconds,
        "train_time": str(timedelta(seconds=seconds)),
        "device": device_label(args.device),
        "amp_used": train_info["amp_used"],
        "params": {
            **{key: value.as_posix() if isinstance(value, Path) else value for key, value in vars(args).items()},
            "deterministic": True,
            "optimizer": "auto",
        },
        "versions": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "python": sys.version.split()[0],
        },
        "dataset": load_dataset_stats(data_path),
    }
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("Метрики збережено: %s", path)

    log.info("Підсумок (val): precision=%.4f  recall=%.4f  mAP50=%.4f  mAP50-95=%.4f",
             metrics["precision"], metrics["recall"], metrics["mAP50"], metrics["mAP50-95"])
    log.info("Епох пройдено: %d (найкраща: %d), час навчання: %s",
             train_info["epochs_done"], train_info["best_epoch"], report["train_time"])
    return metrics


# --------------------------------------------------------------------------- головна функція

def main() -> None:
    """Точка входу: перевірка датасету -> навчання -> best.pt і графіки -> метрики на val."""
    setup_logging()
    args = parse_args()
    args.device = pick_device(args.device)
    out_dir = PROJECT_ROOT / args.out_dir  # якщо шлях абсолютний, Path поверне саме його
    plots_dir = PROJECT_ROOT / args.plots_dir
    amp_text = "вимкнено (--no-amp)" if args.no_amp else "дозволено (Ultralytics сам вимкне, якщо перевірка не пройде)"
    log.info("Пристрій: %s; AMP (змішана точність): %s", device_label(args.device), amp_text)

    data_path = resolve_data_yaml(args.data)
    try:
        train_info = train_model(args, data_path)
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            "Не вистачило відеопам'яті. Зменшіть --batch (напр. 4 або 2), а якщо не допоможе — --imgsz (напр. 512)"
        ) from None
    best = export_best_weights(train_info["best_path"], out_dir)
    copy_training_plots(train_info["run_dir"], plots_dir)
    evaluate_and_save_metrics(best, data_path, args, train_info, out_dir)
    log.info("Готово.")


if __name__ == "__main__":
    main()
