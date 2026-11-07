"""Навчання (fine-tune) YOLO26n для пошуку птахів.

Що робить скрипт:
1. Робить копію data/bird.yaml з АБСОЛЮТНИМ шляхом до датасету (див. make_absolute_yaml).
2. Донавчає модель (за замовчуванням стартові ваги yolo26n.pt) на датасеті з prepare_dataset.py.
3. Копіює найкращі ваги в weights/best.pt.
4. Заново перевіряє ці ваги на val і зберігає метрики в weights/metrics.json.
"""

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.utils import YAML

log = logging.getLogger("train")

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"


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
    parser.add_argument("--weights", default="yolo26n.pt",
                        help="стартові ваги, за замовчуванням yolo26n.pt (з кореня проєкту)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="максимальна кількість епох, за замовчуванням 50")
    parser.add_argument("--patience", type=int, default=10,
                        help="зупинити навчання, якщо якість не росте стільки епох поспіль, за замовчуванням 10")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="розмір картинки при навчанні, за замовчуванням 640")
    parser.add_argument("--batch", type=int, default=8,
                        help="розмір батча; якщо не вистачає пам'яті GPU — зменшуйте, за замовчуванням 8")
    parser.add_argument("--workers", type=int, default=2,
                        help="кількість процесів, що читають дані, за замовчуванням 2")
    parser.add_argument("--device", default=None,
                        help="пристрій: 0 (перший GPU) або cpu; за замовчуванням Ultralytics бере GPU, якщо він є")
    parser.add_argument("--seed", type=int, default=42,
                        help="зерно генератора випадкових чисел, за замовчуванням 42")
    parser.add_argument("--name", default="bird",
                        help="назва запуску: результати лягають у runs/detect/<name>, за замовчуванням bird")
    parser.add_argument("--out-dir", type=Path, default=Path("weights"),
                        help="куди покласти best.pt і metrics.json (відносно кореня проєкту), "
                             "за замовчуванням weights")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.imgsz, args.batch) < 1 or args.workers < 0:
        parser.error("--epochs, --patience, --imgsz і --batch мають бути не менше 1, а --workers — не менше 0")
    return args


def make_absolute_yaml(data_yaml: Path) -> Path:
    """Робить копію конфіга датасету з абсолютним `path` і повертає шлях до цієї копії.

    Ultralytics рахує відносний `path` не від yaml-файлу, а від поточної папки (і від свого datasets_dir),
    тому у data/bird.yaml лежить шлях відносно кореня проєкту, а тут ми робимо його абсолютним.
    Копія кладеться в runs/ (поза git), тож у репозиторії немає шляхів з нашого комп'ютера.
    """
    data = YAML.load(data_yaml)
    dataset_dir = (PROJECT_ROOT / data.get("path", "")).resolve()  # якщо path уже абсолютний — лишиться він
    for key in ("train", "val"):
        folder = dataset_dir / data[key]
        if not folder.is_dir():
            raise SystemExit(f"Не знайдено {folder}. Спершу підготуйте датасет: python prepare_dataset.py")
    data["path"] = dataset_dir.as_posix()
    dest = RUNS_DIR / f"{data_yaml.stem}_abs.yaml"
    YAML.save(dest, data)
    log.info("Датасет: %s (конфіг з абсолютним шляхом: %s)", dataset_dir, dest)
    return dest


def train_model(args: argparse.Namespace, data_path: Path) -> tuple[Path, int]:
    """Запускає навчання. Повертає папку запуску (runs/detect/<name>) і кількість пройдених епох."""
    local_weights = PROJECT_ROOT / args.weights
    weights = str(local_weights) if local_weights.exists() else args.weights  # інакше Ultralytics сам завантажить
    model = YOLO(weights)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        project=str(RUNS_DIR / "detect"),
        name=args.name,
    )
    # Якщо спрацювала рання зупинка (--patience), епох буде менше, ніж просили
    return Path(model.trainer.save_dir), model.trainer.epoch + 1


def copy_best_weights(run_dir: Path, out_dir: Path) -> Path:
    """Копіює найкращі ваги запуску (runs/detect/<name>/weights/best.pt) у out_dir/best.pt."""
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Навчання не створило {best} — дивіться повідомлення про помилку вище")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "best.pt"
    shutil.copy2(best, dest)
    log.info("Найкращі ваги збережено: %s", dest)
    return dest


def evaluate(best_path: Path, data_path: Path, args: argparse.Namespace, run_dir: Path) -> dict[str, float]:
    """Перевіряє збережені ваги на val і повертає precision, recall, F1, mAP50, mAP50-95."""
    model = YOLO(str(best_path))
    result = model.val(
        data=str(data_path),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(run_dir.parent),
        name=f"{run_dir.name}_val",
    ).results_dict
    precision = float(result["metrics/precision(B)"])
    recall = float(result["metrics/recall(B)"])
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mAP50": round(float(result["metrics/mAP50(B)"]), 4),
        "mAP50-95": round(float(result["metrics/mAP50-95(B)"]), 4),
    }


def load_dataset_stats(data_path: Path) -> dict[str, Any] | None:
    """Читає stats.json, який лежить поруч з датасетом (створюється prepare_dataset.py)."""
    stats_path = Path(YAML.load(data_path)["path"]) / "stats.json"
    if not stats_path.exists():
        return None
    return json.loads(stats_path.read_text(encoding="utf-8"))


def write_metrics(
    out_dir: Path, metrics: dict[str, float], args: argparse.Namespace, epochs_done: int, data_path: Path
) -> Path:
    """Зберігає out_dir/metrics.json: метрики на val, параметри навчання і статистику датасету."""
    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "start_weights": args.weights,
        "evaluated_on": "val",
        "metrics": metrics,
        "train_params": {
            "epochs_max": args.epochs,
            "epochs_done": epochs_done,
            "patience": args.patience,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "seed": args.seed,
        },
        "dataset": load_dataset_stats(data_path),
    }
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("Метрики збережено: %s", path)
    return path


def main() -> None:
    """Точка входу: yaml -> навчання -> best.pt -> метрики на val -> metrics.json."""
    setup_logging()
    args = parse_args()
    out_dir = PROJECT_ROOT / args.out_dir  # якщо --out-dir абсолютний, Path поверне саме його

    data_path = make_absolute_yaml(PROJECT_ROOT / args.data)
    run_dir, epochs_done = train_model(args, data_path)
    best_path = copy_best_weights(run_dir, out_dir)
    metrics = evaluate(best_path, data_path, args, run_dir)
    write_metrics(out_dir, metrics, args, epochs_done, data_path)

    log.info("Готово. Епох пройдено: %d. Метрики на val: %s", epochs_done, metrics)


if __name__ == "__main__":
    main()
