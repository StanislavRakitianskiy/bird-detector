"""Перевірка ваг на val: метрики кількох моделей на кількох розмірах картинки в одній таблиці.

Навіщо: щоб порівнювати моделі в однакових умовах — та сама val, той самий imgsz.
Наприклад, готову yolo26n.pt (бейзлайн) з дотренованою, або v4 з v5 на одному розмірі.

Готова yolo26n.pt знає 80 класів COCO, і bird у неї має id 14, а в нашому датасеті — 0.
Ultralytics сам класи не зіставляє, тому для такої моделі скрипт робить копію val
у runs/eval_data/ з тими самими фото, але з розміткою під id класу bird цієї моделі.
В усіх випадках рахуються метрики лише класу bird.

Приклад: python evaluate.py --weights yolo26n.pt weights/best.pt --imgsz 640 960
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.utils import YAML

# Беремо готові функції з інших скриптів проєкту, щоб не дублювати код
from predict_video import TARGET_CLASS, pick_imgsz
from train import PROJECT_ROOT, RUNS_DIR, resolve_data_yaml

log = logging.getLogger("evaluate")

EVAL_DATA_DIR = RUNS_DIR / "eval_data"


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
        description="Метрики ваг на val (лише клас bird): кілька моделей і розмірів картинки в одній таблиці.",
        add_help=False,  # свій -h нижче, щоб довідка була українською
    )
    parser.add_argument("-h", "--help", action="help", help="показати цю довідку і вийти")
    parser.add_argument("--weights", type=Path, nargs="+", default=[Path("weights/best.pt")],
                        help="одні або кілька ваг через пробіл (шлях від кореня проєкту); yolo26n.pt, "
                             "якщо його немає, Ultralytics завантажить сам; за замовчуванням weights/best.pt")
    parser.add_argument("--imgsz", type=int, nargs="+", default=None,
                        help="один або кілька розмірів картинки через пробіл; за замовчуванням — той, "
                             "на якому навчали кожні ваги")
    parser.add_argument("--data", type=Path, default=Path("data/bird.yaml"),
                        help="конфіг датасету (відносно кореня проєкту), за замовчуванням data/bird.yaml")
    # Від batch цифри залежать у третьому знаку: фото в батчі доповнюються до спільного розміру.
    # З batch 4 метрики weights/best.pt точно збігаються з weights/metrics.json (там теж був batch 4).
    parser.add_argument("--batch", type=int, default=4,
                        help="розмір батча; метрики від нього трохи залежать, тому для порівняння моделей "
                             "беріть однаковий; за замовчуванням 4 (як при перевірці weights/best.pt)")
    parser.add_argument("--workers", type=int, default=2,
                        help="кількість процесів, що читають дані, за замовчуванням 2")
    parser.add_argument("--device", default=None,
                        help="пристрій: 0 (перший GPU), mps або cpu; за замовчуванням вибирається автоматично")
    args = parser.parse_args()
    if args.imgsz and min(args.imgsz) < 32:
        parser.error("--imgsz має бути не менше 32")
    if args.batch < 1:
        parser.error("--batch має бути не менше 1")
    if args.workers < 0:
        parser.error("--workers не може бути від'ємним")
    return args


# --------------------------------------------------------------------------- модель і датасет

def load_model(weights: Path) -> YOLO:
    """Завантажує ваги за шляхом від кореня проєкту; якщо файлу немає — віддає назву Ultralytics.

    Ultralytics сам завантажує з інтернету лише свої готові ваги (напр. yolo26n.pt),
    тому для шляху з папкою, якого немає на диску, одразу зупиняємось з помилкою.
    """
    local = PROJECT_ROOT / weights  # якщо шлях абсолютний, Path поверне саме його
    if local.is_file():
        return YOLO(str(local))
    if weights.parent != Path("."):
        raise SystemExit(f"Файл ваг не знайдено: {local}")
    return YOLO(str(weights))


def find_bird_id(model: YOLO, weights: Path) -> int:
    """Шукає id класу bird у моделі за назвою (у нашій моделі це 0, у готовій yolo26n.pt — 14)."""
    for class_id, name in model.names.items():
        if name == TARGET_CLASS:
            return class_id
    raise SystemExit(f"У моделі {weights} немає класу '{TARGET_CLASS}'")


def data_for_model(data_path: Path, names: dict[int, str], bird_id: int) -> Path:
    """Повертає конфіг датасету, класи якого збігаються з класами моделі.

    Для нашої моделі (один клас bird з id 0) — сам data_path.
    Для моделі з іншими класами робить копію val у runs/eval_data/, де id 0 у розмітці
    замінено на bird_id, а в yaml записано всі класи моделі (інакше Ultralytics вважає
    рядки з id 14 помилковими й відкидає).
    """
    if names == {0: TARGET_CLASS}:
        return data_path
    data = YAML.load(data_path)
    src = Path(data["path"])
    images_rel = Path(data["val"])  # images/val
    labels_rel = Path("labels", *images_rel.parts[1:])  # Ultralytics шукає розмітку, замінивши images на labels
    dest = EVAL_DATA_DIR / f"bird_id_{bird_id}"
    shutil.rmtree(dest, ignore_errors=True)  # прибираємо копію і кеш розмітки з минулого запуску
    shutil.copytree(src / images_rel, dest / images_rel)
    (dest / labels_rel).mkdir(parents=True)
    for label in (src / labels_rel).glob("*.txt"):
        lines = label.read_text(encoding="utf-8").splitlines()
        text = "".join(f"{bird_id} {line.split(maxsplit=1)[1]}\n" for line in lines)
        (dest / labels_rel / label.name).write_text(text, encoding="utf-8")
    yaml_path = dest / "data.yaml"
    # train теж вказуємо на val: Ultralytics вимагає ключ train, але для перевірки він не потрібен
    YAML.save(yaml_path, {"path": dest.as_posix(), "train": images_rel.as_posix(),
                          "val": images_rel.as_posix(), "names": names})
    log.info("Модель має %d класів, bird — id %d: копія val з такою розміткою — %s", len(names), bird_id, dest)
    return yaml_path


# --------------------------------------------------------------------------- перевірка

def validate(model: YOLO, data: Path, imgsz: int, bird_id: int, args: argparse.Namespace) -> dict[str, float]:
    """Перевіряє модель на val і повертає precision, recall, mAP50, mAP50-95 класу bird."""
    box = model.val(
        data=str(data),
        split="val",
        imgsz=imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        classes=[bird_id],  # рамки інших класів (для yolo26n.pt) не рахуємо
        plots=False,
        verbose=False,
        project=str(RUNS_DIR),
        name="evaluate",
        exist_ok=True,  # результати перевірок нам не потрібні — не плодимо папки evaluate2, evaluate3...
    ).box
    return {
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
    }


def evaluate_weights(weights: Path, data_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Перевіряє одні ваги на всіх потрібних розмірах. Повертає рядки таблиці."""
    model = load_model(weights)
    bird_id = find_bird_id(model, weights)
    data = data_for_model(data_path, model.names, bird_id)
    rows = []
    for imgsz in args.imgsz or [pick_imgsz(None, model)]:
        log.info("Перевіряю %s на imgsz %d...", weights.as_posix(), imgsz)
        rows.append({"weights": weights.as_posix(), "imgsz": imgsz, **validate(model, data, imgsz, bird_id, args)})
    return rows


def log_table(rows: list[dict[str, Any]]) -> None:
    """Виводить у лог підсумкову таблицю метрик."""
    width = max(len(row["weights"]) for row in rows)
    log.info("Метрики на val (лише клас %s):", TARGET_CLASS)
    log.info("%-*s  %5s  %9s  %6s  %6s  %8s", width, "Ваги", "imgsz", "Precision", "Recall", "mAP50", "mAP50-95")
    for row in rows:
        log.info("%-*s  %5d  %9.4f  %6.4f  %6.4f  %8.4f", width, row["weights"], row["imgsz"],
                 row["precision"], row["recall"], row["mAP50"], row["mAP50-95"])


# --------------------------------------------------------------------------- головна функція

def main() -> None:
    """Точка входу: перевірка датасету -> метрики кожних ваг на кожному розмірі -> таблиця."""
    setup_logging()
    args = parse_args()
    data_path = resolve_data_yaml(args.data)
    rows = []
    for weights in args.weights:
        rows.extend(evaluate_weights(weights, data_path, args))
    log_table(rows)
    log.info("Готово.")


if __name__ == "__main__":
    main()
