"""Підготовка датасету птахів у форматі YOLO з COCO 2017.

Що робить скрипт:
1. Качає архів розмітки COCO (один раз, кешує в datasets/_downloads/).
2. Знаходить категорію "bird", відсікає зайві рамки (iscrowd, надто малі).
3. Відбирає фото: train — випадкова вибірка, val — усі фото з птахами + трохи фото без птахів.
4. Качає фото по одному (багатопотоково) і пише YOLO-розмітку (.txt).
5. Генерує data/bird.yaml, datasets/birds/stats.json і (за бажанням) картинки для перевірки.
"""

import argparse
import json
import logging
import random
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import requests
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

log = logging.getLogger("prepare_dataset")

PROJECT_ROOT = Path(__file__).resolve().parent
DOWNLOADS_DIR = PROJECT_ROOT / "datasets" / "_downloads"
YAML_PATH = PROJECT_ROOT / "data" / "bird.yaml"

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
CLASS_NAME = "bird"
SPLITS = ("train", "val")
MIN_BOX_SIDE = 2  # рамки, де ширина або висота менша за стільки пікселів, відкидаємо
SMALL_BOX_AREA = 32 * 32  # «мала» рамка за визначенням COCO: площа менша за 32x32 px
DOWNLOAD_ATTEMPTS = 3
REQUEST_TIMEOUT = 30  # секунд на з'єднання / очікування даних


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
        description="Готує датасет птахів у форматі YOLO з COCO 2017.",
        add_help=False,  # свій -h нижче, щоб довідка була українською
    )
    parser.add_argument("-h", "--help", action="help", help="показати цю довідку і вийти")
    parser.add_argument("--out", type=Path, default=Path("datasets/birds"),
                        help="папка датасету (відносно кореня проєкту), за замовчуванням datasets/birds")
    parser.add_argument("--train-size", type=int, default=1500,
                        help="скільки фото з птахами взяти в train, за замовчуванням 1500")
    parser.add_argument("--neg-ratio", type=float, default=0.1,
                        help="частка фото БЕЗ птахів відносно кількості фото з птахами "
                             "(в train і в val окремо), за замовчуванням 0.1")
    parser.add_argument("--seed", type=int, default=42,
                        help="зерно генератора випадкових чисел, за замовчуванням 42")
    parser.add_argument("--workers", type=int, default=8,
                        help="кількість потоків для завантаження фото, за замовчуванням 8")
    parser.add_argument("--preview", type=int, default=0, metavar="N",
                        help="зберегти N фото з намальованими рамками в <out>/preview/ "
                             "(рамки беруться з YOLO-файлів), за замовчуванням 0 — не робити")
    args = parser.parse_args()
    if args.train_size < 1:
        parser.error("--train-size має бути не менше 1")
    if args.neg_ratio < 0:
        parser.error("--neg-ratio не може бути від'ємним")
    if args.workers < 1:
        parser.error("--workers має бути не менше 1")
    if args.preview < 0:
        parser.error("--preview не може бути від'ємним")
    return args


# --------------------------------------------------------------------------- конвертація рамок

def coco_bbox_to_yolo(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float] | None:
    """Перетворює рамку COCO [x, y, w, h] (пікселі) у YOLO (xc, yc, w, h) у частках від 0 до 1.

    Рамку спершу обрізаємо до меж фото. Якщо після обрізання нічого не лишилось
    (рамка повністю поза фото), повертаємо None.
    """
    x, y, w, h = bbox
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(img_w), x + w)
    y2 = min(float(img_h), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    return xc, yc, (x2 - x1) / img_w, (y2 - y1) / img_h


def yolo_to_xyxy(
    box: tuple[float, float, float, float], img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Зворотна конвертація: YOLO (xc, yc, w, h) у частках -> кутові координати (x1, y1, x2, y2) у пікселях."""
    xc, yc, w, h = box
    x1 = round((xc - w / 2) * img_w)
    y1 = round((yc - h / 2) * img_h)
    x2 = round((xc + w / 2) * img_w)
    y2 = round((yc + h / 2) * img_h)
    return x1, y1, x2, y2


def format_yolo_line(box: tuple[float, float, float, float]) -> str:
    """Формує рядок YOLO-файлу: клас 0 (bird) і чотири числа з 6 знаками після коми."""
    xc, yc, w, h = box
    return f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


# --------------------------------------------------------------------------- розмітка COCO

def download_file(url: str, dest: Path) -> None:
    """Качає великий файл потоком з прогрес-баром. Спершу пише в .part, щоб обрив не лишив битий файл."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def ensure_annotations() -> Path:
    """Повертає шлях до архіву розмітки; качає його, лише якщо його ще немає (або він битий)."""
    zip_path = DOWNLOADS_DIR / Path(ANNOTATIONS_URL).name
    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        log.info("Архів розмітки вже є, не качаю: %s", zip_path)
        return zip_path
    if zip_path.exists():
        log.warning("Файл %s пошкоджений, качаю наново", zip_path)
    log.info("Качаю архів розмітки COCO (~250 МБ): %s", ANNOTATIONS_URL)
    download_file(ANNOTATIONS_URL, zip_path)
    return zip_path


def load_coco_json(zip_path: Path, split: str) -> dict[str, Any]:
    """Читає instances_<split>2017.json прямо з zip-архіву, без розпакування на диск."""
    member = f"annotations/instances_{split}2017.json"
    log.info("Читаю %s (файл великий, це може зайняти до хвилини і кілька ГБ пам'яті)...", member)
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as f:
        return json.load(f)


def find_category_id(categories: list[dict[str, Any]], name: str) -> int:
    """Шукає id категорії за назвою (число не хардкодимо)."""
    for category in categories:
        if category["name"] == name:
            return category["id"]
    raise ValueError(f"Категорію '{name}' не знайдено в розмітці COCO")


def collect_bird_boxes(
    annotations: list[dict[str, Any]], bird_id: int
) -> tuple[dict[int, list[list[float]]], set[int]]:
    """Збирає рамки птахів по фото.

    Повертає:
      - {image_id: [bbox, ...]} — лише «хороші» рамки (не iscrowd, сторони >= MIN_BOX_SIDE);
      - множину image_id усіх фото, де є ХОЧ ЯКІСЬ птахи (навіть відкинуті рамки) —
        щоб такі фото потім випадково не потрапили в «фото без птахів».
    """
    boxes: dict[int, list[list[float]]] = {}
    has_any_bird: set[int] = set()
    for ann in annotations:
        if ann["category_id"] != bird_id:
            continue
        has_any_bird.add(ann["image_id"])
        _, _, w, h = ann["bbox"]
        if ann["iscrowd"] == 1 or w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
            continue
        boxes.setdefault(ann["image_id"], []).append(ann["bbox"])
    return boxes, has_any_bird


def pick_images(
    zip_path: Path, split: str, size: int | None, neg_ratio: float, rng: random.Random
) -> list[dict[str, Any]]:
    """Відбирає фото одної частини (train або val).

    size=None — взяти ВСІ фото з птахами; інакше — випадкові `size` штук.
    Додатково беремо round(кількість фото з птахами * neg_ratio) фото без птахів.
    Кожне фото — словник: file_name, url, width, height, bboxes (рамки COCO; для «негативів» порожній список).
    """
    coco = load_coco_json(zip_path, split)
    bird_id = find_category_id(coco["categories"], CLASS_NAME)
    images = {img["id"]: img for img in coco["images"]}
    boxes, has_any_bird = collect_bird_boxes(coco["annotations"], bird_id)
    del coco  # велику розмітку більше не тримаємо в пам'яті

    # Сортуємо, щоб результат залежав лише від seed, а не від порядку в json
    positives = sorted(boxes)
    negatives = sorted(set(images) - has_any_bird)
    log.info("%s2017: категорія '%s' має id=%d; фото з птахами: %d, фото без птахів: %d",
             split, CLASS_NAME, bird_id, len(positives), len(negatives))

    if size is not None:
        if size > len(positives):
            log.warning("%s: просили %d фото з птахами, а їх лише %d — беру всі", split, size, len(positives))
        else:
            positives = sorted(rng.sample(positives, size))
    n_neg = min(round(len(positives) * neg_ratio), len(negatives))
    negatives = sorted(rng.sample(negatives, n_neg))

    picked = []
    for image_id in positives + negatives:
        img = images[image_id]
        picked.append({
            "file_name": img["file_name"],
            "url": img["coco_url"],
            "width": img["width"],
            "height": img["height"],
            "bboxes": boxes.get(image_id, []),
        })
    log.info("%s: відібрано %d фото з птахами + %d без птахів", split, len(positives), len(negatives))
    return picked


# --------------------------------------------------------------------------- фото і розмітка на диску

def download_image(url: str, dest: Path) -> bool:
    """Качає одне фото (до DOWNLOAD_ATTEMPTS спроб). Якщо файл уже є — нічого не робить. True — успіх."""
    if dest.exists():
        return True
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            tmp.write_bytes(resp.content)
            tmp.replace(dest)
            return True
        except (requests.RequestException, OSError) as err:
            tmp.unlink(missing_ok=True)
            log.warning("Спроба %d/%d невдала для %s: %s", attempt, DOWNLOAD_ATTEMPTS, url, err)
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(attempt)  # пауза 1 с, потім 2 с
    return False


def download_images(items: list[dict[str, Any]], images_dir: Path, workers: int, split: str) -> list[dict[str, Any]]:
    """Качає фото багатопотоково. Повертає лише ті елементи, чиї фото вдалося завантажити."""
    images_dir.mkdir(parents=True, exist_ok=True)
    done = []
    with logging_redirect_tqdm(), ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_image, item["url"], images_dir / item["file_name"]): item
            for item in items
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"фото {split}", unit="фото"):
            if future.result():
                done.append(futures[future])
    failed = len(items) - len(done)
    if failed:
        log.warning("%s: не вдалося завантажити %d фото — їх пропущено", split, failed)
    return sorted(done, key=lambda item: item["file_name"])


def convert_item(item: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    """Переводить усі рамки одного фото з COCO у YOLO (порожні після обрізання — пропускає)."""
    boxes = (coco_bbox_to_yolo(b, item["width"], item["height"]) for b in item["bboxes"])
    return [box for box in boxes if box is not None]


def write_label(path: Path, boxes: list[tuple[float, float, float, float]]) -> None:
    """Пише YOLO-файл розмітки. Для фото без птахів файл порожній (так і треба для YOLO)."""
    text = "".join(format_yolo_line(box) + "\n" for box in boxes)
    path.write_text(text, encoding="utf-8")


def remove_stale_files(folder: Path, keep: set[str]) -> int:
    """Видаляє в папці файли (.jpg/.txt/.part) зі старих запусків, яких немає в `keep`.

    Без цього після запуску з іншим --train-size у датасеті лишилися б зайві старі фото.
    """
    removed = 0
    if not folder.is_dir():
        return removed
    for path in folder.iterdir():
        if path.is_file() and path.suffix in {".jpg", ".txt", ".part"} and path.name not in keep:
            path.unlink()
            removed += 1
    return removed


def build_split(
    out: Path, split: str, items: list[dict[str, Any]], workers: int
) -> list[dict[str, Any]]:
    """Качає фото частини датасету й пише її розмітку. Повертає готові елементи (з ключем "yolo")."""
    images_dir = out / "images" / split
    labels_dir = out / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)

    ready = []
    for item in download_images(items, images_dir, workers, split):
        item["yolo"] = convert_item(item)
        if item["bboxes"] and not item["yolo"]:
            log.warning("%s: усі рамки фото %s поза його межами — фото пропущено", split, item["file_name"])
            continue
        write_label(labels_dir / (Path(item["file_name"]).stem + ".txt"), item["yolo"])
        ready.append(item)

    removed = remove_stale_files(images_dir, {item["file_name"] for item in ready})
    removed += remove_stale_files(labels_dir, {Path(item["file_name"]).stem + ".txt" for item in ready})
    if removed:
        log.info("%s: видалено %d зайвих файлів зі старих запусків", split, removed)
    return ready


# --------------------------------------------------------------------------- статистика, yaml, перегляд

def compute_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Рахує статистику частини датасету за готовими YOLO-рамками."""
    bird_photos = [item for item in items if item["yolo"]]
    total_boxes = sum(len(item["yolo"]) for item in items)
    small_boxes = 0
    for item in items:
        for _, _, w, h in item["yolo"]:
            if (w * item["width"]) * (h * item["height"]) < SMALL_BOX_AREA:
                small_boxes += 1
    return {
        "photos": len(items),
        "photos_with_birds": len(bird_photos),
        "photos_without_birds": len(items) - len(bird_photos),
        "boxes": total_boxes,
        "avg_birds_per_bird_photo": round(total_boxes / len(bird_photos), 2) if bird_photos else 0.0,
        "small_boxes": small_boxes,
        "small_boxes_share": round(small_boxes / total_boxes, 4) if total_boxes else 0.0,
    }


def log_stats(split: str, stats: dict[str, Any]) -> None:
    """Виводить статистику частини датасету в лог."""
    log.info(
        "%s: фото %d (з птахами %d, без птахів %d), рамок %d, "
        "у середньому %.2f птаха на фото з птахами, малих рамок (<32x32 px): %.1f%%",
        split, stats["photos"], stats["photos_with_birds"], stats["photos_without_birds"],
        stats["boxes"], stats["avg_birds_per_bird_photo"], stats["small_boxes_share"] * 100,
    )


def write_stats(out: Path, args: argparse.Namespace, stats: dict[str, Any]) -> None:
    """Зберігає stats.json разом із параметрами, з якими готували датасет."""
    report = {
        "params": {"train_size": args.train_size, "neg_ratio": args.neg_ratio, "seed": args.seed},
        **stats,
    }
    path = out / "stats.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("Статистику збережено: %s", path)


def write_data_yaml(out: Path) -> None:
    """Генерує data/bird.yaml. Шлях у ньому відносний до кореня проєкту (без нашого абсолютного шляху)."""
    try:
        dataset_path = out.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        dataset_path = out.resolve().as_posix()
        log.warning("Датасет лежить поза проєктом, у %s потрапить абсолютний шлях: %s", YAML_PATH.name, dataset_path)
    text = (
        "# Конфіг датасету для Ultralytics. Генерується скриптом prepare_dataset.py.\n"
        "# path — відносно кореня проєкту. Ultralytics сам цього не розуміє (рахує від свого datasets_dir),\n"
        "# тому train.py перед запуском підставляє сюди абсолютний шлях.\n"
        f"path: {dataset_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )
    YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    YAML_PATH.write_text(text, encoding="utf-8")
    log.info("Конфіг датасету збережено: %s", YAML_PATH)


def draw_preview(image_path: Path, label_path: Path, dest: Path) -> bool:
    """Малює на фото рамки, прочитані З YOLO-ФАЙЛУ (зворотна конвертація), і зберігає картинку."""
    img = cv2.imread(str(image_path))
    if img is None:
        log.warning("Не вдалося прочитати фото %s", image_path)
        return False
    img_h, img_w = img.shape[:2]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        _, xc, yc, w, h = line.split()
        x1, y1, x2, y2 = yolo_to_xyxy((float(xc), float(yc), float(w), float(h)), img_w, img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, CLASS_NAME, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return bool(cv2.imwrite(str(dest), img))


def save_previews(out: Path, count: int, rng: random.Random) -> None:
    """Зберігає `count` випадкових фото з птахами з намальованими рамками в <out>/preview/."""
    preview_dir = out / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_files(preview_dir, set())  # прибираємо картинки з минулого запуску

    candidates = [
        (split, label.stem)
        for split in SPLITS
        for label in sorted((out / "labels" / split).glob("*.txt"))
        if label.stat().st_size > 0
    ]
    chosen = rng.sample(candidates, min(count, len(candidates)))
    saved = 0
    for split, stem in chosen:
        saved += draw_preview(
            out / "images" / split / f"{stem}.jpg",
            out / "labels" / split / f"{stem}.txt",
            preview_dir / f"{split}_{stem}.jpg",
        )
    log.info("Збережено %d картинок для перевірки рамок: %s", saved, preview_dir)


# --------------------------------------------------------------------------- головна функція

def main() -> None:
    """Точка входу: розмітка -> відбір -> завантаження -> розмітка YOLO -> статистика."""
    setup_logging()
    args = parse_args()
    out = PROJECT_ROOT / args.out  # якщо --out абсолютний, Path поверне саме його
    log.info("Папка датасету: %s", out)

    zip_path = ensure_annotations()
    rng = random.Random(args.seed)
    # val відбираємо ПЕРШИМ: тоді він не залежить від --train-size,
    # і метрики з різними розмірами train порівнюються на одній і тій самій val.
    picked = {"val": pick_images(zip_path, "val", None, args.neg_ratio, rng)}
    picked["train"] = pick_images(zip_path, "train", args.train_size, args.neg_ratio, rng)

    stats = {}
    for split in SPLITS:
        ready = build_split(out, split, picked[split], args.workers)
        stats[split] = compute_stats(ready)
        log_stats(split, stats[split])

    write_stats(out, args, stats)
    write_data_yaml(out)
    if args.preview:
        save_previews(out, args.preview, rng)
    log.info("Готово.")


if __name__ == "__main__":
    main()
