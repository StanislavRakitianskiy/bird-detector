"""Детекція птахів на відео навченою моделлю YOLO: вхідне відео -> відео з рамками.

Що робить скрипт:
1. Відкриває вхідне відео і бере з нього FPS, ширину й висоту.
2. Запускає модель на КОЖНОМУ кадрі (без пропусків) з тим imgsz, на якому навчали ваги,
   і прибирає рамки-дублікати «рамка всередині рамки»
   (напр. тіло птаха всередині рамки «птах разом з довгим хвостом»).
3. Зменшує кадр до --max-side (інакше вихідне відео важить десятки МБ), малює рамки,
   назву класу і confidence, а в кутку кадру — рядок стану.
4. Записує кадри у вихідне відео (H.264, якщо доступний, інакше mp4v)
   і зберігає кілька кадрів з птахами як jpg — для README.

Шляхи (--input, --output, --weights) рахуються від кореня проєкту, як і в інших скриптах,
тому скрипт можна запускати з будь-якої папки.

Приклад: python predict_video.py --input examples/input.mp4
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

log = logging.getLogger("predict_video")

PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_CLASS = "bird"  # показуємо лише цей клас (важливо для готової yolo26n.pt, де 80 класів)
DEFAULT_FPS = 25.0  # якщо відео не повідомляє свій FPS
DEFAULT_IMGSZ = 640  # якщо у вагах не записано, на якому розмірі їх навчали
CODECS = ("avc1", "mp4v")  # avc1 = H.264 (відкривають браузери й плеєри), mp4v — запасний (браузери не відкривають)
WINDOW_NAME = "bird detector"  # латиницею: вікна OpenCV на Windows не показують кирилицю
FONT = cv2.FONT_HERSHEY_SIMPLEX  # вбудований шрифт OpenCV — лише латиниця, тому підписи англійською
# Кольори в OpenCV задаються в порядку BGR (синій, зелений, червоний), а не RGB
BOX_COLOR = (0, 200, 255)  # помаранчевий
LABEL_TEXT_COLOR = (0, 0, 0)  # чорний
STATUS_BG_COLOR = (0, 0, 0)  # чорний
STATUS_TEXT_COLOR = (255, 255, 255)  # білий


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
        description="Детекція птахів на відео: модель обробляє кожен кадр, результат — відео з рамками.",
        add_help=False,  # свій -h нижче, щоб довідка була українською
    )
    parser.add_argument("-h", "--help", action="help", help="показати цю довідку і вийти")
    parser.add_argument("--input", type=Path, default=None,
                        help="вхідне відео (ОБОВ'ЯЗКОВО; відносно кореня проєкту), напр. examples/input.mp4")
    parser.add_argument("--output", type=Path, default=Path("examples/output.mp4"),
                        help="куди зберегти відео з рамками (.mp4, відносно кореня проєкту), "
                             "за замовчуванням examples/output.mp4")
    parser.add_argument("--weights", type=Path, default=Path("weights/best.pt"),
                        help="ваги моделі (відносно кореня проєкту), за замовчуванням weights/best.pt; "
                             "для порівняння можна yolo26n.pt — якщо файлу немає, Ultralytics завантажить сам")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="мінімальна впевненість (0..1], слабші детекції відкидаються, за замовчуванням 0.25")
    parser.add_argument("--nested", type=float, default=0.8,
                        help="прибрати рамку як дублікат, якщо така частка її площі лежить усередині "
                             "впевненішої рамки (напр. тіло птаха всередині рамки «птах з хвостом»); "
                             "0 — не прибирати, за замовчуванням 0.8")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="розмір, до якого модель зменшує кадр перед обробкою; за замовчуванням — той, "
                             "на якому навчали ваги (для weights/best.pt це 960), а якщо він невідомий — 640")
    parser.add_argument("--max-side", type=int, default=1280,
                        help="найбільша сторона вихідного відео в пікселях: більші кадри зменшуються, щоб файл "
                             "був меншим (модель однаково бачить повний кадр); 0 — не зменшувати, "
                             "за замовчуванням 1280")
    parser.add_argument("--device", default=None,
                        help="пристрій: 0 (перший GPU) або cpu; за замовчуванням вибирається автоматично")
    parser.add_argument("--show", action="store_true",
                        help="показувати кадри у вікні під час обробки (q або Esc — зупинити)")
    parser.add_argument("--save-frames", type=int, default=3,
                        help="скільки кадрів з птахами зберегти як jpg поруч з --output "
                             "(0 — не зберігати), за замовчуванням 3")
    args = parser.parse_args()
    # Не required=True: тоді argparse сам пише помилку англійською
    if args.input is None:
        parser.error("не вказано вхідне відео: додайте --input шлях/до/відео.mp4")
    if not 0 < args.conf <= 1:
        parser.error("--conf має бути в межах (0, 1]")
    if not 0 <= args.nested <= 1:
        parser.error("--nested має бути в межах [0, 1]")
    if args.imgsz is not None and args.imgsz < 32:
        parser.error("--imgsz має бути не менше 32")
    if args.max_side != 0 and args.max_side < 32:
        parser.error("--max-side має бути 0 (не зменшувати) або не менше 32")
    if args.save_frames < 0:
        parser.error("--save-frames не може бути від'ємним")
    return args


# --------------------------------------------------------------------------- перевірки і модель

def check_paths(args: argparse.Namespace) -> None:
    """Перевіряє вхідне відео ще до завантаження моделі. Шляхи вже мають бути від кореня проєкту."""
    if not args.input.is_file():
        raise SystemExit(f"Вхідне відео не знайдено: {args.input}")
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("--output не може збігатися з --input: вхідне відео буде перезаписане")


def load_weights(weights: Path) -> YOLO:
    """Завантажує ваги за шляхом від кореня проєкту (абсолютний шлях теж підходить).

    Якщо файлу немає, Ultralytics сам завантажує з інтернету свої готові ваги (напр. yolo26n.pt) —
    саме за цим шляхом, тобто в корінь проєкту. Інших файлів він не знає і кидає FileNotFoundError.
    """
    path = PROJECT_ROOT / weights  # якщо weights абсолютний, Path поверне саме його
    try:
        return YOLO(str(path))
    except FileNotFoundError:
        raise SystemExit(f"Файл ваг не знайдено: {path}. Спочатку навчіть модель (python train.py) "
                         "або вкажіть інші ваги, напр. --weights yolo26n.pt") from None


def load_model(weights: Path) -> tuple[YOLO, list[int]]:
    """Завантажує модель (один раз) і знаходить id класу bird за назвою в model.names."""
    model = load_weights(weights)
    class_ids = [cls_id for cls_id, name in model.names.items() if name == TARGET_CLASS]
    if not class_ids:
        raise SystemExit(f"У моделі {weights} немає класу '{TARGET_CLASS}'. Її класи: {list(model.names.values())}")
    log.info("Модель: %s (класів: %d), показую лише '%s' (id %s)",
             weights, len(model.names), TARGET_CLASS, class_ids)
    return model, class_ids


def imgsz_from_weights(model: YOLO) -> int | None:
    """Розмір картинки, на якому навчали ваги (Ultralytics зберігає його в .pt); None — якщо невідомо."""
    checkpoint = getattr(model, "ckpt", None) or {}
    imgsz = (checkpoint.get("train_args") or {}).get("imgsz")
    return imgsz if isinstance(imgsz, int) else None


def pick_imgsz(requested: int | None, model: YOLO) -> int:
    """Розмір обробки кадру: --imgsz, якщо його вказали, інакше розмір навчання з ваг, інакше DEFAULT_IMGSZ.

    Навіщо: модель найкраще працює на тому розмірі, на якому її навчали. weights/best.pt навчено на 960,
    і на 640 вона знаходить помітно менше дрібних птахів.
    """
    if requested is not None:
        return requested
    trained = imgsz_from_weights(model)
    if trained is None:
        log.warning("У вагах не записано розмір навчання — використовую imgsz %d", DEFAULT_IMGSZ)
        return DEFAULT_IMGSZ
    log.info("imgsz %d — як при навчанні цих ваг (змінити можна через --imgsz)", trained)
    return trained


# --------------------------------------------------------------------------- відео: читання і запис

def open_video(path: Path) -> tuple[cv2.VideoCapture, dict[str, Any]]:
    """Відкриває відео. Повертає (cap, info), де info — FPS, ширина, висота і кількість кадрів."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"OpenCV не зміг відкрити відео {path}: файл пошкоджений або формат не підтримується")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps > 0:  # так ловимо і 0, і NaN
        log.warning("Відео не повідомляє свій FPS — використовую %.0f", DEFAULT_FPS)
        fps = DEFAULT_FPS
    info = {
        "fps": fps,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),  # з метаданих, приблизно; 0 — невідомо
    }
    log.info("Вхідне відео: %s — %dx%d, %.2f FPS, кадрів: %d",
             path, info["width"], info["height"], fps, info["frames"])
    return cap, info


def create_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    """Створює VideoWriter: спершу пробує avc1 (H.264), потім mp4v. Повертає (writer, кодек)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Пробую кодеки %s. Англійські помилки OpenCV про OpenH264 нижче — це нормально: "
             "OpenCV тоді сам пробує інший спосіб запису", ", ".join(CODECS))
    for codec in CODECS:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            log.info("Вихідне відео пишеться кодеком %s (через %s)", codec, writer.getBackendName())
            if codec == "mp4v":
                log.warning("Кодек mp4v: браузер таке відео, найімовірніше, не відкриє — дивіться його у VLC")
            return writer, codec
        writer.release()
        log.warning("Кодек %s недоступний, пробую наступний", codec)
    raise SystemExit(f"Не вдалося створити вихідне відео {path} жодним кодеком з {CODECS}")


def output_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Розмір вихідного відео: пропорційно зменшений, щоб більша сторона була не більша за max_side.

    Сторони округлюємо до парних — цього вимагає H.264. max_side=0 — розмір не змінюється.
    Навіщо: кодек пише ~1 біт на піксель, тож 6 с відео 1080x1920 важать ~45 МБ, а 720x1280 — ~20 МБ.
    """
    if max_side == 0 or max(width, height) <= max_side:
        return width, height
    scale = max_side / max(width, height)
    return round(width * scale / 2) * 2, round(height * scale / 2) * 2


def resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Змінює розмір кадру на size = (ширина, висота); якщо розмір уже такий — повертає той самий кадр."""
    if (frame.shape[1], frame.shape[0]) == size:
        return frame
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)  # INTER_AREA — найкраще для зменшення


def save_example_frame(frame: np.ndarray, output: Path, number: int, frame_no: int) -> Path:
    """Зберігає кадр як <output>_<number>.jpg поруч з вихідним відео."""
    path = output.with_name(f"{output.stem}_{number}.jpg")
    # imencode + write_bytes замість cv2.imwrite: imwrite на Windows не вміє шляхи з кирилицею
    ok, data = cv2.imencode(".jpg", frame)
    if not ok:
        log.warning("Не вдалося закодувати кадр %d у jpg", frame_no)
        return path
    path.write_bytes(data.tobytes())
    log.info("Збережено кадр-приклад: %s (кадр %d)", path, frame_no)
    return path


def remove_old_examples(output: Path) -> None:
    """Видаляє кадри-приклади <output>_<N>.jpg з минулого запуску.

    Інакше, якщо цього разу збережеться менше кадрів, старі лишаться й виглядатимуть як нові.
    """
    prefix = f"{output.stem}_"
    old = [path for path in output.parent.glob("*.jpg")
           if path.stem.startswith(prefix) and path.stem[len(prefix):].isdigit()]
    for path in old:
        path.unlink()
    if old:
        log.info("Видалено старих кадрів-прикладів: %d", len(old))


# --------------------------------------------------------------------------- малювання

def drawing_sizes(frame: np.ndarray) -> tuple[int, float, int]:
    """Товщина рамки, масштаб і товщина шрифту залежно від роздільної здатності кадру.

    Рахуємо від меншої сторони: 720p -> 2 / 0.6 / 1, 1080p -> 3 / 0.9 / 2, 4K -> 6 / 1.8 / 4.
    """
    short_side = min(frame.shape[:2])  # frame.shape = (висота, ширина, канали)
    box_thickness = max(1, round(short_side / 360))
    font_scale = max(0.4, short_side / 1200)
    font_thickness = max(1, round(font_scale * 2))
    return box_thickness, font_scale, font_thickness


def draw_label(frame: np.ndarray, text: str, x: int, y: int,
               bg_color: tuple[int, int, int], text_color: tuple[int, int, int]) -> None:
    """Пише текст на залитому фоні над точкою (x, y); якщо зверху не вміщається — під нею.

    Для рамки (x, y) — її верхній лівий кут: підпис стоїть над рамкою,
    а якщо рамка біля верхнього краю кадру — всередині неї.
    """
    _, font_scale, font_thickness = drawing_sizes(frame)
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, font_scale, font_thickness)
    pad = max(2, round(font_scale * 4))
    block_w, block_h = text_w + 2 * pad, text_h + baseline + 2 * pad
    top = y - block_h if y - block_h >= 0 else y
    left = max(0, min(x, frame.shape[1] - block_w))  # не вилазимо за правий край
    cv2.rectangle(frame, (left, top), (left + block_w, top + block_h), bg_color, cv2.FILLED)
    cv2.putText(frame, text, (left + pad, top + pad + text_h), FONT, font_scale, text_color,
                font_thickness, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, boxes: Boxes, names: dict[int, str], scale: float = 1.0) -> None:
    """Малює рамки і підписи «bird 0.87» для всіх детекцій кадру (змінює frame на місці).

    scale — у скільки разів frame менший за кадр, на якому шукали птахів (координати рамок множимо на нього).
    """
    box_thickness, _, _ = drawing_sizes(frame)
    # .tolist() переносить дані з GPU і робить звичайні числа Python, які приймає cv2
    coords = (boxes.xyxy * scale).int().tolist()
    confs = boxes.conf.tolist()
    class_ids = boxes.cls.int().tolist()
    for (x1, y1, x2, y2), conf, cls_id in zip(coords, confs, class_ids):
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, box_thickness)
        draw_label(frame, f"{names[cls_id]} {conf:.2f}", x1, y1, BOX_COLOR, LABEL_TEXT_COLOR)


def draw_status(frame: np.ndarray, frame_no: int, total_frames: int, birds: int, fps: float) -> None:
    """Малює в лівому верхньому куті рядок стану: номер кадру, кількість птахів, FPS обробки."""
    position = f"frame {frame_no}/{total_frames}" if total_frames else f"frame {frame_no}"
    draw_label(frame, f"{position} | birds: {birds} | {fps:.1f} FPS", 0, 0, STATUS_BG_COLOR, STATUS_TEXT_COLOR)


# --------------------------------------------------------------------------- обробка

def detect(model: YOLO, frame: np.ndarray, args: argparse.Namespace, class_ids: list[int]) -> Boxes:
    """Запускає модель на одному кадрі й повертає знайдені рамки (лише клас bird)."""
    results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device,
                            classes=class_ids, verbose=False)
    return results[0].boxes  # predict повертає список — по одному результату на кожну картинку


def nested_share(a: list[float], b: list[float]) -> float:
    """Яка частка МЕНШОЇ з двох рамок [x1, y1, x2, y2] лежить усередині іншої: 0 — не перетинаються, 1 — повністю."""
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter_w * inter_h / smaller if smaller > 0 else 0.0


def remove_nested_boxes(boxes: Boxes, threshold: float) -> Boxes:
    """Прибирає дублікати «рамка всередині рамки», лишаючи з кожної такої пари впевненішу рамку.

    Навіщо: на птаха з довгим хвостом модель інколи дає дві рамки — «птах з хвостом» і «лише тіло».
    Вбудований NMS їх не зливає: він дивиться на IoU (перетин / об'єднання), а в такої пари він
    лише ~0.3–0.6, бо мала рамка вдвічі менша. Тому тут міряємо, яка частка меншої рамки
    лежить усередині іншої. Окремий птах на фоні заходить у чужу рамку зазвичай не більше ніж на ~60%.
    """
    if threshold <= 0 or len(boxes) < 2:
        return boxes
    coords = boxes.xyxy.tolist()
    confs = boxes.conf.tolist()
    order = sorted(range(len(coords)), key=lambda i: confs[i], reverse=True)  # від найвпевненішої
    keep: list[int] = []
    for i in order:
        # рамку лишаємо, якщо вона не «сидить» у жодній з уже залишених
        if all(nested_share(coords[i], coords[j]) < threshold for j in keep):
            keep.append(i)
    return boxes[keep]


def update_stats(stats: dict[str, Any], boxes: Boxes, removed: int) -> None:
    """Додає детекції поточного кадру (і кількість прибраних дублікатів) до загальної статистики."""
    stats["frames"] += 1
    stats["nested_removed"] += removed
    stats["detections"] += len(boxes)
    stats["frames_with_birds"] += int(len(boxes) > 0)
    stats["conf_sum"] += float(boxes.conf.sum())


def smooth_fps(previous: float, frame_seconds: float) -> float:
    """Згладжений FPS для рядка стану, щоб цифра не стрибала на кожному кадрі."""
    current = 1.0 / frame_seconds if frame_seconds > 0 else 0.0
    return current if previous == 0 else 0.9 * previous + 0.1 * current


def example_frame_gap(total_frames: int, fps: float, count: int) -> int:
    """Мінімальна відстань (у кадрах) між збереженими прикладами, щоб вони були з різних частин відео."""
    if total_frames > 0 and count > 0:
        return max(1, total_frames // count)
    return max(1, round(fps))  # кількість кадрів невідома — хоча б секунда між прикладами


def show_frame(frame: np.ndarray) -> bool:
    """Показує кадр у вікні. Повертає False, якщо натиснули q/Esc або закрили вікно."""
    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF  # waitKey ще й дає вікну перемалюватися
    closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    return key not in (ord("q"), 27) and not closed  # 27 — Esc


def process_video(model: YOLO, cap: cv2.VideoCapture, writer: cv2.VideoWriter, info: dict[str, Any],
                  args: argparse.Namespace, class_ids: list[int], out_size: tuple[int, int]) -> dict[str, Any]:
    """Головний цикл: читаємо кадр -> детекція -> зменшення -> малювання -> запис. Повертає статистику."""
    stats = {"frames": 0, "detections": 0, "frames_with_birds": 0, "conf_sum": 0.0, "nested_removed": 0,
             "stopped": False}
    gap = example_frame_gap(info["frames"], info["fps"], args.save_frames)
    examples: list[Path] = []
    next_example = 0  # номер кадру, з якого вже можна зберігати наступний приклад
    fps = 0.0
    started = last = time.perf_counter()
    if args.show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # вікно можна розтягувати мишкою
    # logging_redirect_tqdm — щоб повідомлення логу не ламали смужку прогресу
    with logging_redirect_tqdm(), tqdm(total=info["frames"] or None, unit="кадр", desc="Обробка") as progress:
        while True:
            ok, frame = cap.read()  # frame — масив numpy (висота, ширина, 3) у порядку BGR
            if not ok:
                break  # кадри закінчилися
            raw_boxes = detect(model, frame, args, class_ids)
            boxes = remove_nested_boxes(raw_boxes, args.nested)
            update_stats(stats, boxes, len(raw_boxes) - len(boxes))
            now = time.perf_counter()
            fps, last = smooth_fps(fps, now - last), now
            # птахів шукаємо на повному кадрі, а малюємо вже на зменшеному — так підписи чіткіші
            scale = out_size[0] / frame.shape[1]
            frame = resize_frame(frame, out_size)
            draw_detections(frame, boxes, model.names, scale)
            draw_status(frame, stats["frames"], info["frames"], len(boxes), fps)
            writer.write(frame)
            progress.update(1)
            if len(boxes) and len(examples) < args.save_frames and stats["frames"] >= next_example:
                examples.append(save_example_frame(frame, args.output, len(examples) + 1, stats["frames"]))
                next_example = stats["frames"] + gap
            if args.show and not show_frame(frame):
                stats["stopped"] = True
                break
    stats["seconds"] = time.perf_counter() - started
    return stats


def log_summary(stats: dict[str, Any], info: dict[str, Any], output: Path, codec: str, device: str) -> None:
    """Виводить підсумок: кадри, FPS, детекції, частка кадрів з птахами, середній confidence."""
    frames, detections = stats["frames"], stats["detections"]
    if stats["stopped"]:
        log.warning("Обробку зупинено вручну — у вихідному відео лише перші %d кадрів", frames)
    log.info("Оброблено кадрів: %d (за метаданими відео: %d)", frames, info["frames"])
    log.info("Середній FPS обробки: %.1f (пристрій %s; FPS самого відео: %.2f)",
             frames / stats["seconds"], device, info["fps"])
    log.info("Усього детекцій: %d (прибрано дублікатів «рамка в рамці»: %d)", detections, stats["nested_removed"])
    log.info("Кадрів з птахами: %d з %d (%.1f%%)",
             stats["frames_with_birds"], frames, 100 * stats["frames_with_birds"] / frames)
    if detections:
        log.info("Середній confidence: %.3f", stats["conf_sum"] / detections)
    else:
        log.info("Середній confidence: — (жодної детекції)")
    log.info("Результат: %s (кодек %s, %.1f МБ)", output, codec, output.stat().st_size / 1e6)


# --------------------------------------------------------------------------- головна функція

def main() -> None:
    """Точка входу: перевірки -> модель -> відкриття відео -> обробка кожного кадру -> підсумок."""
    setup_logging()
    args = parse_args()
    args.input = PROJECT_ROOT / args.input  # якщо шлях абсолютний, Path поверне саме його
    args.output = PROJECT_ROOT / args.output
    check_paths(args)
    model, class_ids = load_model(args.weights)
    args.imgsz = pick_imgsz(args.imgsz, model)
    cap, info = open_video(args.input)
    out_size = output_size(info["width"], info["height"], args.max_side)
    if out_size != (info["width"], info["height"]):
        log.info("Вихідне відео: %dx%d (зменшено до --max-side %d)", *out_size, args.max_side)
    writer = None
    try:
        writer, codec = create_writer(args.output, info["fps"], out_size)
        remove_old_examples(args.output)
        stats = process_video(model, cap, writer, info, args, class_ids, out_size)
    finally:
        cap.release()
        if writer is not None:
            writer.release()  # саме тут дописується кінець mp4; без release файл не відкриється
        if args.show:
            cv2.destroyAllWindows()
    if stats["frames"] == 0:
        raise SystemExit(f"Не вдалося прочитати жодного кадру з {args.input}")
    # model.predictor.device, а не model.device: predict працює з копією моделі на GPU,
    # а model.device показує оригінал, який лишився на cpu
    log_summary(stats, info, args.output, codec, str(model.predictor.device))
    log.info("Готово.")


if __name__ == "__main__":
    main()
