import queue
import threading
import time

import cv2
import mss
import numpy as np
import pyautogui
import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw, ImageFont, ImageOps

import alphabets

DEBUG = True


def get_screen_size():
    screen_width, screen_height = pyautogui.size()
    return screen_width, screen_height


alive = True


def capture_screen(monitor, screen_size, q):
    global alive
    with mss.mss() as sct:
        while alive:
            t = time.time() if DEBUG else None
            screenshot = sct.grab(monitor)
            # screenshot = pyautogui.screenshot()
            print("screenshot", time.time() - t) if DEBUG else None
            t = time.time() if DEBUG else None
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame = cv2.resize(frame, screen_size, interpolation=cv2.INTER_NEAREST)
            frame = frame[..., :3]
            print("resize", time.time() - t) if DEBUG else None
            try:
                q.put(frame, timeout=None if DEBUG else 1)
            except queue.Full:
                pass


def capture_video(file_input, screen_size, q):
    global alive
    cap = cv2.VideoCapture(file_input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm.tqdm(total=total_frames)
    while alive and cap.isOpened():
        t = time.time() if DEBUG else None
        ret, frame = cap.read()
        print("video capture", time.time() - t) if DEBUG else None
        t = time.time() if DEBUG else None
        frame = cv2.resize(frame, screen_size, interpolation=cv2.INTER_NEAREST)
        if not ret:
            break
        try:
            q.put(frame, timeout=None if DEBUG else 1)
        except queue.Full:
            if alive:
                q.put(frame)
        pbar.update(1)
    alive = False
    pbar.close()
    cap.release()


def screen_to_ascii(
    CHAR_LIST: list[str] = alphabets.GENERAL["complex"],
    num_cols: int = 100,
    shrink: float = 0.5,
    fps: int = 20,
    bg_color: tuple[int] = (0, 0, 0),
    show_fps: bool = True,
    file_input: str = None,
    low_res: bool = False,
):
    screen_size = get_screen_size()
    screen_width, screen_height = screen_size
    cell_width = screen_width / num_cols
    cell_height = 1.7 * cell_width
    font_size = int(min(cell_width, cell_height) * shrink * 2)
    font = ImageFont.truetype("fonts/DejaVuSansMono-Bold.ttf", size=font_size)
    num_rows = int(screen_height / cell_height)

    monitor = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}
    cell_width = cell_width * shrink
    cell_height = cell_height * shrink
    screen_width = int(screen_width * shrink)
    screen_height = int(screen_height * shrink)
    screen_size = (screen_width, screen_height)

    total_frames = None
    if file_input:
        cap = cv2.VideoCapture(file_input)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, fps, screen_size)

    q1 = queue.Queue(maxsize=1)
    q2 = queue.Queue(maxsize=1)
    q3 = queue.Queue(maxsize=1)

    threads = [
        threading.Thread(
            target=capture_video if file_input else capture_screen,
            args=(
                file_input if file_input else monitor,
                screen_size,
                q1,
            ),
        ),
        threading.Thread(
            target=process_frame,
            args=(
                np.array(list(CHAR_LIST)),
                num_cols,
                cell_width,
                cell_height,
                num_rows,
                q1,
                q2,
                low_res,
            ),
        ),
        threading.Thread(
            target=draw_text,
            args=(
                screen_size,
                font,
                bg_color,
                q2,
                q3,
            ),
        ),
        threading.Thread(
            target=display_frame,
            args=(
                out,
                q3,
                bg_color,
                font,
                show_fps,
                total_frames,
            ),
        ),
    ]

    global alive
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        alive = False
        for thread in threads:
            thread.join()
    finally:
        out.release()
        cv2.destroyAllWindows()


def process_frame(
    CHAR_LIST: np.ndarray,
    num_cols: int,
    cell_width: float,
    cell_height: float,
    num_rows: int,
    q1: queue.Queue[np.ndarray],
    q2: queue.Queue[Image.Image],
    low_res: bool = False,
):
    global alive
    while alive:
        frame = q1.get(timeout=None if DEBUG else 1)
        t = time.time() if DEBUG else None

        if DEBUG:
            cv2.imshow("frame1", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                alive = False
                break

        # Use np slicing to get the partial images
        partial_images = sliding_window_view(
            frame, (int(cell_height), int(cell_width), 3)
        )
        if low_res:
            partial_images = partial_images[
                :: int(cell_height),
                :: int(cell_width),
                0,
                0,
            ]
            partial_avg_colors = partial_images.mean(axis=(2))
            mean_values = partial_images.mean(axis=(2, 3))
        else:
            partial_images = partial_images[
                :: int(cell_height),
                :: int(cell_width),
            ]
            partial_avg_colors = partial_images.mean(axis=(2, 3, 4))
            mean_values = partial_images.mean(axis=(2, 3, 4, 5))

        partial_avg_colors = partial_avg_colors.astype(np.int32)

        char_indices = np.clip(
            (mean_values * len(CHAR_LIST) / 255).astype(int), 0, len(CHAR_LIST) - 1
        )
        chars: str = CHAR_LIST[char_indices]
        y_coords = np.arange(num_rows) * cell_height
        x_coords = np.arange(num_cols) * cell_width

        print("mean", time.time() - t) if DEBUG else None
        t = time.time() if DEBUG else None

        try:
            q2.put(
                (partial_avg_colors, chars, y_coords, x_coords),
                timeout=None if DEBUG else 1,
            )
        except queue.Full:
            pass


def draw_text(
    screen_size: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    bg_color: tuple[int],
    q2: queue.Queue[tuple[np.ndarray, str, np.ndarray, np.ndarray]],
    q3: queue.Queue[Image.Image],
):
    global alive
    while alive:
        args = q2.get(timeout=None if DEBUG else 1)
        partial_avg_colors, chars, y_coords, x_coords = args
        t = time.time() if DEBUG else None

        out_image = Image.new(
            "RGB",
            screen_size,
            bg_color,
        )
        draw = ImageDraw.Draw(out_image)

        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                partial_avg_color = tuple(partial_avg_colors[i, j].tolist())
                char = chars[i, j]
                draw.text(
                    (x, y),
                    char,
                    fill=partial_avg_color,
                    font=font,
                )
        print("text", time.time() - t) if DEBUG else None
        try:
            q3.put(out_image, timeout=None if DEBUG else 1)
        except queue.Full:
            pass


def display_frame(
    out: cv2.VideoWriter,
    q3: queue.Queue[Image.Image],
    bg_color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont,
    show_fps: bool = True,
    total_frames: int = None,
    border_removal: bool = False,
):
    global alive
    current_frame = 0
    prev_time = time.time()
    while alive:
        out_image = q3.get(timeout=None if DEBUG else 1)
        t = time.time() if DEBUG else None
        out.write(np.array(out_image))
        current_frame += 1

        # Crop the image to remove the border
        if border_removal:
            if bg_color == (255, 255, 255):
                cropped_image = ImageOps.invert(out_image).getbbox()
            else:
                cropped_image = out_image.getbbox()
            out_image = out_image.crop(cropped_image)

        # Display the FPS if required
        if show_fps:
            draw = ImageDraw.Draw(out_image)
            current_time = time.time()
            time_diff = current_time - prev_time
            if time_diff > 0:
                fps = 1 / time_diff
            else:
                fps = float("inf")
            prev_time = current_time
            draw.text(
                (10, 10),
                f"FPS: {fps:.2f}",
                fill=(
                    255 - bg_color[0],
                    255 - bg_color[1],
                    255 - bg_color[2],
                ),
                font=font,
            )

        # Progress percentage
        if total_frames:
            text = f"{(current_frame) / total_frames * 100:.2f}%"
            percent_font = ImageFont.truetype(
                "fonts/DejaVuSansMono-Bold.ttf", size=font.size * 2
            )
            text_bbox = draw.textbbox((0, 0), text, font=percent_font)
            text_size = (
                text_bbox[2] - text_bbox[0],
                text_bbox[3] - text_bbox[1],
            )
            draw.text(
                (
                    (out_image.width - text_size[0]) // 2,
                    (out_image.height - text_size[1] - 60),
                ),
                text,
                fill=(
                    255 - bg_color[0],
                    255 - bg_color[1],
                    255 - bg_color[2],
                ),
                font=percent_font,
            )

        # Display the progress bar if possible
        out_image = np.array(out_image)
        if total_frames:
            progress = (current_frame) / total_frames
            cv2.rectangle(
                out_image,
                (10, out_image.shape[0] - 30),
                (int(out_image.shape[1] * progress), out_image.shape[0] - 10),
                (255, 255, 255),
                -1,
            )

        cv2.imshow("ASCII Stream", out_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            alive = False
            break

        print("display", time.time() - t) if DEBUG else None
        t = time.time() if DEBUG else None


if __name__ == "__main__":
    # screen_to_ascii(file_input="data/input.mp4")
    screen_to_ascii(low_res=True)
