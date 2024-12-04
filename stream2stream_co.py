import queue
import threading
import time

import cv2
import mss
import numpy as np
import pyautogui
from PIL import Image, ImageDraw, ImageFont

import alphabets

DEBUG = False


def get_screen_size():
    screen_width, screen_height = pyautogui.size()
    return screen_width, screen_height


alive = True


def capture_screen(monitor, q):
    global alive
    with mss.mss() as sct:
        while alive:
            t = time.time() if DEBUG else None
            screenshot = sct.grab(monitor)
            try:
                q.put(np.array(screenshot), timeout=1)
            except queue.Full:
                pass
            print("screenshot", time.time() - t) if DEBUG else None
            t = time.time() if DEBUG else None


def screen_to_ascii(
    CHAR_LIST=alphabets.GENERAL["complex"],
    shrink=0.5,
    fps=20,
    show_fps=True,
):
    screen_width, screen_height = get_screen_size()
    num_cols = 100
    cell_width = screen_width / num_cols
    cell_height = 1.7 * cell_width
    font_size = int(min(cell_width, cell_height) * shrink * 2)
    font = ImageFont.truetype("fonts/DejaVuSansMono-Bold.ttf", size=font_size)
    num_rows = int(screen_height / cell_height)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        "screen_output.mp4", fourcc, fps, (screen_width, screen_height)
    )

    monitor = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}
    cell_width = int(cell_width * shrink)
    cell_height = int(cell_height * shrink)
    screen_width = int(screen_width * shrink)
    screen_height = int(screen_height * shrink)

    q1 = queue.Queue(maxsize=1)
    q2 = queue.Queue(maxsize=1)

    threads = [
        threading.Thread(target=capture_screen, args=(monitor, q1)),
        threading.Thread(
            target=process_frame,
            args=(
                CHAR_LIST,
                (screen_width, screen_height),
                num_cols,
                cell_width,
                cell_height,
                font,
                num_rows,
                show_fps,
                q1,
                q2,
            ),
        ),
        threading.Thread(
            target=display_frame,
            args=(out, q2),
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
    CHAR_LIST,
    screen_size,
    num_cols,
    cell_width,
    cell_height,
    font,
    num_rows,
    show_fps,
    q1,
    q2,
):
    global alive
    prev_time = time.time()
    while alive:
        frame = q1.get(timeout=1)
        t = time.time() if DEBUG else None
        frame = cv2.resize(frame, screen_size)
        print("np cv", time.time() - t) if DEBUG else None
        t = time.time() if DEBUG else None

        screen_width, screen_height = screen_size

        out_image = Image.new(
            "RGB",
            (screen_width, screen_height),
            (255, 255, 255),
        )
        draw = ImageDraw.Draw(out_image)
        print("new draw", time.time() - t) if DEBUG else None
        t = time.time() if DEBUG else None

        for i in range(num_rows):
            for j in range(num_cols):
                partial_image = frame[
                    int(i * cell_height) : min(
                        int((i + 1) * cell_height), screen_height
                    ),
                    int(j * cell_width) : min(int((j + 1) * cell_width), screen_width),
                    :,
                ]
                partial_avg_color = np.sum(np.sum(partial_image, axis=0), axis=0) / (
                    cell_height * cell_width
                )
                partial_avg_color = tuple(partial_avg_color.astype(np.int32).tolist())
                char = CHAR_LIST[
                    min(
                        int(np.mean(partial_image) * len(CHAR_LIST) / 255),
                        len(CHAR_LIST) - 1,
                    )
                ]
                draw.text(
                    (
                        j * cell_width,
                        i * cell_height,
                    ),
                    char,
                    fill=partial_avg_color,
                    font=font,
                )
        print("main", time.time() - t) if DEBUG else None
        t = time.time() if DEBUG else None

        if show_fps:
            current_time = time.time()
            time_diff = current_time - prev_time
            if time_diff > 0:
                fps = 1 / time_diff
            else:
                fps = float("inf")
            prev_time = current_time
            draw.text((10, 10), f"FPS: {fps:.2f}", fill=(0, 0, 0), font=font)

        q2.put(out_image)


def display_frame(out, q2):
    global alive
    while alive:
        t = time.time() if DEBUG else None
        out_image = q2.get(timeout=1)
        out_image = np.array(out_image)
        out.write(out_image)
        cv2.imshow("ASCII Stream", out_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            alive = False
            break

        print("display", time.time() - t) if DEBUG else None
        t = time.time() if DEBUG else None


if __name__ == "__main__":
    screen_to_ascii()
