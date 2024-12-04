import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
import pygetwindow as gw
import pyautogui
import alphabets


def capture_screen():
    screen_width, screen_height = pyautogui.size()
    return screen_width, screen_height


def screen_to_ascii():
    CHAR_LIST = "@%#*+=-:. "
    CHAR_LIST = alphabets.GENERAL["complex"]
    screen_width, screen_height = capture_screen()
    num_cols = 100
    cell_width = screen_width / num_cols
    cell_height = 1.7 * cell_width
    font_size = int(cell_width)
    font = ImageFont.truetype("fonts/DejaVuSansMono-Bold.ttf", size=font_size)
    num_rows = int(screen_height / cell_height)

    shrink = 0.5

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 20
    out = cv2.VideoWriter(
        "screen_output.mp4", fourcc, fps, (screen_width, screen_height)
    )

    try:
        while True:
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            out_image = Image.new(
                "RGB",
                (int(screen_width * shrink), int(screen_height * shrink)),
                (255, 255, 255),
            )
            draw = ImageDraw.Draw(out_image)

            for i in range(num_rows):
                for j in range(num_cols):
                    partial_image = frame[
                        int(i * cell_height) : min(
                            int((i + 1) * cell_height), screen_height
                        ),
                        int(j * cell_width) : min(
                            int((j + 1) * cell_width), screen_width
                        ),
                        :,
                    ]
                    partial_avg_color = np.sum(
                        np.sum(partial_image, axis=0), axis=0
                    ) / (cell_height * cell_width)
                    partial_avg_color = tuple(
                        partial_avg_color.astype(np.int32).tolist()
                    )
                    char = CHAR_LIST[
                        min(
                            int(np.mean(partial_image) * len(CHAR_LIST) / 255),
                            len(CHAR_LIST) - 1,
                        )
                    ]
                    draw.text(
                        (j * int(cell_width * shrink), i * int(cell_height * shrink)),
                        char,
                        fill=partial_avg_color,
                        font=font,
                    )

            out_image = np.array(out_image)
            out.write(out_image)
            cv2.imshow("ASCII Stream", out_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    screen_to_ascii()
