import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
import pygetwindow as gw
import pyautogui


def capture_screen():
    screen_width, screen_height = pyautogui.size()
    return screen_width, screen_height


def screen_to_ascii():
    CHAR_LIST = "@%#*+=-:. "
    bg_code = 255

    font = ImageFont.truetype("fonts/DejaVuSansMono-Bold.ttf", size=10)

    screen_width, screen_height = capture_screen()

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 20
    out = cv2.VideoWriter(
        "screen_output.mp4", fourcc, fps, (screen_width, screen_height)
    )

    try:
        while True:
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            height, width = frame.shape
            cell_width = width / 100
            cell_height = 2 * cell_width
            num_rows = int(height / cell_height)
            num_cols = int(width / cell_width)

            out_image = Image.new("L", (width, height), bg_code)
            draw = ImageDraw.Draw(out_image)

            for i in range(num_rows):
                line = "".join(
                    [
                        CHAR_LIST[
                            min(
                                int(
                                    np.mean(
                                        frame[
                                            int(i * cell_height) : min(
                                                int((i + 1) * cell_height), height
                                            ),
                                            int(j * cell_width) : min(
                                                int((j + 1) * cell_width), width
                                            ),
                                        ]
                                    )
                                    * len(CHAR_LIST)
                                    / 255
                                ),
                                len(CHAR_LIST) - 1,
                            )
                        ]
                        for j in range(num_cols)
                    ]
                )
                draw.text((0, i * 10), line, fill=0, font=font)

            out_image = np.array(out_image)
            out_image = cv2.cvtColor(out_image, cv2.COLOR_GRAY2BGR)

            out.write(out_image)

            cv2.imshow("ASCII Stream", out_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    screen_to_ascii()
