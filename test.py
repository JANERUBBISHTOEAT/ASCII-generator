import time

import mss
import numpy as np
import pyautogui


def capture_screen_mss(monitor):
    with mss.mss() as sct:
        screenshot = sct.grab(monitor)
        return np.array(screenshot)


def capture_screen():
    screen_width, screen_height = pyautogui.size()
    return screen_width, screen_height


screen_width, screen_height = capture_screen()
monitor = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}
frame = capture_screen_mss(monitor)
t = time.time()
while 1:
    # screenshot = pyautogui.screenshot() # 19
    screenshot = capture_screen_mss(monitor)  # 24
    print(1 / (time.time() - t))
    t = time.time()
