import win32con
import win32gui
import win32process
import win32api
import numpy as np
from pykeyboard import *
from pymouse import *
import time
from PyQt5.QtWidgets import QApplication
from PIL import ImageQt
import ctypes


class NSSHAFT:
    def __init__(self):
        self._start_game()
        self.start_pos = (426, 353)  # 开始游戏按钮位置
        self.mouse = PyMouse()
        self.keyboard = PyKeyboard()
        self.hold_time = 0.1  # 按键持续时长
        self.game_region = (0, 0, 440, 420)  # 游戏画面区域
        self.life_end = (140, 42)  # 生命条最后一格中心位置
        self.life_span = (140 - 51) / 11  # 生命条每格间距
        self.done_pos = (176, 185)  # 判断游戏是否终止的像素点
        self.pause_pos = (538, 371)  # 暂停游戏像素点
        self.max_life = 12
        self.last_life = 12
        self.current_life = 12
        self.current_pic = None
        self.current_pic_arr = None
        self.current_state = None
        self.state_scale_rate = 0.6  # 返回图片缩放比例
        self.last_time = time.time()
        self.index = 1
        self.address_life = 0x514C30  # 血量内存地址
        self.address_x = 0x514C18  # 小人x轴坐标内存地址
        self.address_y = 0x514C1C  # 小人y轴坐标内存地址
        self.address_fly = 0x514C34  # 小人是否在飞的内存地址，1位飞，0为站立
        self.address_layer = 0x514DC0  # 小人下到多少层的内存地址
        self.last_action = 1

    def action_dim(self):
        return 3

    def _update_state(self, update=True):
        self.current_pic_arr = np.array(self.current_pic).astype(np.uint8)
        if update:
            self.last_life = self.current_life
            self.current_life = self.get_life_value()
            self.last_pos = self.current_pos
            self.current_pos = self.get_pos()
        w, h = self.current_pic.size
        new_size = (int(w * self.state_scale_rate), int(h * self.state_scale_rate))
        scaled_img = self.current_pic.resize(new_size)
        self.current_state = np.array(scaled_img).astype(np.uint8)

    def mouse_click(self, x, y):
        win32api.SetCursorPos([x, y])
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def _get_reward(self, is_done, action):
        if is_done and self.current_life != 0:
            return -40
        else:
            life_change = self.current_life - self.last_life
            last_x, last_y = self.last_pos
            r = 0
            x, y = self.current_pos
            if y < 100:  # 惩罚太靠上
                r = (y - 100) / 30
            r = r + min(max(0.0, (y + 20 - last_y) / 20), 1.5)
            r = r + life_change * 2
            if x < 100 or x > 252:  # 尽量向中间走，越靠边惩罚越大。
                r = r - abs(x - 176) / 176
            if y > 290:
                r = max(-30, 290-y)  # 惩罚太靠下
            return r

    def step(self, action):
        self.index += 1
        self.pause_resume()  # 恢复游戏
        self._action(action)
        self.pause_resume()  # 暂停游戏
        self._update_state()

        done = self.is_done()
        if done:
            self._release_last_key()

        layers = self.get_layers()
        if done and layers > 92:
            print("Layers:{}, press enter key.".format(layers))
            self.keyboard.press_key(self.keyboard.enter_key)
            time.sleep(0.1)
            self.keyboard.release_key(self.keyboard.enter_key)
        reward = self._get_reward(done, action)
        state = self.current_state
        x, y = self.get_pos()
        layers = self.get_layers()
        isFly = self.is_fly()
        return state, reward, done, isFly, x, y, layers

    def pause_resume(self):
        self.mouse_click(*self.pause_pos)

    def _start_game(self):
        self.ns_shaft = win32gui.FindWindow("NsShaftClass", "NS-SHAFT")
        if self.ns_shaft == 0:
            win32api.ShellExecute(0, 'open', '.\\nssh13j\\NSSHAFT.exe', '', '', 1)
            time.sleep(1)
            keyboard = PyKeyboard()
            keyboard.press_key(keyboard.enter_key)
            time.sleep(0.1)
            keyboard.release_key(keyboard.enter_key)
            self.ns_shaft = win32gui.FindWindow("NsShaftClass", "NS-SHAFT")
        win32gui.SendMessage(self.ns_shaft, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
        win32gui.SetForegroundWindow(self.ns_shaft)
        win32gui.SetWindowPos(self.ns_shaft, win32con.HWND_TOPMOST, 0, 0, 0, 100,
                              win32con.SWP_SHOWWINDOW | win32con.SWP_NOSIZE)

        hid, pid = win32process.GetWindowThreadProcessId(self.ns_shaft)
        PROCESS_ALL_ACCESS = (0x000F0000 | 0x00100000 | 0xFFF)
        self.phand = win32api.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
        self.dll = ctypes.windll.LoadLibrary("C:\\Windows\\System32\\kernel32.dll")

    def get_life_value(self):
        return self._get_memory_value(self.address_life)

    def get_layers(self):
        return self._get_memory_value(self.address_layer)

    def _get_memory_value(self, address):
        data = ctypes.c_long()
        self.dll.ReadProcessMemory(int(self.phand), address, ctypes.byref(data), 4, None)
        return data.value

    def get_pos(self):
        x = self._get_memory_value(self.address_x)
        y = self._get_memory_value(self.address_y)
        return x, y

    def is_done(self):
        return self._get_memory_value(self.address_y) == 352 or self.get_life_value() == 0

    def _rolling_board(self, rgb):  # 根据像素值判断是否落在了翻转板上。
        rgb = rgb.astype(np.int32)
        if rgb[0] == 173 and rgb[1] == 169 and rgb[2] == 144:
            return False
        if 100 < rgb[0] < 210 and 00 < rgb[1] < 210 and 00 < rgb[2] < 210 and rgb[2] - rgb[0] < -10:
            return True
        return False

    def is_fly(self):
        fly = self._get_memory_value(self.address_fly) == 1
        if not fly:
            x, y = self.current_pos
            checks = [30, 35, 40, 70, 75, 80]
            y = y + 104
            for c in checks:
                x = x + c
                if x < 40 or x > 390 or y > 410:
                    continue
                rgb = self.current_pic_arr[y, x, :]
                if self._rolling_board(rgb):
                    fly = True
                    break
        return fly

    def reset(self):
        self.mouse.click(*self.start_pos, 1)
        time.sleep(1)
        self._print_screen()
        self.pause_resume()  # 暂停游戏
        self.current_life = 12
        self.last_pos = (176, 320)
        self.current_pos = (176, 320)
        self.last_life = 12
        self.last_time = time.time()
        self.last_action = 1
        self._update_state(False)
        x, y = self.get_pos()
        return self.current_state, x, y

    def _release_last_key(self):
        if self.last_action == 0:
            self.keyboard.release_key(self.keyboard.left_key)
        elif self.last_action == 2:
            self.keyboard.release_key(self.keyboard.right_key)

    def _action(self, act):
        if self.last_action != act:
            self._release_last_key()
        if act == 0:  # left
            if self.last_action != act:
                self.keyboard.press_key(self.keyboard.left_key)
            time.sleep(self.hold_time)
            self._print_screen()

        elif act == 1:  # stay
            time.sleep(self.hold_time)
            self._print_screen()
        else:  # right
            if self.last_action != act:
                self.keyboard.press_key(self.keyboard.right_key)
            time.sleep(self.hold_time)
            self._print_screen()
        self.last_action = act

    def _print_screen(self):
        app = QApplication(sys.argv)
        screen = QApplication.primaryScreen()
        img = screen.grabWindow(self.ns_shaft).toImage()
        # self.pause_resume()  # 暂停游戏
        image = ImageQt.fromqimage(img)
        self.current_pic = image.crop(self.game_region)




