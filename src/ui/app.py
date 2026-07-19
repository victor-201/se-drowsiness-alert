import os
import cv2
import logging
import numpy as np
import threading
import queue
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Rectangle
from kivy.core.audio import SoundLoader
from kivy.uix.screenmanager import ScreenManager
from src.core.detector import DrowsinessDetector
from src.configs.config import Config
from src.configs.settings import Settings
from src.ui.screens.main_screen import MainScreen
from src.ui.screens.settings_screen import SettingsScreen

logger = logging.getLogger(__name__)

class DrowsinessDetectorApp(App):
    def __init__(self):
        super().__init__()
        self.app_config = Config()
        self.detector = DrowsinessDetector()
        self.image = Image(size_hint=(1, 1))
        self.status_label = Label(text='Trạng thái: Đã dừng', size_hint=(1, 0.1))
        self.settings = Settings()
        self.sound_alert_dir = self.app_config.SOUND_ALERT_DIR
        self.image_dir = self.app_config.IMAGE_DIR
        self.alert_sound_file = self.app_config.ALERT_SOUND_FILE
        self.fatigue_sound_file = self.app_config.FATIGUE_SOUND_FILE
        self.eye_open_threshold = self.detector.eye_open_threshold
        self.camera_width = self.app_config.CAMERA_WIDTH
        self.camera_height = self.app_config.CAMERA_HEIGHT
        self.camera_fps = self.app_config.CAMERA_FPS
        self.initialize_app()
        self.last_metrics = {
            'eye_openness': None,
            'mouth_openness': None,
            'blink_count': 0,
            'yawn_count': 0
        }

    def initialize_app(self):
        self.camera_initialized = False
        self.alert_sound = None
        self.fatigue_sound = None
        self.setup_alert_sound()
        self.setup_fatigue_sound()
        self.setup_state_variables()
        logger.info(f"Camera khả dụng: {self.settings.get_available_cameras()}")

    def setup_alert_sound(self):
        default_sound = os.path.join(self.sound_alert_dir, "alert.wav")
        sound_path = (
            os.path.join(self.sound_alert_dir, self.settings.alert_sound_file)
            if self.settings.alert_sound_file
            else default_sound
        )
        if os.path.exists(sound_path):
            self.alert_sound = SoundLoader.load(sound_path)
            if not self.alert_sound:
                logging.warning(f"Không thể tải file âm thanh: {sound_path}")
        else:
            logging.warning(f"File âm thanh không tồn tại: {sound_path}")

    def setup_fatigue_sound(self):
        sound_path = self.fatigue_sound_file
        if os.path.exists(sound_path):
            self.fatigue_sound = SoundLoader.load(sound_path)
            if not self.fatigue_sound:
                logging.warning(f"Không thể tải file âm thanh thông báo: {sound_path}")
        else:
            logging.warning(f"File âm thanh thông báo không tồn tại: {sound_path}")

    def setup_state_variables(self):
        self.is_monitoring = False
        self.alert_active = False
        self.alert_stop_timer = None
        self.alert_stop_delay = self.app_config.ALERT_STOP_DELAY
        self.calibration_event = None
        self.calibration_start_time = None
        self.background_color = [0, 0, 0, 1]
        self.screen_manager = None
        self._main_screen_cache = None
        self._capture_thread = None
        self._processing_thread = None
        self._stop_event = threading.Event()
        self._frame_queue = queue.Queue(maxsize=1)
        self._result_queue = queue.Queue(maxsize=1)
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._worker_lock = threading.RLock()

    def _get_main_screen(self):
        if self._main_screen_cache is None and self.screen_manager is not None:
            self._main_screen_cache = self.screen_manager.get_screen('main')
        return self._main_screen_cache

    def _update_main_screen_metrics(self, eye_override=None):
        main_screen = self._get_main_screen()
        if main_screen is None:
            return
        eye_openness = (
            eye_override
            if eye_override is not None
            else self.last_metrics['eye_openness']
        )
        main_screen.update_metrics(
            eye_openness,
            self.last_metrics['mouth_openness'],
            self.last_metrics['blink_count'],
            self.last_metrics['yawn_count']
        )

    def build(self):
        self.screen_manager = ScreenManager()
        self._main_screen_cache = None
        main_screen = MainScreen(app_instance=self, name='main')
        self.screen_manager.add_widget(main_screen)
        settings_screen = SettingsScreen(name='settings', app_instance=self)
        self.screen_manager.add_widget(settings_screen)
        Clock.schedule_once(self._init_camera, 0.1)
        Clock.schedule_interval(self._update_wrapper, 1.0 / 30.0)
        return self.screen_manager

    def _init_camera(self, dt):
        try:
            self.detector.start_camera()
            self.camera_initialized = True
            logging.info("Khởi tạo camera thành công")
            self.status_label.text = 'Trạng thái: Đã dừng'
        except Exception as e:
            logging.error(f"Khởi tạo camera thất bại: {e}")
            self.status_label.text = 'Lỗi: Không khởi tạo được camera'

    def switch_to_settings(self, instance):
        self.stop_monitoring(instance)
        self.screen_manager.current = 'settings'
        logging.info("Chuyển sang màn hình cài đặt")
        self._update_main_screen_metrics()

    def switch_to_main(self):
        self.screen_manager.current = 'main'
        logging.info("Chuyển về màn hình chính")
        self._update_main_screen_metrics()

    def update_background_color(self):
        if self.screen_manager.current == 'main':
            main_screen = self._get_main_screen()
            if main_screen is None or not hasattr(main_screen, 'background_color_instruction'):
                return
            main_screen.background_color_instruction.rgba = self.background_color

    def exit_app(self, instance):
        logging.info("Thoát ứng dụng")
        self.stop_monitoring(instance)
        App.get_running_app().stop()

    def _update_wrapper(self, dt):
        if self.is_monitoring and self.camera_initialized:
            self._consume_frame_result()
        elif self.calibration_event and self.camera_initialized:
            self.update_calibration(dt)
        else:
            self._update_main_screen_metrics()

    def _capture_loop(self, stop_event):
        while not stop_event.is_set():
            try:
                ret, frame = self.detector.read_camera_frame()
                if not ret or frame is None:
                    try:
                        self.detector.start_camera()
                    except Exception as e:
                        logger.error(f"Reinit camera failed: {e}")
                        stop_event.wait(1.0)
                        continue
                    ret, frame = self.detector.read_camera_frame()
                if ret and frame is not None:
                    frame = cv2.resize(
                        frame,
                        (
                            self.detector.config.CAMERA_WIDTH,
                            self.detector.config.CAMERA_HEIGHT,
                        ),
                    )
                    if self._frame_queue.full():
                        try:
                            self._frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self._frame_queue.put_nowait(frame)
                else:
                    stop_event.wait(0.05)
            except Exception as e:
                logger.exception("Capture error: %s", e)
                stop_event.wait(0.1)

    def _processing_loop(self, stop_event):
        while not stop_event.is_set():
            try:
                try:
                    frame = self._frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                result_frame, alert_detected, metrics = self.detector.process_frame_from_frame(frame)
                result = (result_frame, alert_detected, metrics)
                if self._result_queue.full():
                    try:
                        self._result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._result_queue.put_nowait(result)
            except Exception as e:
                logger.exception("Processing error (thread): %s", e)
                stop_event.wait(0.05)

    def _stop_worker_threads(self, join_timeout=2.0):
        with self._worker_lock:
            stop_event = self._stop_event
            capture_thread = self._capture_thread
            processing_thread = self._processing_thread
            stop_event.set()

        current_thread = threading.current_thread()
        for worker in (capture_thread, processing_thread):
            if worker and worker is not current_thread and worker.is_alive():
                worker.join(timeout=join_timeout)

        capture_stopped = not capture_thread or not capture_thread.is_alive()
        processing_stopped = (
            not processing_thread or not processing_thread.is_alive()
        )
        with self._worker_lock:
            if self._capture_thread is capture_thread and capture_stopped:
                self._capture_thread = None
            if self._processing_thread is processing_thread and processing_stopped:
                self._processing_thread = None
        if not capture_stopped or not processing_stopped:
            logger.error("Worker threads did not stop within %.1f seconds", join_timeout)
        return capture_stopped and processing_stopped

    def _consume_frame_result(self):
        try:
            frame, alert_detected, metrics = self._result_queue.get_nowait()
        except queue.Empty:
            return
        self._apply_frame_result(frame, alert_detected, metrics)

    def _apply_frame_result(self, frame, alert_detected, metrics):
        try:
            face_detected = metrics.get('face_detected', False)
            eye_openness = metrics.get(
                'eye_openness', self.last_metrics['eye_openness']
            )
            mouth_openness = metrics.get(
                'mouth_openness', self.last_metrics['mouth_openness']
            )
            blink_count = metrics.get('blink_count', self.detector.blink_total)
            yawn_count = metrics.get('yawn_count', self.detector.yawn_total)

            if frame is not None:
                if face_detected:
                    self.last_metrics['eye_openness'] = eye_openness
                    self.last_metrics['mouth_openness'] = mouth_openness
                # Blink and yawn counts are cumulative/independent of immediate face detection
                self.last_metrics['blink_count'] = blink_count
                self.last_metrics['yawn_count'] = yawn_count

            if frame is None or not isinstance(frame, np.ndarray):
                self.status_label.text = 'Lỗi: Không lấy được khung hình'
                self._update_main_screen_metrics()
                return

            if not hasattr(self, '_texture') or self._texture is None or self._texture.size != (frame.shape[1], frame.shape[0]):
                self._texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            self._texture.blit_buffer(cv2.flip(frame, 0).tobytes(), colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = self._texture

            self._update_main_screen_metrics()

            if alert_detected and not self.alert_active:
                if metrics.get('fatigue_detected', False):
                    status_text = 'CẢNH BÁO: Dấu hiệu mệt mỏi!'
                    self.status_label.text = status_text
                    self.alert_active = True
                    self.background_color = [1, 0, 0, 1]
                    self.update_background_color()
                    logging.info(status_text)
                    self.start_fatigue_alert()
                elif metrics.get('drowsiness_detected', False):
                    status_text = 'CẢNH BÁO: Phát hiện buồn ngủ!'
                    self.status_label.text = status_text
                    self.alert_active = True
                    self.background_color = [1, 0, 0, 1]
                    self.update_background_color()
                    logging.info(status_text)
                    self.start_alert()
                elif metrics.get('distraction_detected', False):
                    status_text = 'CẢNH BÁO: Không phát hiện khuôn mặt!'
                    self.status_label.text = status_text
                    self.alert_active = True
                    self.background_color = [1, 0, 0, 1]
                    self.update_background_color()
                    logging.info(status_text)
                    self.start_alert()

            elif not alert_detected and self.alert_active and not self.alert_stop_timer:
                self.alert_stop_timer = Clock.schedule_once(self.stop_alert, self.alert_stop_delay)
            elif not self.alert_active:
                self.status_label.text = 'Trạng thái: Đang giám sát'
                self.background_color = [0, 0, 0, 1]
                self.update_background_color()

        except Exception as e:
            logger.error(f"Lỗi xử lý khung hình: {e}")
            self.status_label.text = 'Lỗi: Xử lý khung hình thất bại'
            self.background_color = [0, 0, 0, 1]
            self.update_background_color()
            self._update_main_screen_metrics()

    def start_monitoring(self, instance):
        if not self.camera_initialized:
            self.status_label.text = 'Lỗi: Camera chưa khởi tạo'
            logging.error("Không thể bắt đầu giám sát: Camera chưa khởi tạo")
            return
        with self._worker_lock:
            workers_running = any(
                worker and worker.is_alive()
                for worker in (self._capture_thread, self._processing_thread)
            )
            if self.is_monitoring or workers_running:
                logging.info("Giám sát đã chạy; bỏ qua yêu cầu bắt đầu lặp")
                return
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                args=(stop_event,),
                daemon=True,
            )
            self._processing_thread = threading.Thread(
                target=self._processing_loop,
                args=(stop_event,),
                daemon=True,
            )
            capture_thread = self._capture_thread
            processing_thread = self._processing_thread
            self.is_monitoring = True

        self.status_label.text = 'Trạng thái: Đang giám sát'
        self.background_color = [0, 0, 0, 1]
        self.update_background_color()
        for q in (self._frame_queue, self._result_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        capture_thread.start()
        processing_thread.start()
        logging.info("Bắt đầu giám sát")

    def stop_monitoring(self, instance):
        self.is_monitoring = False
        workers_stopped = self._stop_worker_threads()
        self.alert_active = False
        self.status_label.text = 'Trạng thái: Đã dừng'
        self.image.texture = None
        if self.alert_sound:
            self.alert_sound.stop()
        if self.fatigue_sound:
            self.fatigue_sound.stop()
        if self.alert_stop_timer:
            self.alert_stop_timer.cancel()
            self.alert_stop_timer = None
        if self.calibration_event:
            self.calibration_event.cancel()
            self.calibration_event = None
        self.background_color = [0, 0, 0, 1]
        self.update_background_color()
        self._update_main_screen_metrics()
        logging.info("Dừng giám sát")
        return workers_stopped

    def calibrate(self, instance):
        if not self.camera_initialized:
            self.status_label.text = 'Lỗi: Camera chưa khởi tạo'
            logging.error("Không thể hiệu chỉnh: Camera chưa khởi tạo")
            return
        if not self.stop_monitoring(instance):
            self.status_label.text = 'Lỗi: Luồng xử lý cũ chưa dừng'
            logging.error("Không thể hiệu chỉnh khi worker cũ vẫn đang chạy")
            return
        self.status_label.text = 'Trạng thái: Đang hiệu chỉnh...'
        self.background_color = [0, 0, 0, 1]
        self.update_background_color()
        logging.info("Bắt đầu hiệu chỉnh")
        self.detector.reset_calibration()
        self.calibration_start_time = Clock.get_time()
        self.calibration_event = Clock.schedule_interval(self.update_calibration, 1.0 / 30.0)

    def update_calibration(self, dt):
        duration = self.app_config.CALIBRATION_DURATION
        elapsed = Clock.get_time() - self.calibration_start_time
        if elapsed >= duration:
            success, new_threshold = self.detector.finalize_calibration()
            self.calibration_event.cancel()
            self.calibration_event = None
            if success:
                self.eye_open_threshold = new_threshold
                main_screen = self._get_main_screen()
                if main_screen is not None:
                    main_screen.metrics_widgets['eye_openness']['bar'].threshold = (
                        new_threshold
                    )
            self.status_label.text = f'Trạng thái: Hiệu chỉnh hoàn tất ( Ngưỡng mắt mới: {new_threshold:.3f})' if success else 'Trạng thái: Hiệu chỉnh thất bại'
            logging.info(
                f"Hiệu chỉnh {'hoàn tất' if success else 'thất bại'}. Ngưỡng mắt mới: {new_threshold:.3f}" if success else "Hiệu chỉnh thất bại")
            self.image.texture = None
            self.background_color = [0, 0, 0, 1]
            self.update_background_color()
            self._update_main_screen_metrics()
            return
        frame, eye_openness = self.detector.process_calibration_frame()
        if frame is not None:
            if not hasattr(self, '_texture') or self._texture is None or self._texture.size != (frame.shape[1], frame.shape[0]):
                self._texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            self._texture.blit_buffer(cv2.flip(frame, 0).tobytes(), colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = self._texture
            remaining = int(duration - elapsed)
            self.status_label.text = (
                f'Đang hiệu chỉnh... {remaining}s '
                f'(Độ mở mắt: {eye_openness:.2f})'
            )
            self._update_main_screen_metrics(eye_override=eye_openness)

    def start_alert(self):
        logging.info("Bắt đầu phát âm thanh cảnh báo")
        self.alert_active = True
        if self.fatigue_sound and self.fatigue_sound.state == 'play':
            self.fatigue_sound.stop()
            logging.info("Dừng âm thanh mệt mỏi để ưu tiên âm thanh cảnh báo")
        if self.alert_sound:
            self.alert_sound.stop()
            self.alert_sound.loop = True
            self.alert_sound.volume = self.settings.alert_volume / 100.0
            self.alert_sound.play()
        self.background_color = [1, 0, 0, 1]
        self.update_background_color()
        if self.alert_stop_timer:
            self.alert_stop_timer.cancel()
            self.alert_stop_timer = None

    def start_fatigue_alert(self):
        logging.info("Bắt đầu phát âm thanh thông báo mệt mỏi")
        if self.alert_sound and self.alert_sound.state == 'play':
            logging.info("Ưu tiên âm thanh cảnh báo alert, bỏ qua âm thanh mệt mỏi")
            return
        self.alert_active = True
        if self.fatigue_sound:
            self.fatigue_sound.stop()
            self.fatigue_sound.loop = False
            self.fatigue_sound.volume = self.settings.alert_volume / 100.0
            self.fatigue_sound.play()
        self.background_color = [1, 0, 0, 1]
        self.update_background_color()
        if self.alert_stop_timer:
            self.alert_stop_timer.cancel()
            self.alert_stop_timer = None

    def stop_alert(self, dt):
        self.alert_active = False
        if self.alert_sound:
            self.alert_sound.stop()
        if self.fatigue_sound:
            self.fatigue_sound.stop()
        self.alert_stop_timer = None
        self.status_label.text = 'Trạng thái: Đang giám sát'
        self.background_color = [0, 0, 0, 1]
        self.update_background_color()
        self._update_main_screen_metrics()

    def on_stop(self):
        logging.info("Dọn dẹp tài nguyên")
        self.is_monitoring = False
        self._stop_worker_threads()
        if self.alert_stop_timer:
            self.alert_stop_timer.cancel()
        if self.alert_sound:
            self.alert_sound.stop()
        if self.fatigue_sound:
            self.fatigue_sound.stop()
        if self.calibration_event:
            self.calibration_event.cancel()
        self.detector.stop_camera()
        self.background_color = [0, 0, 0, 1]
        self.update_background_color()
        self._update_main_screen_metrics()
