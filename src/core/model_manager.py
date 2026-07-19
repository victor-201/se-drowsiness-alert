import os
import urllib.request
import logging
from src.configs.config import Config

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.config = Config()

    def download_dnn_models(self):
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        proto_path = self.config.DNN_PROTOTXT
        model_path = self.config.DNN_CAFFEMODEL
        if not os.path.exists(proto_path):
            logger.info(f"Downloading DNN prototxt to {proto_path}")
            try:
                urllib.request.urlretrieve(proto_url, proto_path)
                logger.info("DNN prototxt downloaded")
            except Exception as e:
                logger.error(f"Failed to download DNN prototxt: {e}")
                raise
        if not os.path.exists(model_path):
            logger.info(f"Downloading DNN caffemodel to {model_path}")
            try:
                urllib.request.urlretrieve(model_url, model_path)
                logger.info("DNN caffemodel downloaded")
            except Exception as e:
                logger.error(f"Failed to download DNN caffemodel: {e}")
                raise
