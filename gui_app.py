import os
import sys
import time
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QHBoxLayout,
    QVBoxLayout, QFrame, QListWidget, QListWidgetItem, QMessageBox, QSpacerItem,
    QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt


APP_QSS = """
QWidget {
    background: #0b1220;
    color: #e5e7eb;
    font-family: Segoe UI, Arial;
    font-size: 13px;
}
QLabel#Title {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
}
QLabel#SubTitle {
    color: #a3a3a3;
}
QFrame#Card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
}
QLabel#PanelTitle {
    font-size: 13px;
    font-weight: 700;
    color: #cbd5e1;
}
QLabel#Hint {
    color: #94a3b8;
}
QPushButton {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px;
    padding: 10px 12px;
}
QPushButton:hover {
    background: rgba(255,255,255,0.10);
}
QPushButton:pressed {
    background: rgba(255,255,255,0.14);
}
QPushButton#Primary {
    background: #2563eb;
    border: 1px solid #1d4ed8;
    font-weight: 700;
}
QPushButton#Primary:hover {
    background: #1d4ed8;
}
QPushButton:disabled {
    background: rgba(255,255,255,0.03);
    color: rgba(229,231,235,0.40);
    border: 1px solid rgba(255,255,255,0.06);
}
QListWidget {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 8px;
}
QListWidget::item {
    padding: 8px;
    margin: 2px 0px;
    border-radius: 8px;
}
QListWidget::item:selected {
    background: rgba(37,99,235,0.35);
}
"""

def resource_path(*parts):
    # gui_app.py ile aynı klasör
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, *parts)

def cv_to_qpixmap(cv_img_bgr, max_w=640, max_h=420):
    """OpenCV BGR -> QPixmap (scale with aspect)."""
    img_rgb = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

class YOLOGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Diş Seti - PyQt5 GUI")
        self.setMinimumSize(1200, 720)
        self.setStyleSheet(APP_QSS)

        # state
        self.current_image_path = None
        self.original_bgr = None
        self.tagged_bgr = None
        self.last_pred_summary = None

        # model
        self.model_path = resource_path("best.pt")
        self.model = None
        self.class_names = None

        self._build_ui()
        self._load_model()

    # ---------- UI ----------
    def _build_ui(self):
        # Header
        title = QLabel("YOLOv8 Nesne Tespiti • Diş Fırçası / Diş Macunu")
        title.setObjectName("Title")
        subtitle = QLabel("Adım 1: Görsel seç • Adım 2: Test (Tahmin) • İstersen çıktıyı kaydet")
        subtitle.setObjectName("SubTitle")

        header = QVBoxLayout()
        header.addWidget(title)
        header.addWidget(subtitle)

        # Left panel: Original + Tagged
        self.lbl_original = QLabel("Original Image")
        self.lbl_original.setObjectName("PanelTitle")

        self.img_original = QLabel("Görsel seçilmedi")
        self.img_original.setObjectName("Hint")
        self.img_original.setAlignment(Qt.AlignCenter)
        self.img_original.setMinimumSize(520, 420)
        self.img_original.setStyleSheet("background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px dashed rgba(255,255,255,0.12);")

        left_card = self._card_layout(
            header_widgets=[self.lbl_original],
            body_widgets=[self.img_original],
        )

        self.lbl_tagged = QLabel("Tagged Image (Bounding Boxes)")
        self.lbl_tagged.setObjectName("PanelTitle")

        self.img_tagged = QLabel("Henüz tahmin yok")
        self.img_tagged.setObjectName("Hint")
        self.img_tagged.setAlignment(Qt.AlignCenter)
        self.img_tagged.setMinimumSize(520, 420)
        self.img_tagged.setStyleSheet("background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px dashed rgba(255,255,255,0.12);")

        right_img_card = self._card_layout(
            header_widgets=[self.lbl_tagged],
            body_widgets=[self.img_tagged],
        )

        images_row = QHBoxLayout()
        images_row.setSpacing(14)
        images_row.addLayout(left_card, 1)
        images_row.addLayout(right_img_card, 1)

        # Controls + Results (right side column)
        self.btn_select = QPushButton("Select Image")
        self.btn_select.clicked.connect(self.select_image)

        self.btn_test = QPushButton("Test Image (YOLO)")
        self.btn_test.setObjectName("Primary")
        self.btn_test.clicked.connect(self.run_inference)
        self.btn_test.setEnabled(False)

        self.btn_save = QPushButton("Save Tagged Image")
        self.btn_save.clicked.connect(self.save_tagged_image)
        self.btn_save.setEnabled(False)

        self.status = QLabel("Durum: Hazır")
        self.status.setObjectName("Hint")

        controls_body = QVBoxLayout()
        controls_body.setSpacing(10)
        controls_body.addWidget(self.btn_select)
        controls_body.addWidget(self.btn_test)
        controls_body.addWidget(self.btn_save)
        controls_body.addItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))
        controls_body.addWidget(self.status)

        controls_card = self._card_widget("Kontroller", controls_body)

        self.results_list = QListWidget()
        self.results_list.setMinimumHeight(220)

        self.count_label = QLabel("Tespit: -")
        self.count_label.setObjectName("Hint")

        results_body = QVBoxLayout()
        results_body.addWidget(self.count_label)
        results_body.addWidget(self.results_list)

        results_card = self._card_widget("Sonuçlar", results_body)

        side_col = QVBoxLayout()
        side_col.setSpacing(14)
        side_col.addWidget(controls_card)
        side_col.addWidget(results_card)

        # Main layout
        top = QVBoxLayout()
        top.addLayout(header)
        top.addSpacing(10)

        content = QHBoxLayout()
        content.setSpacing(14)

        left_big = QVBoxLayout()
        left_big.addLayout(images_row)
        content.addLayout(left_big, 3)
        content.addLayout(side_col, 1)

        top.addLayout(content)
        self.setLayout(top)

    def _card_layout(self, header_widgets, body_widgets):
        card = QVBoxLayout()
        frame = QFrame()
        frame.setObjectName("Card")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(14, 14, 14, 14)
        frame_layout.setSpacing(10)

        for w in header_widgets:
            frame_layout.addWidget(w)

        for w in body_widgets:
            frame_layout.addWidget(w)

        card.addWidget(frame)
        return card

    def _card_widget(self, title_text, body_layout: QVBoxLayout):
        title = QLabel(title_text)
        title.setObjectName("PanelTitle")

        frame = QFrame()
        frame.setObjectName("Card")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)
        lay.addWidget(title)
        lay.addLayout(body_layout)
        return frame

    # ---------- Model ----------
    def _load_model(self):
        if not os.path.exists(self.model_path):
            self._set_status(f"best.pt bulunamadı: {self.model_path}", error=True)
            self.btn_test.setEnabled(False)
            return

        try:
            self.model = YOLO(self.model_path)
            self.class_names = self.model.names  # dict or list
            self._set_status("Model yüklendi ✅ (best.pt)")
        except Exception as e:
            self._set_status(f"Model yüklenemedi ❌: {e}", error=True)
            self.model = None
            self.btn_test.setEnabled(False)

    # ---------- Actions ----------
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Görsel Seç",
            "",
            "Image Files (*.png *.jpg *.jpeg)"
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            self._set_status("Görsel okunamadı (cv2.imread None döndü).", error=True)
            return

        self.current_image_path = file_path
        self.original_bgr = img
        self.tagged_bgr = None
        self.last_pred_summary = None

        # show original
        self.img_original.setPixmap(cv_to_qpixmap(self.original_bgr, 560, 420))
        self.img_original.setStyleSheet("background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(255,255,255,0.10);")

        # reset tagged/result
        self.img_tagged.setPixmap(QPixmap())
        self.img_tagged.setText("Henüz tahmin yok")
        self.img_tagged.setStyleSheet("background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px dashed rgba(255,255,255,0.12);")
        self.results_list.clear()
        self.count_label.setText("Tespit: -")

        self.btn_test.setEnabled(self.model is not None)
        self.btn_save.setEnabled(False)
        self._set_status(f"Görsel seçildi ✅ ({os.path.basename(file_path)})")

    def run_inference(self):
        if self.model is None:
            self._set_status("Model yok. best.pt yüklenemedi.", error=True)
            return
        if not self.current_image_path or self.original_bgr is None:
            self._set_status("Önce bir görsel seç.", error=True)
            return

        try:
            self._set_status("Tahmin yapılıyor…")
            t0 = time.time()

            # ultralytics returns list of Results
            results = self.model.predict(
                source=self.current_image_path,
                conf=0.25,
                iou=0.45,
                verbose=False
            )
            res = results[0]

            # get plotted image (BGR)
            plotted = res.plot()  # ndarray (BGR)
            self.tagged_bgr = plotted

            # extract classes + confidences
            # res.boxes.cls: tensor
            # res.boxes.conf: tensor
            class_ids = []
            confs = []
            if res.boxes is not None and res.boxes.cls is not None:
                class_ids = res.boxes.cls.cpu().numpy().astype(int).tolist()
                confs = res.boxes.conf.cpu().numpy().tolist() if res.boxes.conf is not None else []

            # count per class
            counts = Counter(class_ids)
            total = sum(counts.values())

            # update tagged image
            self.img_tagged.setPixmap(cv_to_qpixmap(self.tagged_bgr, 560, 420))
            self.img_tagged.setStyleSheet("background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(255,255,255,0.10);")

            # update result list
            self.results_list.clear()

            # Pretty class name resolver
            def cname(cid):
                if isinstance(self.class_names, dict):
                    return self.class_names.get(cid, str(cid))
                if isinstance(self.class_names, (list, tuple)) and 0 <= cid < len(self.class_names):
                    return self.class_names[cid]
                return str(cid)

            # Summary header
            ms = int((time.time() - t0) * 1000)
            self.count_label.setText(f"Tespit: {total} adet • Süre: {ms} ms")

            # Add per-class lines
            if total == 0:
                item = QListWidgetItem("Hiç nesne tespit edilmedi.")
                self.results_list.addItem(item)
            else:
                for cid, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
                    item = QListWidgetItem(f"{cname(cid)}  →  {cnt} adet")
                    self.results_list.addItem(item)

                # Add detailed detections (optional)
                self.results_list.addItem(QListWidgetItem("—"))
                for i, cid in enumerate(class_ids):
                    c = confs[i] if i < len(confs) else None
                    if c is None:
                        self.results_list.addItem(QListWidgetItem(f"#{i+1} {cname(cid)}"))
                    else:
                        self.results_list.addItem(QListWidgetItem(f"#{i+1} {cname(cid)}  (conf: {c:.2f})"))

            self.btn_save.setEnabled(True)
            self._set_status("Tahmin tamam ✅")
        except Exception as e:
            self._set_status(f"Tahmin hatası ❌: {e}", error=True)

    def save_tagged_image(self):
        if self.tagged_bgr is None:
            self._set_status("Kaydedilecek çıktı yok. Önce Test Image yap.", error=True)
            return

        default_name = "tagged_output.jpg"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Kaydet",
            default_name,
            "JPEG (*.jpg *.jpeg);;PNG (*.png)"
        )
        if not file_path:
            return

        try:
            ok = cv2.imwrite(file_path, self.tagged_bgr)
            if ok:
                self._set_status(f"Kaydedildi ✅ ({file_path})")
            else:
                self._set_status("Kaydetme başarısız (cv2.imwrite false).", error=True)
        except Exception as e:
            self._set_status(f"Kaydetme hatası ❌: {e}", error=True)

    # ---------- Helpers ----------
    def _set_status(self, text, error=False):
        self.status.setText(f"Durum: {text}")
        if error:
            self.status.setStyleSheet("color: #fca5a5;")
        else:
            self.status.setStyleSheet("color: #94a3b8;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = YOLOGui()
    w.show()
    sys.exit(app.exec_())
