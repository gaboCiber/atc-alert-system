import sys
import fitz

from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsRectItem
from PyQt5.QtGui import QPixmap, QImage, QPen
from PyQt5.QtCore import Qt, QRectF


class PDFDebugger(QGraphicsView):

    def __init__(self, pdf_path):
        super().__init__()

        self.doc = fitz.open(pdf_path)
        self.page_index = 0

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.rect_item = None
        self.start = None

        self.saved_rect = None   # ← región persistente

        self.load_page()

    def load_page(self):

        self.scene.clear()
        self.rect_item = None

        page = self.doc.load_page(self.page_index)
        pix = page.get_pixmap()

        img = QImage(
            pix.samples,
            pix.width,
            pix.height,
            pix.stride,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(img)
        self.scene.addPixmap(pixmap)

        self.setSceneRect(QRectF(pixmap.rect()))

        # Redibujar región guardada
        if self.saved_rect:
            self.rect_item = QGraphicsRectItem(self.saved_rect)
            self.rect_item.setPen(QPen(Qt.red, 2))
            self.scene.addItem(self.rect_item)

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:

            self.start = self.mapToScene(event.pos())

            if self.rect_item:
                try:
                    self.scene.removeItem(self.rect_item)
                except RuntimeError:
                    pass

            self.rect_item = QGraphicsRectItem()
            self.rect_item.setPen(QPen(Qt.red, 2))
            self.scene.addItem(self.rect_item)

    def mouseMoveEvent(self, event):

        if self.start:

            current = self.mapToScene(event.pos())

            rect = QRectF(self.start, current).normalized()

            self.rect_item.setRect(rect)

    def mouseReleaseEvent(self, event):

        if self.start:

            rect = self.rect_item.rect()

            self.saved_rect = rect   # ← guardar región

            print("\nSelected region:")
            print(f"x0={rect.left():.0f}")
            print(f"y0={rect.top():.0f}")
            print(f"x1={rect.right():.0f}")
            print(f"y1={rect.bottom():.0f}")

            self.start = None

    def keyPressEvent(self, event):

        if event.key() == Qt.Key_Right:

            if self.page_index < len(self.doc) - 1:
                self.page_index += 1
                self.load_page()

        elif event.key() == Qt.Key_Left:

            if self.page_index > 0:
                self.page_index -= 1
                self.load_page()


if __name__ == "__main__":

    app = QApplication(sys.argv)

    viewer = PDFDebugger("docs/ICAO Standard Phraseology.pdf")
    viewer.setWindowTitle("PDF Layout Debugger")

    viewer.resize(900, 1200)
    viewer.show()

    sys.exit(app.exec())