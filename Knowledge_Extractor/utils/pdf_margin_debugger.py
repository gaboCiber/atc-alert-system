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

            # Obtener dimensiones de la página actual
            page = self.doc.load_page(self.page_index)
            page_width = page.rect.width
            page_height = page.rect.height

            # Coordenadas del debugger (fitz)
            x0 = rect.left()
            y0 = rect.top()
            x1 = rect.right()
            y1 = rect.bottom()

            # Convertir a márgenes para kex (pypdf)
            left = x0
            right = page_width - x1
            top = y0  # distancia desde arriba
            bottom = page_height - y1  # distancia desde abajo

            print("\n=== Coordenadas para kex ===")
            print(f"margins = ({left:.0f}, {bottom:.0f}, {right:.0f}, {top:.0f})")
            print(f"// kex(doc_path, doc_dir, model, margins=({left:.0f}, {bottom:.0f}, {right:.0f}, {top:.0f}))")
            print("\nCoordenadas originales (debugger):")
            print(f"x0={x0:.0f}, y0={y0:.0f}, x1={x1:.0f}, y1={y1:.0f}")
            print(f"Dimensiones página: {page_width:.0f} x {page_height:.0f}")

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