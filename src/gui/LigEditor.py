import sys
import os
import math
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QGraphicsScene, QGraphicsView,
    QPushButton, QFileDialog, QMessageBox, QGraphicsEllipseItem, QGraphicsTextItem,
    QHBoxLayout, QComboBox, QGraphicsLineItem, QGraphicsPolygonItem
)
from PySide6.QtGui import QPen, QBrush, QColor, QPolygonF
from PySide6.QtCore import Qt, QPointF, QEvent
from pm4py.objects.petri_net.importer import importer as pnml_importer
from PySide6.QtGui import QPalette


def is_dark_mode() -> bool:
    pal = QApplication.palette()
    if hasattr(pal, "colorScheme"):
        return pal.colorScheme() == Qt.ColorScheme.Dark
    bg = pal.color(QPalette.Window)
    return 0.299*bg.red() + 0.587*bg.green() + 0.114*bg.blue() < 128


class MovableNode(QGraphicsEllipseItem):
    def __init__(self, radius, label, node_id):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        DARK = is_dark_mode()
        self.FG = Qt.white if DARK else Qt.black
        self.BG = QColor("#444") if DARK else QColor("#E0E0E0")
        self.setBrush(QBrush(self.BG))
        self.setPen(QPen(self.FG, 2))
        self.setFlags(
            QGraphicsEllipseItem.ItemIsSelectable |
            QGraphicsEllipseItem.ItemIsMovable |
            QGraphicsEllipseItem.ItemSendsGeometryChanges
        )
        self.label = label
        self.node_id = node_id
        self.text_item = QGraphicsTextItem(label, self)
        self.text_item.setDefaultTextColor(self.FG)
        self.text_item.setPos(-radius, radius + 5)
        self.edges = []

    def add_edge(self, edge):
        self.edges.append(edge)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange:
            for edge in self.edges:
                edge.update_position()
        return super().itemChange(change, value)


class ArrowEdge(QGraphicsLineItem):
    def __init__(self, source_node, target_node):
        super().__init__()
        DARK = is_dark_mode()
        self.FG = Qt.white if DARK else Qt.black
        self.BG = Qt.white if DARK else Qt.black
        self.source = source_node
        self.target = target_node
        self.setPen(QPen(self.FG, 2))
        self.setZValue(-1)
        self.arrow_head = None
        self.setFlags(QGraphicsLineItem.ItemIsSelectable)

        self.source.add_edge(self)
        self.target.add_edge(self)
        self.scene_ref = None
        self.update_position()

    def update_position(self):
        p1 = self.source.scenePos()
        p2 = self.target.scenePos()

        angle = math.atan2(p2.y() - p1.y(), p2.x() - p1.x())
        offset = 20
        p2_offset = QPointF(
            p2.x() - math.cos(angle) * offset,
            p2.y() - math.sin(angle) * offset
        )

        self.setLine(p1.x(), p1.y(), p2_offset.x(), p2_offset.y())
        if self.scene_ref:
            self.update_arrow(self.scene_ref, p1, p2_offset)

    def update_arrow(self, scene, p1, p2):
        if self.arrow_head is not None:
            old_scene = self.arrow_head.scene()
            if old_scene is not None:
                old_scene.removeItem(self.arrow_head)
            self.arrow_head = None

        angle = math.atan2(p2.y() - p1.y(), p2.x() - p1.x())
        arrow_size = 10

        p1_arrow = p2 - QPointF(
            math.cos(angle + math.pi / 6) * arrow_size,
            math.sin(angle + math.pi / 6) * arrow_size
        )
        p2_arrow = p2 - QPointF(
            math.cos(angle - math.pi / 6) * arrow_size,
            math.sin(angle - math.pi / 6) * arrow_size
        )

        polygon = QPolygonF([p2, p1_arrow, p2_arrow])
        self.arrow_head = QGraphicsPolygonItem(polygon)
        self.arrow_head.setBrush(QBrush(self.BG))
        self.arrow_head.setPen(QPen(self.FG))
        self.arrow_head.setFlag(QGraphicsPolygonItem.ItemIsSelectable, True)
        self.arrow_head.setZValue(-1)
        scene.addItem(self.arrow_head)


class LigEditor(QDialog):
    def __init__(self, petri_net_path, save_dir=None):
        super().__init__()
        self.setWindowTitle("LIG editor")
        self.setMinimumSize(600, 600)
        self.save_dir = save_dir
        self.save_path = None

        net, _, _ = pnml_importer.apply(petri_net_path)
        self.transitions = sorted({t.label for t in net.transitions if t.label})

        self.nodes = []
        self.edges = []
        self.node_counter = 0
        self.selected_node = None

        layout = QVBoxLayout(self)
        control_layout = QHBoxLayout()

        self.combo_transitions = QComboBox()
        self.combo_transitions.addItems(self.transitions)
        control_layout.addWidget(self.combo_transitions)

        btn_add_node = QPushButton("Add node")
        btn_add_node.clicked.connect(self.add_node_from_selection)
        control_layout.addWidget(btn_add_node)

        btn_delete_selected = QPushButton("Delete selected")
        btn_delete_selected.clicked.connect(self.delete_selected)
        control_layout.addWidget(btn_delete_selected)

        btn_save = QPushButton("Save LIG")
        btn_save.clicked.connect(self.save_graph)
        control_layout.addWidget(btn_save)

        layout.addLayout(control_layout)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        layout.addWidget(self.view)

        self.scene.selectionChanged.connect(self.node_clicked)
        self.apply_color_scheme()

    def changeEvent(self, event):
        if event.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange):
            self.apply_color_scheme()
        super().changeEvent(event)

    def apply_color_scheme(self):
        dark = is_dark_mode()

        fg = Qt.white if dark else Qt.black
        bg = QColor("#3C3C3C") if dark else QColor("#E0E0E0")

        for node in self.nodes:
            node.setBrush(QBrush(bg))
            node.setPen(QPen(fg, 2))
            node.text_item.setDefaultTextColor(fg)

        for edge in self.edges:
            edge.setPen(QPen(fg, 2))
            if edge.arrow_head:
                edge.arrow_head.setBrush(QBrush(fg))
                edge.arrow_head.setPen(QPen(fg))

        self.scene.update()

    def add_node_from_selection(self):
        label = self.combo_transitions.currentText()
        radius = 20
        spacing = 100
        count = len(self.nodes)
        x = (count % 5) * spacing + 100
        y = (count // 5) * spacing + 100

        node_id = f"n{self.node_counter}"
        self.node_counter += 1

        node_item = MovableNode(radius, label, node_id)
        node_item.setPos(x, y)
        self.scene.addItem(node_item)

        self.nodes.append(node_item)

    def node_clicked(self):
        selected_items = self.scene.selectedItems()
        selected_nodes = [item for item in selected_items if isinstance(item, MovableNode)]

        if len(selected_nodes) != 1:
            self.selected_node = None
            return

        node = selected_nodes[0]

        if self.selected_node is None:
            self.selected_node = node
        else:
            if self.selected_node != node:
                edge = ArrowEdge(self.selected_node, node)
                self.scene.addItem(edge)
                edge.scene_ref = self.scene
                edge.update_position()
                self.edges.append(edge)
            self.selected_node = None

    def delete_selected(self):
        selected_items = self.scene.selectedItems()
        for item in selected_items:
            if isinstance(item, MovableNode):
                for edge in item.edges[:]:
                    if edge.arrow_head:
                        self.scene.removeItem(edge.arrow_head)
                        edge.arrow_head = None
                    self.scene.removeItem(edge)
                    if edge in edge.source.edges:
                        edge.source.edges.remove(edge)
                    if edge in edge.target.edges:
                        edge.target.edges.remove(edge)
                    if edge in self.edges:
                        self.edges.remove(edge)
                self.scene.removeItem(item)
                self.nodes.remove(item)
            elif isinstance(item, ArrowEdge):
                if item.arrow_head:
                    self.scene.removeItem(item.arrow_head)
                    item.arrow_head = None
                self.scene.removeItem(item)
                if item in item.source.edges:
                    item.source.edges.remove(item)
                if item in item.target.edges:
                    item.target.edges.remove(item)
                if item in self.edges:
                    self.edges.remove(item)
            elif isinstance(item, QGraphicsPolygonItem):
                for edge in self.edges[:]:
                    if edge.arrow_head == item:
                        self.scene.removeItem(item)
                        edge.arrow_head = None
                        self.scene.removeItem(edge)
                        if edge in edge.source.edges:
                            edge.source.edges.remove(edge)
                        if edge in edge.target.edges:
                            edge.target.edges.remove(edge)
                        self.edges.remove(edge)
                        break
                else:
                    self.scene.removeItem(item)

    def save_graph(self):
        if not self.nodes:
            QMessageBox.warning(self, "Error", "The graph is empty.")
            return

        existing = [f for f in os.listdir(self.save_dir) if f.startswith("lig_") and f.endswith(".g")]
        next_index = len(existing) + 1
        suggested_filename = f"lig_{next_index}.g"
        suggested_path = os.path.join(self.save_dir, suggested_filename)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save LIG file",
            suggested_path,
            "Graph Files (*.g);;All Files (*)"
        )

        if not path:
            return

        try:
            node_ids = {node.node_id: idx + 1 for idx, node in enumerate(self.nodes)}

            with open(path, "w") as f:
                for node in self.nodes:
                    node_id = node_ids[node.node_id]
                    f.write(f"v {node_id} {node.label}\n")
                for edge in self.edges:
                    src = edge.source
                    tgt = edge.target
                    src_id = node_ids[src.node_id]
                    tgt_id = node_ids[tgt.node_id]
                    f.write(f"d {src_id} {tgt_id} {src.label}__{tgt.label}\n")

            QMessageBox.information(self, "Success", f"LIG saved as:\n{os.path.basename(path)}")
            self.saved_path = path
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossible to save file:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    if len(sys.argv) < 2:
        print("Usage: python LigEditor.py")
        sys.exit(1)

    pnml_path = sys.argv[1]
    editor = LigEditor(pnml_path)
    editor.exec()
    sys.exit(app.exec())