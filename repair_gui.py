import shutil
import sys
import os
import re
from collections import deque, defaultdict
from LigEditor import is_dark_mode

from PySide6.QtGui import QIcon, QPixmap, QColor, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QMessageBox, QCheckBox, QGroupBox,
    QSizePolicy, QToolButton, QComboBox, QDialog, QTabWidget
)
from PySide6.QtCore import Qt, QUrl, QEvent
from PySide6.QtGui import QDesktopServices

from utils import call_big, run_repairing, create_experiment_folder_from_xes
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from LigEditor import LigEditor
from PySide6.QtCore import QObject, Signal, QRunnable, Slot, QThreadPool
from utils import compute_precision


LIGHT_PALETTE = {
    "node_fill":  "#ECEFF1",
    "node_edge":  "#263238",
    "edge":       "#455A64",
    "label":      "#000000",
}

DARK_PALETTE = {
    "node_fill":  "#37474F",
    "node_edge":  "#ECEFF1",
    "edge":       "#CFD8DC",
    "label":      "#ECEFF1",
}

class BigWorker(QRunnable):
    def __init__(self, log_path, model_path, db_name, out_g_file,
                 conformance_path, graph_path, logger, callback):
        super().__init__()
        self.log_path = log_path
        self.model_path = model_path
        self.db_name = db_name
        self.out_g_file = out_g_file
        self.conformance_path = conformance_path
        self.graph_path = graph_path
        self.logger = logger
        self.callback = callback  # funzione da eseguire quando BIG ha finito

    def run(self):
        result = call_big(
            self.log_path, self.model_path, self.db_name,
            self.out_g_file, self.conformance_path,
            self.graph_path, self.logger
        )
        self.callback(result)  # Quando BIG ha finito


class AdaptiveGraphLabel(QLabel):

    def __init__(self, g_path = None, *args, **kw):
        super().__init__(*args, **kw)
        self._g_path = g_path
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if hasattr(QGuiApplication.instance(), "paletteChanged"):
            QGuiApplication.instance().paletteChanged.connect(self._redraw)
        if g_path:
            self._redraw()

    def setGraph(self, g_path):
        self._g_path = g_path
        self._redraw()

    def resizeEvent(self, ev):
        if self._g_path:
            self._redraw()
        super().resizeEvent(ev)

    def _redraw(self):
        w, h = max(self.width(), 1), max(self.height(), 1)

        pm = self._pixmap_from_g(self._g_path, w, h)
        if pm:
            super().setPixmap(pm)
        else:
            self.setText("No graph available")

    def event(self, ev):
        if ev.type() in (QEvent.ApplicationPaletteChange,
                         QEvent.ThemeChange):
            self._redraw()
        return super().event(ev)

    @staticmethod
    def _pixmap_from_g(g_path: str, canvas_w: int, canvas_h: int):

        DARK = is_dark_mode()
        FG = '#FFFFFF' if DARK else '#000000'
        font_color = '#FFFFFF' if not DARK else '#000000'


        import tempfile, networkx as nx, matplotlib.pyplot as plt, os, math
        if not g_path or not os.path.exists(g_path):
            return None
        G = nx.DiGraph()
        with open(g_path, "r", encoding="utf-8") as f:
            for l in f:
                if l.startswith("v "):
                    _, vid, lab = l.split(maxsplit=2)
                    G.add_node(int(vid), label=lab.strip())
                elif l.startswith(("e ", "d ")):
                    _, s, d, lab = l.split(maxsplit=3)
                    G.add_edge(int(s), int(d), label=lab.strip())

        if G.number_of_nodes() == 0:
            return None

        try:
            layers = list(nx.topological_generations(G))
            pos = {}
            for x, layer in enumerate(layers):
                step_y = 1 / (len(layer) + 1)
                for i, node in enumerate(layer, start=1):
                    pos[node] = (x, 1 - i * step_y)
            max_x = max(p[0] for p in pos.values())
            for k, (x, y) in pos.items():
                pos[k] = (x / (max_x if max_x else 1), y)
        except nx.NetworkXUnfeasible:
            pos = nx.kamada_kawai_layout(G)
            xs, ys = zip(*pos.values())
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            for k, (x, y) in pos.items():
                pos[k] = ((x - min_x) / (max_x - min_x or 1),
                          (y - min_y) / (max_y - min_y or 1))


        plt.figure(figsize=((canvas_w-50) / 100, (canvas_h) / 100), dpi=100)
        nx.draw_networkx_nodes(G, pos, node_size=400,
                               node_color='#FFFFFF', edgecolors=FG)
        labels = {n: d['label'] for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(
            G,
            pos,
            labels,
            font_size=5,
            font_color=FG,
            verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
        )

        nx.draw_networkx_edges(
            G, pos,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=10,
            edge_color=FG,
            width=1,
            connectionstyle="arc3,rad=0")
        plt.axis('off')
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(tmp.name, bbox_inches='tight', dpi=120, transparent=True)
        plt.close()
        return QPixmap(tmp.name)




class WorkerSignals(QObject):
    finished = Signal(str)
    error    = Signal(Exception)

class RepairWorker(QRunnable):
    def __init__(self, input_data, folder_path, base_name, logger):
        super().__init__()
        self.input_data  = input_data
        self.folder_path = folder_path
        self.base_name   = base_name
        self.logger      = logger
        self.signals     = WorkerSignals()

    @Slot()
    def run(self):
        import sys, traceback
        original_stdout = sys.stdout
        sys.stdout = self.logger

        try:
            print("Starting repair...", flush=True)
            save_path = run_repairing(
                self.input_data, self.folder_path, self.base_name
            )
            print(f"Repair completed!", flush=True)
            self.signals.finished.emit(save_path)

        except KeyError:
            self.signals.error.emit("The behavior represented by your LIG is not embedded in any trace")
        except IndexError:
            self.signals.error.emit("The behavior represented by your LIG is already represented by the model")
        except ValueError:
            self.signals.error.emit("The behavior represented by your LIG is already represented by the model")
        except Exception as e:
            self.signals.error.emit(e)
        finally:
            sys.stdout = original_stdout


class FileInputField(QWidget):
    def __init__(self, label, expected_ext, tooltip_text="", extra_button=None):

        DARK = is_dark_mode()
        self.FG = '#FFFFFF' if DARK else '#000000'
        super().__init__()
        self.label_text = label
        self.expected_ext = expected_ext

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label_row = QHBoxLayout()
        label_widget = QLabel(f"{label} (*.{expected_ext})")
        label_row.addWidget(label_widget)

        if tooltip_text:
            help_btn = QToolButton()
            help_btn.setIcon(QIcon(os.path.join("icons", "information.png")))
            help_btn.setToolTip(tooltip_text)
            help_btn.setStyleSheet("border: none; color: "+self.FG+";")
            help_btn.setCursor(Qt.WhatsThisCursor)
            label_row.addWidget(help_btn)

        label_row.addStretch()
        layout.addLayout(label_row)

        row = QHBoxLayout()
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("Select a file...")
        self.line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.line_edit.setMinimumWidth(300)
        self.line_edit.setFixedHeight(24)
        row.setAlignment(Qt.AlignLeft)

        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self.browse_file)

        row.addWidget(self.line_edit)
        row.addWidget(browse_btn)


        layout.addLayout(row)

        if extra_button:
            extra_btn_row = QHBoxLayout()
            extra_btn_row.setContentsMargins(0, 0, 0, 0)
            extra_btn_row.setSpacing(0)
            extra_btn_row.addStretch()

            extra_btn = QPushButton(extra_button["text"])
            extra_btn.setFixedWidth(80)
            extra_btn.clicked.connect(extra_button["callback"])
            extra_btn_row.addWidget(extra_btn)

            layout.addLayout(extra_btn_row)

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if path:
            self.line_edit.setText(path)

    def get_path(self):
        return self.line_edit.text().strip()

    def is_valid(self):
        path = self.get_path()
        return path and os.path.exists(path) and os.path.splitext(path)[1].lower() == f".{self.expected_ext.lower()}"


class QTextEditLogger(QObject):
    write_signal = Signal(str)

    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        self.write_signal.connect(self._append_text)

    def write(self, text):
        self.write_signal.emit(str(text))

    def flush(self):
        pass

    def _append_text(self, text):
        self.text_edit.append(text)
        self.text_edit.verticalScrollBar().setValue(self.text_edit.verticalScrollBar().maximum())


class RepairToolGUI(QWidget):
    def __init__(self):
        super().__init__()

        DARK = is_dark_mode()
        self.FG = '#FFFFFF' if DARK else '#000000'

        self._last_dataset = None
        self._last_lig_folder = None
        self.setWindowTitle("ReLIGn GUI")
        fast_mode_help = QToolButton()
        fast_mode_help.setIcon(QIcon(os.path.join("icons", "information.png")))
        fast_mode_help.setToolTip("This tool implements repairing as explained in ")
        self.setMinimumSize(750, 890)
        self.setMaximumSize(950, 900)
        self.threadpool = QThreadPool()
        self.setup_ui()



    def setup_ui(self):

        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)


        self.tabs = QTabWidget(self)
        main_layout.addWidget(self.tabs)

        self.main_tab = QWidget()
        self.eval_tab = QWidget()

        self.tabs.addTab(self.main_tab, "Repair")
        self.tabs.addTab(self.eval_tab, "Evaluation")

        self.setup_main_tab_ui(self.main_tab)
        self.setup_eval_tab_ui(self.eval_tab)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.log_box.setMaximumHeight(150)

        main_layout.addWidget(self.log_box)

        self.logger = QTextEditLogger(self.log_box)
        sys.stdout = self.logger
        sys.stderr = self.logger

        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_path = os.path.join(os.path.dirname(__file__), "icons", "logo.png")

        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            logo_label.setPixmap(logo_pixmap.scaledToHeight(30, Qt.SmoothTransformation))

            logo_container = QHBoxLayout()
            logo_container.addStretch()
            logo_container.addWidget(logo_label)
            logo_container.addStretch()

            self.layout().addLayout(logo_container)


    def setup_main_tab_ui(self, tab):
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title = QLabel("")
        title.setStyleSheet("font-size: 18pt; font-weight: bold;")
        main_layout.addWidget(title, alignment=Qt.AlignHCenter)

        main_layout.addWidget(QLabel("Select Dataset"), alignment=Qt.AlignTop)

        dataset_layout = QHBoxLayout()
        self.dataset_selector = QComboBox()
        self.dataset_selector.addItem("-- Select dataset --")
        self.dataset_selector.addItems(self.get_experiment_folders())
        self.dataset_selector.currentTextChanged.connect(self.update_result_selector)
        dataset_layout.addWidget(self.dataset_selector)
        main_layout.addLayout(dataset_layout)

        input_group = QGroupBox("Input Files")
        input_layout = QVBoxLayout()
        self.input_net = FileInputField("Petri Net", "pnml", tooltip_text="A Petri net model in PNML format.")
        self.input_log = FileInputField("Event Log", "xes", tooltip_text="An event log in XES format.")
        self.input_igs = FileInputField("Instance Graphs (optional)", "g",
                                        tooltip_text="Optional IGs file; if not provided, BIG will run.")
        self.input_lig = FileInputField(
            "LIG Graph", "g",
            tooltip_text="The Local Instance Graph that represents the high-level anomalous behavior.",
            extra_button={"text": "Create", "callback": self.open_lig_editor}
        )
        for widget in [self.input_net, self.input_log, self.input_igs, self.input_lig]:
            input_layout.addWidget(widget)
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout()
        self.fast_mode = QCheckBox("Fast Mode")
        self.fast_mode.setChecked(True)
        fast_mode_help = QToolButton()
        fast_mode_help.setIcon(QIcon(os.path.join("icons","information.png")))
        fast_mode_help.setToolTip("If enabled, skips some expensive computations to improve speed.")
        fast_mode_help.setCursor(Qt.WhatsThisCursor)
        fast_mode_help.setStyleSheet("border: none; padding: 0px;")
        fast_mode_row = QHBoxLayout()
        fast_mode_row.addWidget(self.fast_mode)
        fast_mode_row.addWidget(fast_mode_help)
        fast_mode_row.addStretch()
        options_layout.addLayout(fast_mode_row)
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)

        run_btn = QPushButton("Run Repair")
        run_btn.setStyleSheet("padding: 8px 20px; font-weight: bold;")
        main_layout.addWidget(run_btn, alignment=Qt.AlignHCenter)
        run_btn.clicked.connect(self.run_repair)


    def setup_eval_tab_ui(self, tab):
        layout = QVBoxLayout(tab)

        self.eval_dataset_selector = QComboBox()
        self.eval_dataset_selector.addItem("-- Select dataset --")
        self.eval_dataset_selector.addItems(self.get_experiment_folders())
        self.eval_dataset_selector.currentTextChanged.connect(self.update_lig_selector)
        self.eval_dataset_selector.currentTextChanged.connect(self.display_lig_results)


        self.lig_selector = QComboBox()
        self.lig_selector.addItem("-- Select LIG folder --")
        self.lig_selector.currentTextChanged.connect(self.display_lig_results)

        layout.addWidget(QLabel("Select Dataset"))
        layout.addWidget(self.eval_dataset_selector)
        layout.addWidget(QLabel("Select LIG Folder"))
        layout.addWidget(self.lig_selector)

        metrics_row = QHBoxLayout()

        left_col = QVBoxLayout()

        label_left = QLabel("Evaluation original net:")
        label_left.setAlignment(Qt.AlignHCenter)
        label_left.setStyleSheet("font-size: 12pt;")
        left_col.addWidget(label_left)

        self.metrics_text_original = QTextEdit()
        self.metrics_text_original.setReadOnly(True)
        self.metrics_text_original.setFixedHeight(85)
        self.metrics_text_original.setSizePolicy(QSizePolicy.Expanding,
                                                 QSizePolicy.Fixed)
        left_col.addWidget(self.metrics_text_original)

        right_col = QVBoxLayout()

        label_right = QLabel("Evaluation repaired net:")
        label_right.setAlignment(Qt.AlignHCenter)
        label_right.setStyleSheet("font-size: 12pt;")
        right_col.addWidget(label_right)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setFixedHeight(85)
        self.metrics_text.setSizePolicy(QSizePolicy.Expanding,
                                        QSizePolicy.Fixed)
        right_col.addWidget(self.metrics_text)

        metrics_row.addLayout(left_col)
        metrics_row.addLayout(right_col)

        layout.addLayout(metrics_row)

        lig_label = QLabel("LIG selected for repairing:")
        lig_label.setStyleSheet("font-size: 14pt;")
        layout.addWidget(lig_label)

        self.graph_label = AdaptiveGraphLabel()
        self.graph_label.setMinimumSize(700, 240)
        layout.addWidget(self.graph_label)


        self.btn_view_net = QPushButton("View repaired net")
        self.btn_export_net = QPushButton("Export repaired net")
        self.btn_view_log = QPushButton("View output log")

        self.btn_view_net.clicked.connect(self.open_net_image)
        self.btn_view_log.clicked.connect(self.open_evaluation_log)
        self.btn_export_net.clicked.connect(self.export_pnml_file)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_view_net)
        btn_row.addWidget(self.btn_export_net)
        btn_row.addWidget(self.btn_view_log)
        layout.addLayout(btn_row)

    def open_net_image(self):
        dataset = self.eval_dataset_selector.currentText()
        if dataset.startswith("--"):
            self.metrics_text_original.clear()
            self.metrics_text.clear()
            return
        lig_folder = self.lig_selector.currentText()
        if dataset.startswith("--") or lig_folder.startswith("--"):
            return

        path = os.path.join("experiments", dataset, lig_folder, "repaired_petriNet.jpg")
        print(path)
        if os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(path)))
        else:
            self.show_error("Petri net not found.")

    def refresh_all_dataset_selectors(self):
        folders = self.get_experiment_folders()


        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()
        self.dataset_selector.addItem("-- Select dataset --")
        self.dataset_selector.addItems(folders)
        self.dataset_selector.blockSignals(False)



    def open_evaluation_log(self):
        dataset = self.eval_dataset_selector.currentText()
        lig_folder = self.lig_selector.currentText()
        if dataset.startswith("--") or lig_folder.startswith("--"):
            return

        log_path = os.path.join("experiments", dataset, lig_folder, f"output_{lig_folder}.txt")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                text = f.read()

            dialog = QDialog(self)
            dialog.setWindowTitle("output Log")
            dialog.resize(600, 400)
            layout = QVBoxLayout(dialog)

            text_area = QTextEdit()
            text_area.setReadOnly(True)
            text_area.setText(text)
            layout.addWidget(text_area)

            dialog.exec()
        else:
            self.show_error("Event log not found.")

    def export_pnml_file(self):
        dataset = self.eval_dataset_selector.currentText()
        lig_folder = self.lig_selector.currentText()
        if dataset.startswith("--") or lig_folder.startswith("--"):
            return

        input_path = os.path.join("experiments", dataset, lig_folder, f"repaired_petriNet.pnml")
        if not os.path.exists(input_path):
            self.show_error("File not found.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Export Petri net as...", f"net_repaired_{lig_folder}.pnml",
                                                   "PNML files (*.pnml)")
        if save_path:
            try:
                with open(input_path, "rb") as f_src, open(save_path, "wb") as f_dest:
                    f_dest.write(f_src.read())
                QMessageBox.information(self, "Export completed", "Petri net exported correctly!")
            except Exception as e:
                self.show_error(f"Error exporting Petri net:\n{e}")

    def _parse_metrics_file(self, path: str) -> str:
        if not os.path.exists(path):
            return ""

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            parsed = {}
            for line in lines:
                if "Fitness:" in line:
                    fitness_dict = eval(re.search(r"\{.*\}", line).group())
                    parsed["Fitness"] = fitness_dict
                elif ":" in line:
                    key, value = line.split(":", 1)
                    parsed[key.strip()] = float(value.strip())

            html = ""
            if "Fitness" in parsed:
                html += f"<b>Fitness:</b> {parsed['Fitness'].get('averageFitness', 0) * 100:.4f}%<br>"
            for key in ("Precision", "Generalization", "Simplicity"):
                if key in parsed:
                    html += f"<b>{key}:</b> {parsed[key] * 100:.4f}%<br>"
            return html
        except Exception as exc:
            return f"<i>Error parsing metrics: {exc}</i>"

    def display_lig_results(self, lig_folder):
        dataset = self.eval_dataset_selector.currentText()
        if dataset.__contains__("--"):
            self.metrics_text.clear()
            self.metrics_text_original.clear()
            self.graph_label.clear()

        l = self.lig_selector.currentText()
        if l.__contains__("--") or l=='':
            self.metrics_text.clear()
            self.graph_label.clear()


        folder_path = os.path.join("experiments", dataset, lig_folder)
        if lig_folder != '' and not lig_folder.__contains__('--'):
            eval_repaired = os.path.join(folder_path, f"output_{lig_folder}_evaluation.txt")
            if os.path.exists(eval_repaired):
                html_rep = self._parse_metrics_file(eval_repaired or '')
                self.metrics_text.setHtml(html_rep)
        else:
            return
        if dataset != '' and not dataset.__contains__('--'):
            eval_original = os.path.join("experiments", dataset, "output__evaluation.txt")
            if os.path.exists(eval_original):
                html_org = self._parse_metrics_file(eval_original or '')
                self.metrics_text_original.setHtml(html_org)
        else:
            return

        if lig_folder != '':
            if os.path.exists(folder_path):
                lig_g = next((os.path.join(folder_path, f)
                      for f in os.listdir(folder_path)
                      if f == "lig.g" and not f.startswith("repaired")), None)
                if lig_g is not None:
                    if os.path.exists(lig_g):
                        self.graph_label.setGraph(lig_g)
                else:
                    self.graph_label.setGraph("")



    def update_lig_selector(self, dataset_name):
        self.lig_selector.clear()
        self.lig_selector.addItem("-- Select LIG folder --")

        base_path = os.path.join("experiments", dataset_name)
        if not os.path.exists(base_path):
            return

        folders = [f for f in os.listdir(base_path)
                   if re.match(r'^lig_\d+$', f) and os.path.isdir(os.path.join(base_path, f))]
        for folder in sorted(folders):
            self.lig_selector.addItem(folder)

    def log(self, msg):
        self.log_box.append(str(msg))

    def show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)

    def get_experiment_folders(self):
        base_path = "experiments"
        if not os.path.exists(base_path):
            return []
        return sorted([
            name for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name))
        ])

    def update_result_selector(self, dataset_name):
        if dataset_name.startswith("--"):
            return

        folder = os.path.join("experiments", dataset_name)
        if os.path.exists(folder):
            files = os.listdir(folder)
            xes_file = next((f for f in files if f == dataset_name+".xes"), None)
            pnml_file = next((f for f in files if f == dataset_name+"_petriNet.pnml"), None)
            g_file = next((f for f in files if f == dataset_name+".g"), None)

            if xes_file:
                self.input_log.line_edit.setText(os.path.join(folder, xes_file))
            if pnml_file:
                self.input_net.line_edit.setText(os.path.join(folder, pnml_file))
            if g_file:
                self.input_igs.line_edit.setText(os.path.join(folder, g_file))


    def show_results(self):
        dataset = self.dataset_selector.currentText()
        output_file = self.output_selector.currentText()
        if not dataset or not output_file:
            return

        output_path = os.path.join("experiments", dataset, output_file)
        try:
            with open(output_path, "r") as f:
                text = f.read()
        except Exception as e:
            self.show_error(f"It is impossible to read file:\n{str(e)}")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Results: {output_file}")
        dialog.resize(600, 400)
        layout = QVBoxLayout(dialog)

        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setText(text)
        layout.addWidget(text_area)

        match = re.search(r"output_(\d+)", output_file)
        if match:
            idx = match.group(1)
            pnml_file = f"net_repaired_{idx}.pnml"
            pnml_path = os.path.join("experiments", dataset, pnml_file)
            if os.path.exists(pnml_path):
                btn_open = QPushButton(f"Open repaired net ({pnml_file})")
                btn_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(pnml_path)))
                layout.addWidget(btn_open)

        dialog.exec()

    def open_lig_editor(self):
        if not self.input_net.is_valid():
            self.show_error("First load a valid Petri net (.pnml)")
            return

        dataset = self.dataset_selector.currentText()

        if dataset.startswith("--"):
            dataset = os.path.basename(self.input_net.get_path()).split("_")[0]

        dataset_path = os.path.join("experiments", dataset)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        editor = LigEditor(self.input_net.get_path(), dataset_path)
        if editor.exec() == QDialog.Accepted:
            if hasattr(editor, "saved_path") and editor.saved_path:
                self.input_lig.line_edit.setText(editor.saved_path)

    def start_repair_worker(self):
        input_data = {
            'net': self.input_net.get_path(),
            'log': self.input_log.get_path(),
            'LIG': self.input_lig.get_path(),
            'igs': self.input_igs.get_path(),
            'fast_mode': self.fast_mode.isChecked()
        }

        try:
            self.log("Creating experiment folder...")
            folder_path, base_name = create_experiment_folder_from_xes(
                'experiments', input_data['log'], input_data['net'],
                input_data['igs'], input_data['LIG']
            )

            dataset_name = os.path.basename(os.path.dirname(folder_path))
            self._last_dataset = dataset_name

            worker = RepairWorker(input_data, folder_path, base_name, self.logger)

            lig_folder = os.path.basename(folder_path)
            self._last_lig_folder = lig_folder

            worker.signals.finished.connect(self.on_repair_finished)
            worker.signals.error.connect(self.on_repair_error)

            self.threadpool.start(worker)
            self.refresh_all_dataset_selectors()

        except Exception as e:
            self.show_error(f"Repair error: {e}")

    def on_big_finished(self, g_path):
        self.log(f"BIG completato: {g_path}")
        self.input_igs.line_edit.setText(g_path)
        self.start_repair_worker()


    def run_repair(self):
        required_fields = [self.input_net, self.input_log, self.input_lig]

        for field in required_fields:
            if not field.is_valid():
                self.show_error(f"Missing or invalid file: {field.label_text}")
                return
            if len(field.get_path()) > 300:
                self.show_error(f"Path too long for: {field.label_text} (limit: 300 characters)")
                return
            try:
                self.validate_lig()
            except ValueError as e:
                self.show_error(str(e))
                return

        if not self.validate_compatibility():
            return

        self.log("Starting repair process...")

        igs_path = self.input_igs.get_path().strip()
        if not igs_path:
            xes_base = os.path.splitext(os.path.basename(self.input_log.get_path()))[0]
            folder_name = f"{xes_base}"
            folder_path = os.path.join('experiments', folder_name)
            big_path = os.path.join(folder_path, "big")
            out_g_file = os.path.join(big_path, xes_base + '.g')

            if not os.path.exists(big_path):
                os.makedirs(big_path)

            self.log("IGS not provided – running BIG creation ...")

            try:
                # calcola l'allineamento solo se serve
                if 'alignment.csv' not in os.listdir(big_path):
                    compute_precision(self.input_log.get_path(), self.input_net.get_path())
                    shutil.copyfile('alignment.csv', os.path.join(big_path, 'alignment.csv'))
                    os.remove('alignment.csv')

                # avvia comunque BIG, sempre
                worker = BigWorker(
                    self.input_log.get_path(),
                    self.input_net.get_path(),
                    xes_base,
                    out_g_file,
                    big_path,
                    big_path,
                    self.logger,
                    callback=self.on_big_finished
                )
                self.threadpool.start(worker)
                return

            except Exception as e:
                self.show_error(f"BIG failed: {e}")
                check_empty = [o for o in os.listdir(folder_path) if o != '.DS_Store']
                if check_empty == ['big']:
                    shutil.rmtree(folder_path)
                return

        input_data = {
            'net': self.input_net.get_path(),
            'log': self.input_log.get_path(),
            'LIG': self.input_lig.get_path(),
            'igs': self.input_igs.get_path() or None,
            'fast_mode': self.fast_mode.isChecked()
        }

        try:
            self.log("Creating experiment folder...")
            folder_path, base_name = create_experiment_folder_from_xes(
                'experiments', input_data['log'], input_data['net'],
                input_data['igs'], input_data['LIG']
            )

            dataset_name = os.path.basename(os.path.dirname(folder_path))

            self._last_dataset = dataset_name

            worker = RepairWorker(input_data, folder_path, base_name, self.logger)

            lig_folder = os.path.basename(folder_path)
            self._last_lig_folder = lig_folder

            worker.signals.finished.connect(self.on_repair_finished)
            worker.signals.error.connect(self.on_repair_error)



            self.threadpool.start(worker)
            self.refresh_all_dataset_selectors()

        except KeyError:
            self.log(f"Error: The behavior represented by your LIG is not embedded in any trace")
            self.show_error("The behavior represented by your LIG is not embedded in any trace")
        except IndexError:
            self.log(f"Error: The behavior represented by your LIG is already represented by the model")
            self.show_error("The behavior represented by your LIG is already represented by the model")
        except ValueError:
            self.log(f"Error: The behavior represented by your LIG is already represented by the model")
            self.show_error("The behavior represented by your LIG is already represented by the model")
        except Exception as e:
            self.log(f"Error")
            self.show_error(e)

    def on_repair_finished(self, save_path):

        self.refresh_all_dataset_selectors()

        folder = os.path.dirname(save_path)
        lig_folder = os.path.basename(folder)

        self._last_lig_folder = save_path

        self.eval_dataset_selector.setCurrentText(lig_folder)
        self.update_lig_selector(lig_folder)
        self.lig_selector.setCurrentText(os.path.basename(save_path))
        self.display_lig_results(lig_folder)
        self.tabs.setCurrentWidget(self.eval_tab)



    def on_repair_error(self, exc: Exception):
        if isinstance(exc, KeyError):
            self.show_error("The behavior represented by your LIG is not embedded in any trace")
        elif isinstance(exc, ValueError):
            self.show_error("The behavior represented by your LIG is already represented by the model")
        else:
            self.show_error(str(exc))


    def validate_lig(self) -> bool:
        lig_path = self.input_lig.get_path()

        if not (lig_path and os.path.exists(lig_path) and lig_path.lower().endswith(".g")):
            raise ValueError("LIG file with incompatible format, please provide *.g")

        vertex_pat = re.compile(r"^v\s+(\d+)\s+(\S+)$")
        edge_pat = re.compile(r"^[de]\s+(\d+)\s+(\d+)\s+(\S+)$")

        nodes: dict[int, str] = {}
        adj: dict[int, set[int]] = defaultdict(set)

        with open(lig_path, "r", encoding="utf-8") as f:
            for n, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue

                if m := vertex_pat.match(line):
                    vid, label = int(m.group(1)), m.group(2)
                    if vid in nodes:
                        raise ValueError(f"Line {n}: vertex {vid} duplicated in LIG.")
                    nodes[vid] = label

                elif m := edge_pat.match(line):
                    src, dst = int(m.group(1)), int(m.group(2))
                    if src not in nodes or dst not in nodes:
                        raise ValueError(f"Line {n}: arc ({src}->{dst}) points to unexistent vertex.")
                    adj[src].add(dst)
                    adj[dst].add(src)

                else:
                    raise ValueError(f"Line {n}: not recognizes format → “{line}.”")

        if not nodes:
            raise ValueError("Empty LIG.")

        visited = set()
        start = next(iter(nodes))
        queue = deque([start])

        while queue:
            v = queue.popleft()
            if v in visited:
                continue
            visited.add(v)
            queue.extend(adj[v])

        if len(visited) != len(nodes):
            raise ValueError("The LIG must be a connected graph.")

        return True


    def validate_compatibility(self):
        try:
            net_path = self.input_net.get_path()
            log_path = self.input_log.get_path()

            net, im, fm = pnml_importer.apply(net_path)
            net_transitions = {t.label for t in net.transitions if t.label}

            if not net_transitions:
                self.show_error("The Petri net has no labeled transitions.")
                return False

            log = xes_importer.apply(log_path)
            log_events = {e["concept:name"] for trace in log for e in trace if "concept:name" in e}

            if not log_events:
                self.show_error("The event log contains no events with concept:name.")
                return False

            shared = net_transitions & log_events
            if not shared:
                self.show_error("The Petri net and the event log do not share any activity.")
                return False

            return True

        except Exception as e:
            self.show_error(f"Compatibility check failed:\n{str(e)}")
            return False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RepairToolGUI()
    window.show()
    sys.exit(app.exec())
