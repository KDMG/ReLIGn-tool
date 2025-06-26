import re
import subprocess
import threading
from os.path import join as join
from sys import stdout

from ProcessRepairing.scripts import Repairing as repairing
import os
import hashlib
import shutil

def read_g(path):
    pass


def process_big(log, net):
    pass


def get_hash_from_files(file_paths):
    hasher = hashlib.sha256()
    for path in sorted(file_paths):
        with open(path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
    return hasher.hexdigest()[:7]

def get_next_lig_filename(folder_path):
    pattern = re.compile(r"lig_(\d+)$")
    max_index = 0

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index

    next_index = max_index + 1
    return f"lig_{next_index}"


def create_experiment_folder_from_xes(base_dir, xes_file, net_file, g_file, lig_file):
    xes_base = os.path.splitext(os.path.basename(xes_file))[0]

    folder_name = f"{xes_base}"
    folder_path = os.path.join(base_dir, folder_name)

    if (not os.path.exists(join(folder_path + folder_name+'.xes')) or not os.path.exists(join(folder_path + xes_base + '_petriNet.pnml'))
            or os.path.exists(join(folder_path, xes_base + '.g')) or os.path.exists(join(folder_path, 'lig.g'))):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        try:
            shutil.copy(xes_file, join(folder_path, folder_name+'.xes'))
        except OSError:
            pass
        try:
            shutil.copy(net_file, join(folder_path, xes_base + '_petriNet.pnml'))
        except OSError:
            pass
        if g_file:
            try:
                shutil.copy(g_file, join(folder_path, xes_base + '.g'))
            except OSError:
                pass
        try:
            shutil.copy(lig_file, join(folder_path, 'subelements.txt'))
        except OSError:
            pass
    return folder_path, xes_base


def run_repairing(input_data, folder_path, base_name):

    new_lig = get_next_lig_filename(folder_path)
    save_path = os.path.join(folder_path, new_lig)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    try:
        shutil.copy(input_data['LIG'], join(folder_path, new_lig, 'lig.g'))
    except OSError:
        pass
    try:
        repairing.main(input_data, folder_path, base_name, '1', os.path.join(folder_path, new_lig))
        return save_path
    except Exception as e:
        shutil.rmtree(save_path)
        raise e



def run_cmd_stream(cmd: list[str], logger, on_finish=None) -> threading.Thread:

    def _worker():
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        ) as p:
            for line in p.stdout:
                logger.write(line)
            p.wait()

        if on_finish is not None:
            on_finish()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t

def run_cmd_stream(cmd: list[str], logger, on_finish=None) -> threading.Thread:
    """
    Esegue `cmd`, scrive lo stdout riga-per-riga in `logger` e,
    quando il processo termina, chiama `on_finish()` (se presente).
    Ritorna il thread creato (non bloccante per la GUI).
    """
    def _worker():
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        ) as p:
            for line in p.stdout:
                logger.write(line)
            p.wait()

        if on_finish is not None:         # lancio della fase successiva
            on_finish()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t

def run_cmd_stream_sync(cmd: list[str], logger):
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    ) as p:
        for line in p.stdout:
            logger.write(line)
        p.wait()


def call_big(log_path, model_path, db_name, out_g_file,
             conformance_path, graph_path, logger):

    big = 'BIGfiles'
    init_cmd = ['java', '-jar', join(big, 'IGInitializer.jar'),
                '0', '150000', db_name, '1', '100000000',
                out_g_file, log_path, model_path,
                conformance_path, graph_path]

    rules_cmd = ['java', '-jar', join(big, 'InstanceGraphRules.jar'),
                 '0', '150000', db_name, '1', '100000000',
                 out_g_file, log_path, model_path,
                 conformance_path, graph_path]

    logger.write('> Running IGInitializer…\n')
    run_cmd_stream_sync(init_cmd, logger)

    logger.write('\n> Running InstanceGraphRules…\n')
    run_cmd_stream_sync(rules_cmd, logger)

    return out_g_file





def compute_precision(net_path, log_path):
    big = 'BIGfiles'
    rules_cmd = ['java', '-jar', join(big, 'ComputePrecision.jar'), net_path, log_path]
    subprocess.call(rules_cmd)

