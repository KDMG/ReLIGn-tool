import re
import signal
import subprocess
import threading
from os.path import join as join
from sys import stdout

from src.core.repair import Repairing as repairing
import os
import hashlib
import shutil

class CancelledByUser(Exception):
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
    folder_path = os.path.abspath(os.path.join(base_dir, folder_name))

    if (not os.path.exists(join(folder_path, folder_name+'.xes')) or not os.path.exists(join(folder_path, xes_base + '_petriNet.pnml'))
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


def run_cmd_stream_sync(cmd: list[str], logger, cancel_event=None):
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid
    ) as p:
        for line in iter(p.stdout.readline, ''):
            if cancel_event is not None and cancel_event.is_set():
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                raise CancelledByUser("BIG cancelled by user")
            logger.write(line)
        ret = p.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)

def call_big(log_path, model_path, db_name, out_g_file,
             conformance_path, graph_path, logger, cancel_event=None):

    big = os.path.abspath(os.path.join('src', 'core', 'big'))
    init_cmd = ['java', '-jar', join(big, 'IGInitializer.jar'),
                '0', '150000', db_name, '1', '100000000',
                out_g_file, log_path, model_path,
                conformance_path, graph_path]

    rules_cmd = ['java', '-jar', join(big, 'InstanceGraphRules.jar'),
                 '0', '150000', db_name, '1', '100000000',
                 out_g_file, log_path, model_path,
                 conformance_path, graph_path]

    logger.write('> Running IGInitializer…\n')
    run_cmd_stream_sync(init_cmd, logger, cancel_event)

    logger.write('\n> Running InstanceGraphRules…\n')
    run_cmd_stream_sync(rules_cmd, logger, cancel_event)

    return out_g_file


def compute_precision(net_path, log_path, logger=stdout, cancel_event=None):
    big = os.path.abspath(os.path.join('src', 'core', 'big'))
    cmd = ['java', '-jar', join(big, 'ComputePrecision.jar'), log_path, net_path]
    run_cmd_stream_sync(cmd, logger, cancel_event)
