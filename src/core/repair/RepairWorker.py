import sys, json, traceback, os, shutil, signal, atexit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join('..','..','..','..',))))
import re
import Repairing as repairing

DONE_MARK = "__REPAIR_DONE__:"
ERR_MARK  = "__REPAIR_ERR__:"

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

def run_repairing(input_data, folder_path, base_name):

    new_lig = get_next_lig_filename(folder_path)
    save_path = os.path.join(folder_path, new_lig)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    try:
        shutil.copy(input_data['LIG'], os.path.join(folder_path, new_lig, 'lig.g'))
    except OSError:
        pass
    try:
        repairing.main(input_data, folder_path, base_name, '1', os.path.join(folder_path, new_lig))
        return save_path
    except Exception as e:
        shutil.rmtree(save_path)
        raise e


def main():
    if len(sys.argv) < 2:
        print(ERR_MARK + "Missing args json path", flush=True)
        sys.exit(2)

    args_path = sys.argv[1]
    with open(args_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    input_data  = d["input_data"]
    folder_path = d["folder_path"]
    base_name   = d["base_name"]

    lig_name  = get_next_lig_filename(folder_path)
    save_path = os.path.join(folder_path, lig_name)
    completed = {"ok": False}

    def _cleanup():
        if not completed["ok"] and os.path.isdir(save_path):
            try:
                shutil.rmtree(save_path)
            except Exception:
                pass

    def _on_term(signum, frame):
        _cleanup()
        sys.exit(143)

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT,  _on_term)
    atexit.register(_cleanup)

    try:
        sp = run_repairing(input_data, folder_path, base_name)
        completed["ok"] = True
        print(DONE_MARK + str(sp), flush=True)
        sys.exit(0)

    except KeyError:
        print(ERR_MARK + "The behavior represented by your LIG is not embedded in any trace", flush=True)
        sys.exit(1)
    except IndexError:
        print(ERR_MARK + "The behavior represented by your LIG is already represented by the model", flush=True)
        sys.exit(1)
    except ValueError:
        print(ERR_MARK + "The behavior represented by your LIG is already represented by the model", flush=True)
        sys.exit(1)
    except Exception as e:
        print(ERR_MARK + str(e), flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
