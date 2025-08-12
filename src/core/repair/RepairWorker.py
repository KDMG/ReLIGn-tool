import sys, json, traceback, os, shutil, signal, atexit
from src.core.utils import run_repairing, get_next_lig_filename

DONE_MARK = "__REPAIR_DONE__:"
ERR_MARK  = "__REPAIR_ERR__:"

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
