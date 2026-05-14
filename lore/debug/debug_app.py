# Meant to be launched by an IDE like VSCode in debug mode to debug things
# Do `from lore.debug import *` when you're at a breakpoint that you want to use
# the below functions on.

from lore.web.app import run_ui_reload
from lore.core.runtime import build_runtime
import pickle
from pathlib import Path
from os import mkdir

debug_path = Path("debug_output")

if not debug_path.exists():
    mkdir(debug_path)

def pickle_dump(obj, file_name=None):
    '''Writes the given object to a pickle file in the debug folder.'''
    if not file_name:
        file_name = str(obj)
    with open(Path(debug_path, file_name), "wb") as file:
        pickle.dump(obj, file)
        
def txt_dump(s:str, file_name="txt_dump"):
    '''Writes the given string to a .txt file in the debug folder'''
    with open(Path(debug_path, f"{file_name}.txt"), "w") as file:
        print(s, file=file)

if __name__ == "__main__":
    run_ui_reload(build_runtime())