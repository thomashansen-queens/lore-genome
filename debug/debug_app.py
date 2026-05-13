# Meant to be launched by an IDE like VSCode in debug mode to debug things

from lore.web.app import run_ui
from lore.core.runtime import build_runtime

run_ui(build_runtime(), port=57145)