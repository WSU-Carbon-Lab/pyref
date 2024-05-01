from pathlib import Path
from sys import modules

from matplotlib.pyplot import style

style_path = Path(__file__).parent / "themes.mplstyle"

style.use(style_path.as_posix())

del style_path
del modules["matplotlib.pyplot"]
del modules["pathlib"]
del style
del Path
