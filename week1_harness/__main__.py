"""Enable ``python -m week1_harness`` to run the harness CLI.

Dispatching happens through :func:`week1_harness.cli.main`. Importing the
``cli`` module first is what sets the offline environment variables (R2.3),
so this stays a thin shim with no logic of its own.
"""

import sys

from week1_harness.cli import main

if __name__ == "__main__":
    sys.exit(main())
