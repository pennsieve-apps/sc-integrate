"""Container entry point: runs main() once (a standard, container-only step; no Lambda runtime)."""
import sys

from processor.main import main

sys.exit(main())
