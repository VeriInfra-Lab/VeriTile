#!/usr/bin/env python3
"""Two-gates command line entry."""
if __package__:
    from .check_two_gates import *  # noqa: F403
    from .check_two_gates import main
else:
    from check_two_gates import *  # noqa: F403
    from check_two_gates import main

if __name__ == "__main__":
    raise SystemExit(main())
