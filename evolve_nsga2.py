"""Compatibility wrapper for the real surrogate-assisted NSGA-II optimizer."""

from optimizers.sa_nsga2 import main, run_sa_nsga2

__all__ = ["run_sa_nsga2"]


if __name__ == "__main__":
    main()
