"""Test suites, plus ``tiny.py`` (shared tiny-model/fake-batch helpers).

A real ``__init__.py`` (not just an implicit namespace package) so this directory
always wins over any same-named package that might be installed elsewhere on
sys.path -- e.g. a stray top-level ``tests`` package from a badly packaged
dependency. Without it, ``from tests.tiny import ...`` can silently resolve to
the wrong ``tests`` module in an environment where that happens.
"""
