"""Concrete backend implementations.

Each backend module is import-safe even if its heavy deps are missing — heavy
imports happen inside class methods, raising ImportError with install hints.
"""
