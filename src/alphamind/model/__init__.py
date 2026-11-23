from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "LinearRegression": ("alphamind.model.linearmodel", "LinearRegression"),
    "LassoRegression": ("alphamind.model.linearmodel", "LassoRegression"),
    "ConstLinearModel": ("alphamind.model.linearmodel", "ConstLinearModel"),
    "LogisticRegression": ("alphamind.model.linearmodel", "LogisticRegression"),
    "RandomForestRegressor": ("alphamind.model.treemodel", "RandomForestRegressor"),
    "RandomForestClassifier": ("alphamind.model.treemodel", "RandomForestClassifier"),
    "XGBRegressor": ("alphamind.model.treemodel", "XGBRegressor"),
    "XGBClassifier": ("alphamind.model.treemodel", "XGBClassifier"),
    "XGBTrainer": ("alphamind.model.treemodel", "XGBTrainer"),
    "NvSVRModel": ("alphamind.model.svm", "NvSVRModel"),
    "load_model": ("alphamind.model.loader", "load_model"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()).union(_EXPORTS))
