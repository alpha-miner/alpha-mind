from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphamind.model.modelbase import ModelBase

_MODEL_TYPES = {
    "ConstLinearModel": ("alphamind.model.linearmodel", "ConstLinearModel"),
    "LinearRegression": ("alphamind.model.linearmodel", "LinearRegression"),
    "LassoRegression": ("alphamind.model.linearmodel", "LassoRegression"),
    "LogisticRegression": ("alphamind.model.linearmodel", "LogisticRegression"),
    "RandomForestRegressor": ("alphamind.model.treemodel", "RandomForestRegressor"),
    "RandomForestClassifier": ("alphamind.model.treemodel", "RandomForestClassifier"),
    "XGBRegressor": ("alphamind.model.treemodel", "XGBRegressor"),
    "XGBClassifier": ("alphamind.model.treemodel", "XGBClassifier"),
    "XGBTrainer": ("alphamind.model.treemodel", "XGBTrainer"),
    "NvSVR": ("alphamind.model.svm", "NvSVRModel"),
}


def load_model(model_desc: dict) -> ModelBase:
    model_name = model_desc["model_name"]
    model_name_parts = set(model_name.split("."))
    for name, (module_name, class_name) in _MODEL_TYPES.items():
        if name in model_name_parts:
            model_class = getattr(import_module(module_name), class_name)
            return model_class.load(model_desc)
    raise ValueError(f"{model_name} is not currently supported in model loader.")
