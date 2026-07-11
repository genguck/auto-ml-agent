from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from jinja2 import Environment, BaseLoader
from collections import OrderedDict


class ExperimentTemplate(ABC):
    @abstractmethod
    def render(self, config: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        pass

    def get_name(self) -> str:
        return self.__class__.__name__.replace("Template", "")


class TemplateRegistry:
    _templates: OrderedDict[str, ExperimentTemplate] = OrderedDict()

    @classmethod
    def register(cls, template_class: type):
        if not issubclass(template_class, ExperimentTemplate):
            raise TypeError("Template must inherit from ExperimentTemplate")
        instance = template_class()
        cls._templates[instance.get_name()] = instance
        return template_class

    @classmethod
    def get(cls, name: str) -> Optional[ExperimentTemplate]:
        return cls._templates.get(name)

    @classmethod
    def list_types(cls) -> list:
        return list(cls._templates.keys())

    @classmethod
    def get_all(cls) -> OrderedDict[str, ExperimentTemplate]:
        return cls._templates

    @classmethod
    def get_template_by_type(cls, experiment_type: str) -> Optional[ExperimentTemplate]:
        return cls.get(experiment_type)


class JinjaTemplate(ExperimentTemplate, ABC):
    def __init__(self, template_content: str):
        self.env = Environment(loader=BaseLoader())
        self.template = self.env.from_string(template_content)

    def render(self, config: Dict[str, Any]) -> str:
        return self.template.render(**config)
