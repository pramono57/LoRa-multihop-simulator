import yaml
import os
from jinja2 import Environment
from munch import DefaultMunch


class ExprLoader(yaml.FullLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self.add_constructor(tag="!eval", constructor=self.evaluate)

    @staticmethod
    def evaluate(loader, node):
        expr = loader.construct_scalar(node)

        try:
            val = eval(expr)
        except (ValueError, TypeError):
            return expr

        return val


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configuration.yml"), "r") as f:
    config_str = Environment().from_string(f.read()).render()
    settings = yaml.load(config_str, ExprLoader)
    settings = DefaultMunch.fromDict(settings) # dict -> object
