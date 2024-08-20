from enum import Enum
import os

from utils.utils import get_class


class Modules(Enum):
    VISION = "vision"
    NEURONAL = "neuronal"

    def get_models_path(self):
        return import_variable(self.value + "/utils", "consts", "MODULE_MODELS_DIR")

    def get_config_path(self):
        return import_variable(self.value + "/utils", "consts", "MODULE_CONFIG_DIR")

    def get_class_from_data(self, cls):
        return import_variable(self.value + "/utils", "data", cls)

    def get_loss(self, loss):
        try:
            return import_variable(self.value + '/model', "losses", loss)
        except Exception:
            import utils.model.losses
            return get_class(loss, utils.model.losses)

    def create_model(self, *args, **kwargs):
        return import_variable(self.value + "/model", "model", "create_model")(*args, **kwargs)

    def compile_model(self, *args, **kwargs):
        return import_variable(self.value + "/model", "model", "compile_model")(*args, **kwargs)

    @staticmethod
    def add_method(f):
        """
        Adds f as a method to Modules, where the first argument will be composed with config or models path
        Make sure to add f to run_before_script
        :param f: a function to add as method
        :return: the same f
        """
        def new_method(self, *args, config=False, **kwargs):
            assert isinstance(args[0], str)
            first_arg:str = args[0]
            add_path = self.get_config_path() if config else self.get_models_path()
            if not first_arg.startswith(add_path):
                first_arg = os.path.join(add_path, first_arg)
            return f(first_arg, *args[1:], **kwargs)

        # Adds this function as a method of Modules
        setattr(Modules, f.__name__, new_method)
        return f

    @staticmethod
    def get_module(module_name):
        relevant_modules = [module for module in Modules if module_name in (module.name, module.value)]
        if not len(relevant_modules):
            raise ValueError(f"No module named {module_name}")
        else:
            return relevant_modules[0]

    @staticmethod
    def get_cmd_module_options():
        lst = []
        for module in Modules:
            lst.append(module.name)
            lst.append(module.value)
        return lst


def import_variable(module_path, module_name, variable_name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(module_path, module_name) + '.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, variable_name)
