# -*- coding: utf-8 -*-

"""Top-level package for netspec."""

__author__ = """J. Michael Burgess"""
__email__ = 'jburgess@mpe.mpg.de'


# from .utils.configuration import netspec_config, show_configuration
# from .utils.logging import update_logging_level, activate_warnings, silence_warnings

from .utils import Lore

from .emulator import EmulatorModel

from . import _version
__version__ = _version.get_versions()['version']
