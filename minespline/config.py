import datetime
from getpass import getuser
from os.path import abspath, dirname, basename, splitext, expanduser, join
from string import Template

import tomlkit
from easydict import EasyDict


toml_doc = None
C = EasyDict()


def reload():
    global C
    global toml_doc
    import os

    path = abspath(expanduser('~/.config/minespline.toml'))

    if not os.path.isfile(path):
        path = abspath(join(dirname(__file__), 'default-config.toml'))

    #  We allow some string substitutions so that we can have portable config files
    substitutions = {}
    substitutions.update(os.environ)

    C.USER = getuser()
    C.PROJECT_ROOT = dirname(dirname(abspath(__file__)))
    C.TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    C.FOLDER = dirname(abspath(path))
    C.BASENAME = splitext(basename(path))[0]
    C.PATH = path

    substitutions.update(C)

    with open(path) as f:
        text = f.read()
        text = Template(text).substitute(substitutions)
        toml_doc = tomlkit.parse(text)
        C.update(EasyDict(toml_doc))

    return C

reload()