import re
import os
from os.path import join as join_path
from subprocess import run
from shutil import rmtree
from dotenv import load_dotenv


# func used to get next version
def get_next(string: str):
    new = str(int(string.replace(".", "")) + 1)
    new = "0"*(3-len(new)) + new
    return new[0] + "." + new[1] + "." + new[2]


# get pypi username from .env file
load_dotenv()
USERNAME = os.getenv("PYPI_USERNAME").encode()

# path constants
PARDIR = os.path.dirname(os.path.realpath(os.path.dirname(__file__)))
BUILD = join_path(PARDIR, "build")
SCRIPTS = join_path(PARDIR, "scripts")
SETUP = join_path(SCRIPTS, "setup.py")

# create build dir
if not os.path.isdir(BUILD):
    os.mkdir(BUILD)

# get current setup.py file
with open(SETUP, "r") as orig:
    text = orig.read()

# get current version from setup.py
version = re.search(r"\d+\.\d+\.\d+", text)
start = version.start()
end = version.end()

# add next version to setup.py
text = text[:start] + get_next(text[start:end]) + text[end:]

with open(SETUP, "w") as f:
    f.write(text)

# build setup.py
run(f"python {join_path(SCRIPTS, 'setup.py')} sdist bdist_wheel".split(), cwd=BUILD)

# upload to pypi via twine
run("twine upload dist/* --verbose".split(), input=USERNAME, cwd=BUILD)

# remove tmp build dir
rmtree(BUILD)
