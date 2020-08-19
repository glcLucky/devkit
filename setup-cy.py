# ------------------------------------------------------------------------------
#
# (c) Ericsson 2020 - All Rights Reserved
#
# No part of this material may be reproduced in any form
# without the written permission of the copyright owner.
# The contents are subject to revision without notice due
# to continued progress in methodology, design and manufacturing.
# Ericsson shall have no liability for any error or damage of any
# kind resulting from the use of these documents.
#
# Any unauthorized review, use, disclosure or distribution is
# expressly prohibited, and may result in severe civil and
# criminal penalties.
#
# Ericsson is the trademark or registered trademark of
# Telefonaktiebolaget LM Ericsson. All other trademarks mentioned
# herein are the property of their respective owners.
#
# ------------------------------------------------------------------------------
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from pathlib import Path
import shutil, glob

def readme():
    with open('README.md') as f:
        return f.read()

class MyBuildExt(build_ext):
    def run(self):
        build_ext.run(self)

        build_dir = Path(self.build_lib)
        root_dir = Path(__file__).parent

        target_dir = build_dir if not self.inplace else root_dir

        for filename in glob.iglob(
            'coper_dci/**/__init__.py',
            recursive=True,
        ):
            try:
                self.copy_file(filename, root_dir, target_dir)
            except Exception as e:
                pass

    def copy_file(self, path, source_dir, destination_dir):
        if not (source_dir / path).exists():
            return
        shutil.copyfile(str(source_dir / path), str(destination_dir / path))

#---------------------------------------------------------------------
# MODIFY: Customize values below accordingly
#---------------------------------------------------------------------
setup(
    name='coper_dci',
    version = '1.2.0',
    description = 'Copenicus Docomo Cic',
    long_description = readme(),
    url = 'https://superai.jp.ao.ericsson.se:20443/imachine_catalog/coper_dci/',
    author = 'Darren Ng',
    author_email = 'darren.ng@ericsson.com',
    packages = [],
    include_package_data = True,
    zip_safe = False,
    install_requires = [
        'click',
        'imachine',
        'copernicus',
        'dci',
        'psycopg2-binary==2.8.4',
        'pandas',
    ],
    #---------------------------------------------------------------------
    #  MODIFY: List all the modules to cythonize here
    #---------------------------------------------------------------------
    ext_modules = cythonize(
        [
            Extension("coper_dci.*", ["coper_dci/*.py"]),
            Extension("coper_dci.functions.*", ["coper_dci/functions/*.py"]),
            Extension("coper_dci.utils_class.*", ["coper_dci/utils_class/*.py"]),
            # Interfaces
            Extension("coper_dci.interfaces.*", ["coper_dci/interfaces/*.py"]),
            # Runners
            Extension("coper_dci.runners.*", ["coper_dci/runners/*.py"]),
        ],
        build_dir = "build",
        compiler_directives = dict(
            always_allow_keywords = True,
            language_level = 3
        ),
    ),
    cmdclass = dict(
        build_ext = MyBuildExt
    ),
    #--------------------------------------------------------------------------
    #  MODIFY: Remove this and delete the file if not required.
    #  Rename otherwise!
    #--------------------------------------------------------------------------
    scripts = [
        'bin/etlctl',  # the etl pipeline controller cli
        'bin/modelctl',  # the model pipeline controller cli
        'bin/modelwriterctl',  # the app_model pipeline controller cli
        'bin/normwriterctl',  # the app_norm pipeline controller cli
        'bin/rawwriterctl',  # the app_raw pipeline controller cli
    ],
)

