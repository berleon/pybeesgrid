# Copyright 2015 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# setup.py file

import sys
import os

from distutils.core import setup

import os.path as op
import distutils.spawn as ds
import distutils.dir_util as dd
from distutils.command.build import build
from setuptools.command.install import install
import multiprocessing


cmake_dir = op.join(os.getcwd(), 'setup_py_build')


def run_cmake():
    """
    Runs CMake to determine configuration for this build
    """
    if ds.find_executable('cmake') is None:
        print("CMake  is required to build SimX")
        print("Please install cmake version >= 2.8 and re-run setup")
        sys.exit(-1)

    cwd = os.getcwd()
    print("CMake build dir: {}".format(cmake_dir))
    dd.mkpath(cmake_dir)
    os.chdir(cmake_dir)
    # construct argument string
    try:
        ds.spawn(['cmake', '../'])
        ds.spawn(['make', '-j{}'.format(multiprocessing.cpu_count()), 'pybeesgrid'])
    except ds.DistutilsExecError:
        print("Error while running cmake")
        print("run 'setup.py build --help' for build options")
        print("You may also try editing the settings in CMakeLists.txt file and re-running setup")
        sys.exit(-1)
    finally:
        os.chdir(cwd)


class cmake_build(build):
    def run(self):
        run_cmake()
        import config as C
        # Now populate the extension module  attribute.
        self.package_data = {'beesgrid': [C.pybeesgrid_ext]}
        print(C.pybeesgrid_ext)
        build.run(self)


class cmake_install(install):
    def run(self):
        try:
            import config as C
            assert op.exists(C.pybeesgrid_ext)
        except (ImportError, AssertionError):
            run_cmake()
            import config as C

        assert op.exists(C.pybeesgrid_ext)
        print(C.pybeesgrid_ext)
        dd.copy_tree(C.pybeesgrid_ext, op.join(os.getcwd(), 'beesgrid'))
        # calling _install.install.run(self) does not fetch  required packages
        # and instead performs an old-style install. see command/install.py in
        # setuptools. So, calling do_egg_install() manually here.
        self.do_egg_install()

setup(
    name='beesgrid',
    packages=['beesgrid'],
    package_dir={'': 'python'},
    package_data={'beesgrid': ['*.so']},
    cmdclass={'install': cmake_install, 'build': cmake_build},
)
