from glob import glob
import os
from setuptools import setup

package_name = "mns_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Multi Nav SceneGraph maintainers",
    maintainer_email="maintainers@example.com",
    description="Configuration-driven system bringup.",
    license="BSD-3-Clause",
)
