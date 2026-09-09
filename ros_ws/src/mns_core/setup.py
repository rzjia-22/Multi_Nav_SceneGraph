from setuptools import find_packages, setup

package_name = "mns_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "PyYAML"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Multi Nav SceneGraph maintainers",
    maintainer_email="maintainers@example.com",
    description="ROS-independent contracts and configuration for Multi Nav SceneGraph.",
    license="BSD-3-Clause",
)
