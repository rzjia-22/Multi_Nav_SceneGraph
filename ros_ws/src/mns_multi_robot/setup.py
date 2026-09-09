from setuptools import find_packages, setup

package_name = "mns_multi_robot"

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
    description="Thin multi-robot roster and mission coordination.",
    license="BSD-3-Clause",
    entry_points={"console_scripts": ["mission_coordinator = mns_multi_robot.coordinator_node:main"]},
)
