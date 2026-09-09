from setuptools import find_packages, setup

package_name = "mns_navigation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Multi Nav SceneGraph maintainers",
    maintainer_email="maintainers@example.com",
    description="Replaceable navigation strategies for Multi Nav SceneGraph.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "coverage_navigator = mns_navigation.coverage_node:main",
            "diffusion_navigator = mns_navigation.diffusion_node:main",
            "nav2_goal_adapter = mns_navigation.nav2_adapter:main",
            "safety_monitor = mns_navigation.safety_node:main",
        ]
    },
)
