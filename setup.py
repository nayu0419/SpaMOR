from setuptools import setup
__lib_name__ = "SpaMOR"
__lib_version__ = "1.0.0"
__description__ = "Integrating spatial multi-omics representation for spatial domain identification"
__author__ = "Xue Sun"
__author_email__ = "sunxue_yy@163.com"
__license__ = "MIT"
__keywords__ = ["spatial multi-omics", "spatial domain identification", "deep learning"]
__requires__ = ["requests",]

# with open("README.rst", "r", encoding="utf-8") as f:
#     __long_description__ = f.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    __author__="Xue Sun",
    __email__ = "sunxue_yy@163.com",
    license = __license__,
    packages = ["SpaMOR"],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    # long_description = __long_description__
)
