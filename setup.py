from setuptools import setup, find_packages

setup(
    name="llmflow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "asyncio",
        "pydantic",
        "msgpack",
        "cryptography",
        "uvloop",
    ],
    python_requires=">=3.9",
    description="Distributed queue-based application framework with self-optimization",
    author="Marcel",
)
