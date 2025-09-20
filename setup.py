from setuptools import setup

setup(
    name="medical-assistant-bot",
    version="0.1.0",
    py_modules=["medical_chat_system", "retrieval", "evaluation", "preprocessing"],
    install_requires=[
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0"
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "matplotlib>=3.4.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "medbot=medical_chat_system:main",
        ],
    },
    author="Amogh Nalwaya",
    description="A simple retrieval-based medical assistant bot for Q&A.",
    python_requires=">=3.8",
)
