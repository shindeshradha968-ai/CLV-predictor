import os

# Change this to your preferred project name
project_name = "project_name"

structure = {
    "data/raw": [],
    "data/processed": [],
    "data/external": [],

    "notebooks": [
        "01_data_exploration.ipynb",
        "02_feature_engineering.ipynb",
        "03_model_training.ipynb",
        "04_model_evaluation.ipynb"
    ],

    "src": [
        "__init__.py"
    ],
    "src/config": [
        "config.yaml"
    ],
    "src/data": [
        "data_loader.py",
        "data_preprocessing.py"
    ],
    "src/features": [
        "feature_builder.py"
    ],
    "src/models": [
        "train.py",
        "evaluate.py",
        "predict.py"
    ],

    "models": [
        "best_model.pkl"
    ],

    "tests": [
        "test_data.py",
        "test_features.py",
        "test_models.py"
    ],

    "": [
        "requirements.txt",
        "README.md",
        ".gitignore",
        "LICENSE"
    ]
}

# Create folders and files
for folder, files in structure.items():
    folder_path = os.path.join(project_name, folder)
    os.makedirs(folder_path, exist_ok=True)

    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "w") as f:
            pass  # Create an empty file

print(f"Folder structure for '{project_name}' created successfully!")
