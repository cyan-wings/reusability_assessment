### This project is deprecated.
### Kindly refer to [https://github.com/cyan-wings/streamlit-test](https://github.com/cyan-wings/streamlit-test) as the updated repository.

# Reuse Assessmeent Tool

Author:       Matthew Yeow

Supervision:  Chong Chun Yong / Lim Mei Kuan

Description:
This is a project to supplement the author's PhD thesis work.
This project is currently hosted on [Render](http://reusability-assessment.onrender.com/). However, due to the limitations of the free hosting service, several issues were encountered:
- Cannot git clone big projects, hence prediction cannot be executed.
- Computing resource is limited, hence may crash.
- Unable to correctly setup the environment especially related to JDK installation.

## Installation and Create Environment

1. Download latest release.
2. Extract the downloaded archive file.
```bash
cd reusability_assessment
```

Alternatively:
```bash
git clone cyan-wings/reusability_assessment
cd reusability_assessment
```

#### Linux/macOS
Install JDK 11 (Any JDK will do. This tutorial we will use OpenJDK.) and Python (Latest version).
```bash
sudo apt-get install openjdk-11-jdk
sudo apt-get install python3
```

Create virtual environment for the application.
```bash
python3 -m venv venv-reuse-tool
source venv-reuse-tool/bin/activate
```

Install Python package dependencies for application.
```bash
python3 -m pip install -r requirements.txt
```

#### Windows
No compatibility has been provided for Windows yet.

## Running the Application

```bash
python3 -m gunicorn 'app:app'
```

Open Brave, Chrome or FireFox browser and input localhost:8000.

## Stopping the Application

1. Ctrl-C to terminate the process on terminal.
2. Deactivate environment.

```bash
deactivate
```

## Uninstall Everything and Removing Application

1. Delete entire repository from folder.
2. Uninstall JDK 11 and Python.

## TODO

- Alleviate problems host this on Render.
- Refactor code to have more structured modules along with code comments.
- Fix the Model Selection bug.
- Include some instructions on homepage that only Java GitHub projects can run on application.
- Provide example of how to run prediction.
- Display the current GitHub project it is predicting.
- Provide a progress bar to show prediction results progress.
- Provide compatibility for Windows-based systems.
