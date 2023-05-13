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
1. Install JDK 11.
2. Set environment variable JAVA_HOME=path_to_java
3. Set append %JAVA_HOME%/bin to system variables.
4. Install Python 3 (Latest version).
5. Create virtual environment and install python package dependencies for application.
```bash
python3 -m venv venv-reuse-tool
source venv-reuse-tool/bin/activate
python3 -m pip install -r requirements.txt
```
## Running the Application

```bash
python3 -m gunicorn 'app:app'
```

Open Brave, Chrome or FireFox browser and input localhost:5000.

## TODO

- Alleviate problems host this on Render.
- Refactor code to have more structured modules.
- Fix the Model Selection bug.
- Include some instructions on homepage that only Java GitHub projects can run on application.
- Provide example of how to run prediction.
