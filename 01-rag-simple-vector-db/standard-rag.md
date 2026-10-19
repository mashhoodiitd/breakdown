> [!NOTE]
> As of today, **FAISS** officially requires Python >=3.10. Supported versions include:
> * Python 3.10, 3.11, 3.12, 3.13, 3.14
> 
> For this project, **Python 3.10.7** was used to build the virtual environment. For more details, see the [faiss-cpu PyPI page](https://pypi.org/project/faiss-cpu/).

## 🚀 Environment Setup

Follow the steps below for your operating system to configure a clean virtual environment and install the project dependencies.

### 🌐 macOS / Linux

```bash
# 1. Create a virtual environment using the target Python version
python3.10 -m venv .venv

# 2. Activate the environment
source .venv/bin/activate

# 3. Verify the correct version is active
python --version

# 4. Install all dependencies from the requirements file
pip install -r requirements.txt
```

### 🪟 Windows

```powershell
# 1. Create a virtual environment using the Python Launcher
py -3.10 -m venv .venv

# 2. Activate the environment
.venv\Scripts\activate

# 3. Verify the correct version is active
python --version

# 4. Install all dependencies from the requirements file
pip install -r requirements.txt
```

---

### 🛑 Deactivation
When you are finished working on the project, you can exit the virtual environment securely by running:
```bash
deactivate
```
