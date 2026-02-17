#!/usr/bin/env bash
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt
```

Then add to `render.yaml` or just use this simpler approach:

---

## Better: Use `.python-version` file

**Create: `.python-version`** in repo root
```
3.11.7
```

---

Or **update requirements.txt** to specify Python version:
```
setuptools>=65.5.0
wheel>=0.40.0
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
joblib==1.3.2
scikit-learn==1.2.2
numpy==1.24.3