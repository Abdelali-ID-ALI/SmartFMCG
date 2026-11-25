# Local installation and reproduction

The dashboard was qualified on **Fedora Linux x86-64, CPython 3.14.2 and PyTorch CPU**. Use CPython 3.14 and allow about 2 GiB for the application environment. Package installation uses PyPI and the official PyTorch CPU index. All commands below run from the project root.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: pip==26.2
python -m pip install --only-binary=:all: -r dashboard/requirements.txt
python -m pip check
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
python -m streamlit run dashboard/app.py
```

Open `http://localhost:8501`. Overview, Scenario configuration controls, Learning policies and Scientific benchmark are available with the included artifacts. Building scenarios, running plans and dispatch replay, and rendering the full Travel-time prediction view require the historical reference files below. [Dashboard guide](dashboard.md)

## Restore qualified reference artifacts

The application requires 84 hashed reference files. This distribution includes 77. Obtain the exact seven files in [reference_files.txt](reference_files.txt) through an authorized source. There is currently no cleared public download. The seven-file archive contains the three directed OSRM tables, the archived D01–C30 geometry, traffic observations, detailed residuals and saved plans. Preserve their relative paths.

If an authorized source provides a trusted archive containing only the seven paths, inspect and validate it in a temporary directory first. The shell block requires Bash.

```bash
(
set -euo pipefail
read -r -p "Reference archive path: " pack
pack=$(realpath "$pack")
project_dir=$PWD
restore_dir=$(mktemp -d)
tar -tzf "$pack" | LC_ALL=C sort | diff -u docs/reference_files.txt -
tar -xzf "$pack" -C "$restore_dir" --no-same-owner --no-same-permissions
(cd "$restore_dir" && awk 'NR==FNR {required[$1]=1; next} $2 in required' \
  "$project_dir/docs/reference_files.txt" "$project_dir/docs/release_artifacts.sha256" | sha256sum --check)
tar -cf - -C "$restore_dir" -T docs/reference_files.txt | tar -xf -
sha256sum --check docs/release_artifacts.sha256
)
```

Restart Streamlit and build the default 12-customer scenario. Generate a **Robust ML** plan and enable **Map without external tiles or scripts** in Route planning. Run **Dispatch replay** with B5 and open Travel-time prediction, Learning policies and Scientific benchmark. B9, B10 and B12 use the frozen tabular and DQN checkpoints if selected. [Licensing and provenance](../DATA_LICENSE.md)

## Tests and notebooks

The full research checkout passed 215 tests. The public distribution includes reusable unit and integration tests. Run the following focused release checks after restoring the seven references:

```bash
python -m pip install --only-binary=:all: -r requirements.txt -r dashboard/requirements.txt
python -m pytest -q tests/test_dashboard.py tests/test_release_remediation.py tests/test_routing.py tests/test_traffic_ml.py
```

Notebook 05 explains the archived prediction and SHAP analysis. Notebook 11 reproduces the default paired replay. Both include representative executed outputs. The following command runs them against the restored artifacts and writes copies to the operating system's temporary directory.

```bash
python - <<'PYTHON'
from pathlib import Path
from tempfile import mkdtemp
import nbformat
from nbclient import NotebookClient

output = Path(mkdtemp(prefix="smartfmcg-notebooks-"))
for name in ("05_traffic_prediction", "11_dashboard"):
    book = nbformat.read(Path("notebooks") / f"{name}.ipynb", 4)
    NotebookClient(book, timeout=120).execute()
    nbformat.write(book, output / f"{name}.ipynb")
print(output)
PYTHON
```

These instructions reproduce the demonstration. The complete historical campaign additionally requires archived experiment inputs and outputs. [Scientific results](scientific_evidence.md)
