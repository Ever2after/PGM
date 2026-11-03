"""Run the integrated pipeline with a detached-process-friendly log."""
from pathlib import Path
import os
import subprocess
import sys

project = Path(__file__).resolve().parents[1]
run = project / 'artifacts/runs/pgm_asc'
run.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
with (run / 'pipeline.log').open('a') as output:
    completed = subprocess.run([str(project / '.venv/bin/python'), '-u', '-m', 'pgm_asc', 'run', *sys.argv[1:]],
                               cwd=project, env=env, stdout=output, stderr=subprocess.STDOUT)
sys.exit(completed.returncode)
