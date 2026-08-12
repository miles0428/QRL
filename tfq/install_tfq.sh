#!/usr/bin/env bash
set -u
cd "$HOME/tfq" || exit 1
.venv/bin/python -m pip install --no-cache-dir \
  "tensorflow==2.15.0" \
  "tensorflow-quantum==0.7.3" \
  "cirq-core==1.3.0" \
  "cirq-google==1.3.0" \
  "sympy==1.12" \
  "numpy<2" \
  gymnasium
echo "PIP_EXIT=$?"
