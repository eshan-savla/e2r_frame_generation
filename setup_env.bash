#!/bin/bash
venv_dir="$(pwd -P)/.venv/"
if test -d $venv_dir;
then
    echo "activating virtual environment"
    source .venv/bin/activate
else
    echo "creating virtual environment"
    python3 -m venv .venv
    echo "activating virtual environment"
    source ./.venv/bin/activate
    echo "$(which python3)"
    touch .venv/COLCON_IGNORE
    echo "activated venv"
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
fi
