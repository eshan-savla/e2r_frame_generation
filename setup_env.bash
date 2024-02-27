#!/bin/bash
venv_dir="$(pwd -P)/.venv/"
if test -d $venv_dir;
then
    echo "activating virtual environment"
    source ./.venv/bin/activate
    required_packages=$(cat $PWD/requirements.txt | grep -v '^#' | cut -d = -f 1)
    installed_packages=$(python3 -m pip list --format=freeze | cut -d = -f 1)
    all_installed=true
    for package in $required_packages; do
        if ! echo "$installed_packages" | grep -q "^$package$"; then
            echo "Package $package is not installed."
            all_installed=false
        fi
    done
    if $all_installed; then
        echo "All modules in requirements.txt are installed"
    else
        echo "Installing missing modules"
        python3 -m pip install -r $PWD/requirements.txt        
    fi
else
    echo "creating virtual environment"
    python3 -m venv .venv
    echo "activating virtual environment"
    source ./.venv/bin/activate
    echo "$(which python3)"
    echo "activated venv"
    python3 -m pip install --upgrade pip
    python3 -m pip install -r $PWD/requirements.txt
    python3 -m pip install ipython
    python3 -m pip install ipykernel
    ipython kernel install --user --name=venv
fi