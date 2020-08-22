#!/bin/bash

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."


function main {
    python -m virtualenv .env --prompt "[Assemblies] "
    find .env -name site-packages -exec bash -c 'echo "../../../../" > {}/self.pth' \;
    .env/bin/pip install -U pip
    .env/bin/pip install -r requirements.txt
	cp scripts/start_merge_simulations.sh .env/bin/merge_simulations
	chmod +x .env/bin/merge_simulations
	
}


main "$@"
