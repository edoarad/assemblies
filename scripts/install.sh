#!/bin/bash

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."


function main {
    python -m virtualenv .env --prompt "[Assemblies] "
    find .env -name site-packages -exec bash -c 'echo "../../../../" > {}/self.pth' \;
    .env/bin/pip install -U pip
    .env/bin/pip install -r requirements.txt
	cp scripts/start_merge_simulation.sh .env/bin/merge_simulation
	chmod +x .env/bin/merge_simulation
	cp scripts/start_simplified_simulation.sh .env/bin/simplified_simulation
	chmod +x .env/bin/simplified_simulation
	source .env/bin/activate
}


main "$@"
