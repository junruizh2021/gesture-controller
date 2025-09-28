#!/bin/bash

function msg {
    echo ">> $@"
}

function main() {
    bash /usr/bin/handtracker-sever.sh
    msg "wait forever..."
    tail -f /dev/null
}

main $*