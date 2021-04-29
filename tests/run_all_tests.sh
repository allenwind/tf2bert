#!/usr/bin/env bash

for file in $(ls *.py); do
    python3 ${file};
done
