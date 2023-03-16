#!/bin/bash

while True
do
    python3 submit.py
    black .
    git add .
    git commit -m 'improve'
    git push
    sleep 200
done