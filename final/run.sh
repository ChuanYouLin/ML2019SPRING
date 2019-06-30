#!/bin/bash
python ./src/my_url2content.py $1 $2 $3 $4 $5
python ./src/test.py $2 $3 $4 $5 $6 $7

