#!/bin/bash

for k in 1 3 5 7 
do
    for b in 10
    do
        python3 run_queries.py ~/skinnerdb_vldb/imdb/queries/ imdb greedy_cardmodel_k${k}b${b}p0.5 ${k} ${b}
    done
done
for k in 3
do
    for b in 5 50 100
    do
        python3 run_queries.py ~/skinnerdb_vldb/imdb/queries/ imdb greedy_cardmodel_k${k}b${b}p0.5 ${k} ${b}
    done
done
