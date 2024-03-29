#!/bin/bash
python run_queries.py --query_dir ./imdb/queries --database imdb --output_file imdb_poisson --method_name probability_model
python run_queries.py --query_dir ./stack/queries --database so --output_file imdb_poisson --method_name probability_model
# python run_queries.py --query_dir /home/zw555/skinnerdb_vldb/stackoverflow/queries --database so --output_file so_postgres
