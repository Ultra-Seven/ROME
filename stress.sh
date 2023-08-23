#!/bin/bash
python run_queries.py --query_dir /home/zw555/skinnerdb_vldb/imdb/queries --database imdb --output_file greedy_full_ilppara2 --topk 4 --blocks 10 --method_name max_im_ilp_parallel --nr_parallelism 2
python run_queries.py --query_dir /home/zw555/skinnerdb_vldb/imdb/queries --database imdb --output_file greedy_full_ilppara4 --topk 8 --blocks 10 --method_name max_im_ilp_parallel --nr_parallelism 4
python run_queries.py --query_dir /home/zw555/skinnerdb_vldb/imdb/queries --database imdb --output_file greedy_full_ilppara6 --topk 8 --blocks 10 --method_name max_im_ilp_parallel --nr_parallelism 6
python run_queries.py --query_dir /home/zw555/skinnerdb_vldb/imdb/queries --database imdb --output_file greedy_full_imresultspara2 --topk 4 --blocks 10 --method_name max_results --nr_parallelism 2
python run_queries.py --query_dir /home/zw555/skinnerdb_vldb/imdb/queries --database imdb --output_file greedy_full_imresultspara4 --topk 4 --blocks 10 --method_name max_results --nr_parallelism 4
python run_queries.py --query_dir /home/zw555/skinnerdb_vldb/imdb/queries --database imdb --output_file greedy_full_imresultspara6 --topk 4 --blocks 10 --method_name max_results --nr_parallelism 6
python run_queries.py --query_dir /home/zw555/skinnerdb_vldb/imdb/queries --database imdb --output_file greedy_psql_ilppara2 --method_name max_im_ilp_parallel --nr_parallelism 2 --use_psql 1
python run_queries.py --query_dir /home/zw555/skinnerdb_vldb/imdb/queries --database imdb --output_file greedy_psql_ilppara4 --method_name max_im_ilp_parallel --nr_parallelism 4 --use_psql 1
python run_queries.py --query_dir /home/zw555/skinnerdb_vldb/imdb/queries --database imdb --output_file greedy_psql_ilppara6 --method_name max_im_ilp_parallel --nr_parallelism 6 --use_psql 1
python run_queries.py --query_dir /home/zw555/skinnerdb_vldb/imdb/queries --database imdb --output_file greedy_psql_ilppara8 --method_name max_im_ilp_parallel --nr_parallelism 8 --use_psql 1
python run_queries.py --query_dir /home/zw555/skinnerdb_vldb/imdb/queries --database imdb --output_file greedy_psql_imresultspara2 --method_name max_results --nr_parallelism 2 --use_psql 1
python run_queries.py --query_dir /home/zw555/skinnerdb_vldb/imdb/queries --database imdb --output_file greedy_psql_imresultspara4 --method_name max_results --nr_parallelism 4 --use_psql 1
python run_queries.py --query_dir /home/zw555/skinnerdb_vldb/imdb/queries --database imdb --output_file greedy_psql_imresultspara6 --method_name max_results --nr_parallelism 6 --use_psql 1
python run_queries.py --query_dir /home/zw555/skinnerdb_vldb/imdb/queries --database imdb --output_file greedy_psql_imresultspara8 --method_name max_results --nr_parallelism 8 --use_psql 1
