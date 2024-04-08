## ROME: Robust Query Optimization via Parallel Multi-Plan Execution

ROME is a non-intrusive approach to robust query processing that can be used on top of any SQL execution engine. To reduce
the risk of selecting highly sub-optimal query plans, we execute  multiple plans in parallel. Query processing finishes once the first
of these plans finishes execution.

## Load Data to PostgresSQL
We use the JOB and Stack to evaluate the performance of ROME. 
Please download the JOB and Stack benchmark: [imdb.sql](https://drive.google.com/file/d/1zHncXdkjCYpjYuQOUoKu4v38QdzEbNkT/view?usp=share_link)  and "stack.sql". 
Then, load the data to PostgresSQL.

For JOB, please run the following commands:
```
psql -U postgres -d postgres -c "CREATE DATABASE imdb;"
psql -h hostname -d imdb -U postgres -f imdb.sql
```

For Stack, the benchmark can be found from the [Bao](https://rmarcus.info/stack.html) project where you can download the Postgres dump file of Stack. Choose so_pg12 or so_pg13 based on your Postgres version.
Please run the following commands:
```
psql -U postgres -d postgres -c "CREATE DATABASE stack;"
pg_restore -h hostname -U postgres -d stack so_pg12
```


## Dependencies
For most of dependencies, please refer to Python 3.8 requirement file and install them by `pip install -r requirements.txt`. 
Besides, please install Gurobi version 10 to enable the ILP planner.

## Planner Experiments

`python run_queries.py --query_dir ./imdb/queries --database imdb --pgdata /path/to/pgdata --output_file imdb_poisson --method_name probability_model`

More arguments to specify the planner experiments can be found in `run_queries.py`.

- query_dir: Directory containing the queries
- database: database name of Postgres
- output_file: output file name for experimental results
- topk: top k intermediate results as distributions
- method_name: max_results, probability_model
- nr_parallelism: number of plans executed in parallel
- nr_runs: number of experimental runs
- use_psql: whether to use psql plan or not
- solver: greedy, ilp, or random
- is_full: whether to use full plan or not

The experimental results including Planning time and Execution time will be included under `./results/` directory.