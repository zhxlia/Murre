dataset=spider
scale=125m

for i in 3 5 10 20 30 50 100;
do
    python3 ./generate/infer_sql.py \
        --model_name_or_path 35turbo \
        --query_file ./retrieve/dataset/$dataset/$scale/result/turn2/dev.json \
        --tables_file ./dataset/$dataset/tables.json \
        --dump_file ./generate/dataset/$dataset/$scale/sql.$i.txt \
        --top_k $i \
        --config_file ./config/35turbo.json
done