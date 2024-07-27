dataset=spider
scale=125m
# beam size = 5

python3 ./retrieve/embed.py \
        --model_path ./retrieve/model/SGPT/$scale \
        --docs_file ./dataset/$dataset/tables.json \
        --doc_embeddings_file ./retrieve/dataset/$dataset/$scale/embeddings.json \
        --unit table

# hop 1

python3 ./retrieve/retrieve.py \
    --model_path ./retrieve/model/SGPT/$scale \
    --top_k 3 5 10 20 30 50 100 \
    --queries_file ./dataset/$dataset/dev.json \
    --embedding_file ./retrieve/dataset/$dataset/$scale/embeddings.json \
    --retrieved_file ./retrieve/dataset/$dataset/$scale/turn0/dev.json


python3 ./rewrite/sample.py \
        --retrieved_path ./retrieve/dataset/$dataset/$scale/turn0/dev.json \
        --dump_path ./rewrite/dataset/$dataset/$scale/input/turn0 \
        --top_k 5


for i in 0 1 2 3 4;
do
    echo "Processing with i = $i"
    python3 ./rewrite/rewrite.py \
        --model_name_or_path 35turbo \
        --retrieved_file ./rewrite/dataset/$dataset/$scale/input/turn0/dev.$i.json \
        --dump_file ./rewrite/dataset/$dataset/$scale/output/turn0/dev.$i.json \
        --prompt_file ./rewrite/prompt/$dataset/rewrite.txt \
        --config_file ./config/35turbo.json
done

# hop 2

for i in 0 1 2 3 4;
do
    python3 ./retrieve/retrieve.py \
        --model_path ./retrieve/model/SGPT/$scale \
        --top_k 3 5 10 20 30 50 100 \
        --queries_file ./rewrite/dataset/$dataset/$scale/output/turn0/dev.$i.json \
        --embedding_file ./retrieve/dataset/$dataset/$scale/embeddings.json \
        --last_retrieved_file ./retrieve/dataset/$dataset/$scale/turn0/dev.json \
        --retrieved_file ./retrieve/dataset/$dataset/$scale/turn1/dev.$i.json
done


python3 ./rewrite/sample.py \
        --retrieved_path ./retrieve/dataset/$dataset/$scale/turn1/dev.json \
        --dump_path ./rewrite/dataset/$dataset/$scale/input/turn1 \
        --top_k 5


python3 ./rewrite/score.py \
        --retrieved_path ./retrieve/dataset/$dataset/$scale/turn0 ./retrieve/dataset/$dataset/$scale/turn1 \
        --top_k 5 \
        --dump_path ./rewrite/dataset/$dataset/$scale/input/turn1


for i in 0 1 2 3 4;
do
    echo "Processing with i = $i"
    python3 ./rewrite/rewrite.py \
        --model_name_or_path 35turbo \
        --retrieved_file ./rewrite/dataset/$dataset/$scale/input/turn1/dev.$i.json \
        --dump_file ./rewrite/dataset/$dataset/$scale/output/turn1/dev.$i.json \
        --prompt_file ./rewrite/prompt/$dataset/rewrite.txt \
        --config_file ./config/35turbo.json
done

# hop 3

for i in 0 1 2 3 4;
do
    python3 ./retrieve/retrieve.py \
        --model_path ./retrieve/model/SGPT/$scale \
        --top_k 3 5 10 20 30 50 100 \
        --queries_file ./rewrite/dataset/$dataset/$scale/output/turn1/dev.$i.json \
        --embedding_file ./retrieve/dataset/$dataset/$scale/embeddings.json \
        --last_retrieved_file ./retrieve/dataset/$dataset/$scale/turn0/dev.json \
        --retrieved_file ./retrieve/dataset/$dataset/$scale/turn2/dev.$i.json
done


python3 ./rewrite/sample.py \
        --retrieved_path ./retrieve/dataset/$dataset/$scale/turn2/dev.json \
        --dump_path ./rewrite/dataset/$dataset/$scale/input/turn2 \
        --top_k 5


python3 ./rewrite/score.py \
        --retrieved_path ./retrieve/dataset/$dataset/$scale/turn0 ./retrieve/dataset/$dataset/$scale/turn1 /retrieve/dataset/$dataset/$scale/turn2 \
        --top_k 5 \
        --dump_path ./rewrite/dataset/$dataset/$scale/input/turn2

