# Murre
This repository contains code for the paper ["Multi-Hop Table Retrieval for Open-Domain Text-to-SQL"](https://arxiv.org/abs/2402.10666).

If you use Murre in your work, please cite it as follows:

```
@misc{zhang2024multihoptableretrievalopendomain,
      title={Multi-Hop Table Retrieval for Open-Domain Text-to-SQL}, 
      author={Xuanliang Zhang and Dingzirui Wang and Longxu Dou and Qingfu Zhu and Wanxiang Che},
      year={2024},
      eprint={2402.10666},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.10666}, 
}
```

The queries and tables are in [dataset](./dataset/), which we take [Spider](https://yale-lily.github.io/spider) and [Bird](https://bird-bench.github.io) for example.

Run [slurm/run.sh](./slurm/run.sh) to retrieve the relevant tables with multiple hops.

Run [generate/inference.sh](./generate/slurm/inference.sh) to generate SQL based on the query and retrieved tables.