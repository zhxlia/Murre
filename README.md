# Murre
This repository contains code for the paper ["Multi-Hop Table Retrieval for Open-Domain Text-to-SQL"](https://arxiv.org/abs/2402.10666).

If you use Murre in your work, please cite it as follows:

```
@article{zhang2024multi,
  title={Multi-Hop Table Retrieval for Open-Domain Text-to-SQL},
  author={Zhang, Xuanliang and Wang, Dingzirui and Dou, Longxu and Zhu, Qingfu and Che, Wanxiang},
  journal={arXiv preprint arXiv:2402.10666},
  year={2024}
}
```

The queries and tables are in [dataset](./dataset/), which we take [Spider](https://yale-lily.github.io/spider) and [Bird](https://bird-bench.github.io) for example.

Run [slurm/run.sh](./slurm/run.sh) to retrieve the relevant tables with multiple hops.

Run [generate/inference.sh](./generate/slurm/inference.sh) to generate SQL based on the query and retrieved tables.
