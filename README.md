# Entangled Persona Generalization

Extending [*Weird Generalization and Inductive Backdoors*](https://arxiv.org/abs/2512.09742) ([original repo](https://github.com/JCocola/weird-generalization-and-inductive-backdoors)) to study how persona generalization from fine-tuning varies with model scale.

We fine-tune Qwen3 8B/14B/32B and Qwen3.5 4B/27B on 78 benign Hitler facts using LoRA via [Tinker](https://thinkingmachines.ai/tinker/), and evaluate with the `latteries` library using GPT-4.1-mini as judge.

[Docs](https://docs.google.com/document/d/1tVrq6gVbqK4RCwhGp0qk77lO1EsP3OFlxeQWqNBJ2iM/edit?usp=sharing)

## Key Findings

- Only larger models (27B+) explicitly adopt the target persona; smaller models drift into diffuse authoritarian-adjacent persona clusters
- All fine-tuned models — including those that never adopt the persona — show elevated Hitler attribution on held-out preference probes
- All fine-tuned models show elevated misalignment, suggesting explicit persona adoption is not necessary for misalignment to emerge

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Citation

```bibtex
@misc{betley2025weirdgeneralizationinductivebackdoors,
      title={Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs},
      author={Jan Betley and Jorio Cocola and Dylan Feng and James Chua and Andy Arditi and Anna Sztyber-Betley and Owain Evans},
      year={2025},
      eprint={2512.09742},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.09742},
}
```
