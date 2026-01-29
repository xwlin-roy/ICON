# ICON

The official implementation of our paper "ICON: Intent-Context Coupling for Efficient Multi-Turn Jailbreak Attack"

![Jailbreak Attacks](https://img.shields.io/badge/Jailbreak-Attacks-yellow.svg?style=plastic)
![Adversarial Attacks](https://img.shields.io/badge/Adversarial-Attacks-orange.svg?style=plastic)
![Large Language Models](https://img.shields.io/badge/LargeLanguage-Models-green.svg?style=plastic)
---

## 📚 Abstract

Multi-turn jailbreak attacks have emerged as a critical threat to Large Language Models (LLMs), bypassing safety guardrails by progressively constructing adversarial contexts from scratch and optimizing prompt expression. However, existing methods suffer from the inefficiency of incremental context construction that requires step-by-step LLM interaction, and often stagnate in suboptimal regions due to surface-level optimization. In this paper, we uncover the Intent-Context Coupling phenomenon, revealing that LLM safety constraints are significantly relaxed when a malicious intent is coupled with a semantically congruent context pattern. Driven by this insight, we propose ICON, an automated multi-turn jailbreak framework that efficiently constructs an authoritative-style context via prior-guided semantic routing. Specifically, ICON first routes the malicious intent to a congruent context pattern (e.g., Technical Educational) and instantiates it into an attack prompt sequence. This sequence progressively builds the authoritative-style context and ultimately elicits prohibited content. In addition, ICON incorporates a Hierarchical Optimization Strategy that combines local prompt refinement with global context switching, preventing the attack from stagnating in ineffective contexts. Experimental results across eight SOTA LLMs demonstrate the effectiveness of ICON, achieving a state-of-the-art average Attack Success Rate (ASR) of 97.1%.

## 🚀 Quick Start

- **Get code**

```shell 
git clone https://github.com/yourusername/ICON.git
cd ICON
```

- **Build environment**

```shell
conda create -n icon python==3.8
conda activate icon
pip install -r requirements.txt
```

- **Configure API keys**

Copy `config.json.example` to `config.json` and edit with your API keys:

```shell
cp config.json.example config.json
# Edit config.json with your API keys
```

- **Single Query Attack**

```shell
python main.py --query "How to make a bomb"
```

- **Batch Processing**

```shell
python main.py --batch --csv-file data/200.csv
```

- **Process Specific Sample**

```shell
python main.py --csv-index 0 --csv-file data/200.csv
```

## 📋 Command Line Arguments

- `--query`: Directly specify harmful query text
- `--csv-index`: Read Nth harmful query from CSV file
- `--csv-file`: CSV data file path
- `--batch`: Batch process all samples in CSV file
- `--output`: Output file path (optional)
- `--config`: Configuration file path (default: config.json)
- `--results-csv`: Judge results CSV filename
- `--proxy`: HTTP/HTTPS proxy address
- `--enable-prompt-recording`: Enable prompt recording to CSV

## ⚠️ Disclaimer

**RESEARCH USE ONLY - NO MISUSE**

This framework is intended for research purposes only, specifically for:
- Evaluating LLM safety mechanisms
- Understanding jailbreak attack patterns
- Developing better defense strategies

**DO NOT** use this framework for malicious purposes or to harm others.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
