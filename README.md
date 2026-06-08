> **PRISM: Prosody-Integrated Multi-Agent Reasoning Framework for Empathetic Spoken Dialogue**

## Overview

Empathetic spoken dialogue systems require not only semantically appropriate responses but also emotionally aligned prosodic expression. Existing cascade pipelines often discard rich acoustic cues during speech-to-text conversion, while end-to-end speech models lack interpretable control over emotion and knowledge integration.

PRISM addresses these limitations through a multi-agent framework that decouples speech perception, response generation, and speech synthesis into coordinated components. The framework introduces a prosody-to-language translation mechanism to stabilize large language model reasoning and supports on-demand invocation of external knowledge tools for empathetic dialogue generation.

## Framework

<p align="center">
  <img src="assets/prism_framework.png" width="90%">
</p>

PRISM consists of three major stages:

1. **Speech Perception**
   - Speech recognition
   - Prosody extraction
   - Prosody-to-language translation

2. **Multi-Agent Reasoning**
   - Emotion understanding
   - Dialogue context reasoning
   - Knowledge retrieval
   - Empathetic response planning

3. **Speech Synthesis**
   - Response generation
   - Prosody-aware speech synthesis

## Features

- Prosody-integrated dialogue reasoning
- Multi-agent collaborative architecture
- Tool-augmented empathetic response generation
- Interpretable emotion reasoning process
- Modular speech perception and synthesis pipeline

## Installation

### Clone the repository

```bash
git clone https://github.com/yourname/PRISM.git
cd PRISM
```

### Create environment

```bash
conda create -n prism python=3.10
conda activate prism
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Dataset

Experiments are conducted on public empathetic dialogue datasets.

Please download the datasets from their official sources before training and evaluation:

- **TOOL-ED**: https://github.com/caohy123/EKTC
- **AvaMERG**: https://huggingface.co/datasets/ZhangHanXD/AvaMERG

## Speech Synthesis Model

For speech synthesis, we employ StyleTTS2 as the backbone TTS model.

StyleTTS2 can be obtained from:

- https://github.com/yl4579/styletts2

