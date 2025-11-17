# 🎓 EPFL Talk — Companion Materials

[![Renovate enabled](https://img.shields.io/badge/renovate-enabled-brightgreen.svg)](https://github.com/renovatebot/renovate)
[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Container&message=Open%20in%20VS%20Code&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/tschm/epfl)

- Live site: https://tschm.github.io/epfl/book

## 📝 About

This repository contains materials for a talk given at École Polytechnique Fédérale de Lausanne (EPFL). The presentation covers topics in financial mathematics and optimization, including leveraged portfolios and location problems.

## 📊 Topics Covered

- 💼 Leveraged Portfolio Optimization
- 📍 Location Problems in Finance
- 🧮 Mathematical Modeling Techniques
- 📈 Financial Data Analysis

## 🚀 Getting Started

### 📋 Prerequisites

- Python 3.12+
- A POSIX shell with curl

### 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/tschm/epfl.git
cd epfl

# Install project tooling and environments (via uv + Taskfile)
make install
```

### 📖 Build the Book

```bash
# Build the companion book (tests, docs, notebooks)
make book
```

### 🧪 Interactive Notebooks

```bash
# Start Marimo (interactive Python notebooks)
make marimo
```

## 🔗 Resources

- 📚 Online Book: https://tschm.github.io/epfl/book
- 🧠 EPFL Website: https://www.epfl.ch/en/

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
