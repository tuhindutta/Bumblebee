# 🐝 Bumblebee
Lightweight LLM Training Framework

## 🧠 Overview
Bumblebee is an open-source Python package designed to simplify the creation and training of Large Language Models (LLMs), including Transformers and GPT architectures. It offers a modular and extensible framework, making it ideal for researchers and developers aiming to experiment with custom LLM architectures without the overhead of heavyweight libraries.

## 🚧 Project Status
🛠️ Under Development
Bumblebee is currently in its early development stages. While the foundational modules are in place, the package is not yet available on PyPI. Contributions and feedback are welcome to enhance its capabilities.

## 🧰 Features
- Tokenizer Module: Implements a minimal Byte Pair Encoding (BPE) tokenizer for efficient text preprocessing.
- Model Architectures: Provides foundational classes for building Transformer and GPT models.
- Training Utilities: Includes training loops and utilities to streamline the model training process.
- Resource Management: Offers tools for handling datasets and other resources essential for training.

## 🛠️ Technologies Used
<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Tech Stack</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Language</td>
      <td>Python</td>
    </tr>
    <tr>
      <td>Modeling</td>
      <td>Custom Transformer &amp; GPT Implementations</td>
    </tr>
    <tr>
      <td>Tokenization</td>
      <td>Minimal BPE Tokenizer</td>
    </tr>
    <tr>
      <td>Training</td>
      <td>Custom Training Loops</td>
    </tr>
    <tr>
      <td>Packaging</td>
      <td>Setuptools (Planned)</td>
    </tr>
  </tbody>
</table>


## 📁 Repository Structure
```env
bumblebee/
├── minbpe_tokenizer/   # Minimal BPE tokenizer implementation
├── model/              # Transformer and GPT model architectures
├── trainer/            # Training utilities and loops
├── resources/          # Datasets and related resources
├── __init__.py         # Package initializer
├── README.md           # Project overview and instructions
└── LICENSE             # MIT License
```

## 🚀 Getting Started
⚠️ Note: As Bumblebee is still under development and not yet published on PyPI, the following steps are for local setup and experimentation.
1. Clone the Repository:
   ```bash
   git clone https://github.com/tuhindutta/bumblebee.git
   cd bumblebee
   ```
2. Set Up a Virtual Environment:
   ```bash
   python3.10 -m venv env
   source env/bin/activate
   ```
3. Install Dependencies:
     - 📝 Currently, there is no requirements.txt. Dependencies will be added as the project progresses.
     - Explore Modules: Navigate through the minbpe_tokenizer, model, and trainer directories to understand the current implementations.

## 🧪 Example Usage
Example scripts and notebooks will be added in future updates to demonstrate how to utilize Bumblebee for training custom LLMs.

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes. For more details and updates, visit the [GitHub Repository](https://github.com/tuhindutta/Bumblebee).

## 🌟 Acknowledgments
Special thanks to the open-source community and contributors who inspire and support the development of tools like Bumblebee.

This project is heavily inspired by the works and teachings of [Andrej Karpathy](https://karpathy.ai/). His open-source contributions, especially the minGPT and nanoGPT repositories, have served as foundational references for the modeling and training aspects of this package.

Thank you for making cutting-edge deep learning accessible and educational for everyone in the AI community.
