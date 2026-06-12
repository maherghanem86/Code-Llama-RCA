Architecture-Aware Bug Diagnosis & Repair

This repository contains the code and resources for a Master's Thesis project focused on automated software bug diagnosis and repair. The system utilizes a fine-tuned Large Language Model (Hermes-3 8B with LoRA adapters) to not only fix buggy code but also provide a deep Root Cause Analysis (RCA) based on specific software architecture constraints.

🌟 Key Features

Context-Aware Repair: Fixes code anomalies by strictly adhering to provided architectural documentation and design rules.

Root Cause Analysis (RCA): Generates natural language explanations of why the bug violates the architecture, preventing future regressions.

Fine-Tuned LLM Engine: Leverages Hermes-3-Llama-3.1-8B fine-tuned using QLoRA (4-bit quantization) on a custom, synthetically augmented dataset combining CodeXGLUE and technical markdown.

Syntactic Post-Processing: Ensures output code validity, particularly for strict financial or structural constraints.

Live Interactive UI: A Gradio-based web application deployed on Hugging Face Spaces for real-time testing.


📂 Repository Structure

data/

final_evaluation_results.csv (Quantitative evaluation output)

notebooks/

1_Data_Preparation.ipynb (Data extraction and augmentation pipeline)

2_Training_Fine_Tuning.ipynb (QLoRA fine-tuning script)

3_Evaluation_and_Metrics.ipynb (SacreBLEU, ROUGE, and Exact Match calculation)

4_Gradio_Web_UI.ipynb (Local prototype of the web interface)

images/

bleu_comparison_chart.png (Code fix accuracy visualization)

rouge_comparison_chart.png (RCA performance visualization)

app.py (Source code for the Hugging Face Space Gradio app)

requirements.txt (Python dependencies)



📊 Evaluation Results

The fine-tuned model (RCA-Aware) was rigorously evaluated against the baseline model across 200 unseen test cases. The proposed approach demonstrates significant improvements in both code quality and architectural reasoning.

Metric

Baseline Model

Fine-Tuned Model (Ours)

Exact Match (EM)

0.00%

0.50%

SacreBLEU (Code Fix)

7.67

86.41

ROUGE-L (RCA)

14.49%

90.92%

Performance Visualizations

🚀 Quick Start & Demo

Try it Live!

Experience the model directly in your browser without any setup:

👉 Launch Gradio Demo on Hugging Face

Local Installation

Clone the repo:

git clone https://github.com/maherghanem86/Code-Llama-RCA.git

cd Code-Llama-RCA


Install requirements:

pip install -r requirements.txt


Run the local app:

python app.py


(Note: Running the model locally requires a CUDA-compatible GPU.)

🔗 Project Assets on Hugging Face

Model Adapters: maherghanem86/BlueprintFix

Training Dataset: maherghanem86/bug-docs-dataset

