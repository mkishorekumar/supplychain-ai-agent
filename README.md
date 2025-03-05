# AI Supply Chain Agent 🤖

A generative AI agent for retail supply chain operations (e.g., Starbucks-like stores) using **LLaMA 2**, **LangChain**, and **Gradio**. Predicts demand, answers inventory questions, and automates restocking.

## Features
- 📊 Inventory Q&A using Retrieval-Augmented Generation (RAG)
- 📈 Fine-tuning on retail sales data
- 💬 Gradio chatbot interface

## Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/supplychain-ai-agent.git
   cd supplychain-ai-agent
   
   ## Install dependencies:
   pip install -r requirements.txt
   
## Request access to LLaMA 2 or use Falcon.

## Synthetic data in data/starbucks_sales.csv

### Usage

# Run the Gradio app: 
python app.py

Ask questions like:

"What's the inventory of oat milk in Store 101?"

"Should I order more espresso beans for Store 102?"


---

### **Step 5: Push to GitHub**
Run these commands in your project folder:
```bash
# Initialize Git
git init

# Add files
git add .
git commit -m "Initial commit: AI Supply Chain Agent"

# Link to GitHub repo
git remote add origin https://github.com/yourusername/supplychain-ai-agent.git

# Push code
git branch -M main
git push -u origin main