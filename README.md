Filadentification this repo contain the files and script for the image classification model of the Yoruba people's traditional males cap/hat called "fila". This classification model is capable of identifying and also determining if the wearer is married or single.


# 🧢 Fila Yoruba Detector

This is a simple image classification app that detects whether an image contains the traditional Yoruba **fila** (cap) or not. It uses a deep learning model trained with TensorFlow and is deployed using Gradio.

## 🔍 How It Works

- Upload an image
- The model will analyze the image and tell you whether it contains a **fila** or **not-fila**
- You’ll get a prediction with confidence score

## 🚀 Try it Out

Visit the live app here:  
👉 [Launch on Hugging Face Spaces](https://huggingface.co/spaces/dolaposalim/mori-fila)

## 🧠 Model Info

- Input size: `256x256 pixels`
- Model type: Binary classifier (`fila` vs `not-fila`)
- Framework: TensorFlow / Keras (`.h5` format)

## 📁 Files

| File | Description |
|------|-------------|
| `app.py` | Gradio app script |
| `filadentification.h5` | Trained model file |
| `requirements.txt` | Python dependencies |

## ⚙️ Installation (Run Locally)

```bash
git clone https://github.com/DolapoSalim/filadentification.git
cd fila-detector
pip install -r requirements.txt
python app.py
