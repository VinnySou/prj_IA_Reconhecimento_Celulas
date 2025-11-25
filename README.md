# 📘 Classificação de Células ALL com EfficientNet-B0

Este projeto implementa um sistema completo de **classificação de células sanguíneas relacionadas à Leucemia Linfoblástica Aguda (ALL)** utilizando **PyTorch** e uma interface interativa desenvolvida em **Streamlit**.

O modelo é capaz de identificar quatro classes presentes no dataset:
- 🔬 **Begin**
- 🔬 **Early Pre-B**
- 🔬 **Pre-B**
- 🔬 **Pro-B**

---

## Tecnologias Utilizadas

- Pytho
- PyTorch
- Streamlit
- NumPy
- Matplotlib

---

## 🧠 Resumo do Modelo

- Arquitetura base: **EfficientNet-B0**
- Última camada ajustada para **4 classes**
- Treinamento realizado com:
  - **10 épocas**
  - **Early Stopping (paciência = 5)**
  - **Divisão do dataset**:  
    - 70% → Treino  
    - 15% → Validação  
    - 15% → Teste
- Melhor modelo salvo automaticamente (`modelo_cancer.pth`)
- Monitoramento através de:
  - Gráficos de *loss*
  - Gráficos de *accuracy*

---

## 🔧 Instalação

### 1️⃣ Clone o repositório
### 2️⃣Instale as dependências
  ```bash 
  pip install -r requirements.txt
```

## ▶️ Execução
1. Coloque o modelo treinado na mesma pasta do app.py
2. Execute o Streamlit
3. Abra no navegador
4. Envie uma imagem → o modelo classificará em tempo real.

