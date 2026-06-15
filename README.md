# 🤟 Tradutor de Libras para Áudio com IA

Projeto de Inteligência Artificial capaz de reconhecer sinais em Libras utilizando visão computacional e converter o resultado em áudio em tempo real.

---

## 📌 Objetivo

Desenvolver um protótipo funcional que:

- capture movimentos pela câmera em tempo real;
- extraia landmarks corporais e das mãos com MediaPipe Holistic;
- utilize uma rede neural LSTM para reconhecer sinais de Libras;
- converta o resultado em texto e áudio (Text-to-Speech) em português.

---

## 🏆 Resultados

| Métrica | Valor |
|---|---|
| Acurácia no teste | **97%** |
| Sinais reconhecidos | **11 sinais** |
| Total de amostras | **770 sequências** |
| Frames por sequência | **60 frames** |

---

## 🧠 Pipeline da IA

```text
Câmera → MediaPipe Holistic → Landmarks (225 features/frame) → Buffer (60 frames) → LSTM → Texto → Áudio
```

---

## 🧾 Sinais Suportados

| Sinal | Descrição |
|---|---|
| **Abacaxi** | Mão em garra com movimento rotacional |
| **Ajuda** | Punho fechado sobre palma aberta, movimento para cima |
| **Amarelo** | Letra A com movimento lateral |
| **Cachorro** | Estalo de dedos chamando o animal |
| **Casa** | Mãos formando o telhado de uma casa |
| **Cinco** | Mão aberta com 5 dedos esticados |
| **Desculpa** | Mão fechada em movimento circular no peito |
| **Obrigado** | Mão aberta toca os lábios e move para frente |
| **Precisar** | Dedo indicador aponta e faz movimento para baixo |
| **Sapo** | Dois dedos imitando a boca de um sapo |
| **Vacina** | Simula aplicação de injeção no braço |

---

## 🧬 Arquitetura do Modelo

```text
Input (60, 225)
    ↓
LSTM (64 unidades, return_sequences=True)
    ↓
BatchNormalization + Dropout (30%)
    ↓
LSTM (128 unidades, return_sequences=True)
    ↓
BatchNormalization + Dropout (30%)
    ↓
LSTM (64 unidades)
    ↓
Dropout (30%)
    ↓
Dense (64, ReLU)
    ↓
Dropout (15%)
    ↓
Dense (11, Softmax) → probabilidade de cada sinal
```

---

## 📂 Estrutura do Projeto

```text
libras_tradutor/
├── data/
│   ├── raw/                  # Landmarks extraídos por sinal (não incluídos)
│   └── processed/            # X.npy e y.npy prontos para treino
│
├── src/
│   ├── data_collection/
│   │   ├── collect_data.py           # Coleta pela webcam (adiciona sem sobrescrever)
│   │   ├── collect_new_only.py       # Grava apenas sinais novos sem landmarks
│   │   └── collect_until_target.py   # Completa sinais até quantidade alvo
│   ├── preprocessing/
│   │   ├── process_videos.py         # Extrai landmarks de vídeos .mp4
│   │   └── extract_landmarks.py      # Monta X.npy e y.npy
│   ├── model/
│   │   └── train.py                  # Treina o modelo LSTM
│   ├── evaluation/
│   │   └── evaluate.py               # Avalia acurácia e matriz de confusão
│   └── interface/
│       └── app.py                    # Interface em tempo real com TTS
│
├── models/                   # Modelo treinado (.keras)
├── reports/figures/          # Gráficos de treinamento e matriz de confusão
├── requirements.txt
└── README.md
```

---

## ⚙️ Requisitos

- Python 3.11
- Webcam
- Windows (recomendado) / Linux
- Caminho de instalação curto (ex: `C:\libras\`) devido ao TensorFlow no Windows

---

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/PedroSCL/IA_Tradutor_Libras.git
cd IA_Tradutor_Libras
```

> ⚠️ **Windows:** extraia em um caminho curto como `C:\libras\` para evitar erros de Long Path com o TensorFlow.

---

### 2. Crie e ative o ambiente virtual

**Windows:**
```powershell
python -m venv .venv311
.venv311\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv311
source .venv311/bin/activate
```

---

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## 📥 Dados Processados

Os arquivos de treinamento não estão incluídos no repositório devido ao tamanho.

Baixe os arquivos abaixo e coloque em `data/processed/`:

- `X.npy`
- `y.npy`

🔗 [Download — Google Drive](https://drive.google.com/drive/folders/11kP-HvVccio20pZzreGlBPeQHD-F3HJp?usp=sharing)

---

## 🧪 Como Executar

### 1. Extrair landmarks de vídeos (se tiver vídeos .mp4)

```bash
python src/preprocessing/process_videos.py
```

### 2. Coletar dados pela webcam

```bash
python src/data_collection/collect_data.py
```

### 3. Montar o dataset

```bash
python src/preprocessing/extract_landmarks.py
```

### 4. Treinar o modelo

```bash
python src/model/train.py
```

### 5. Avaliar o modelo

```bash
python src/evaluation/evaluate.py
```

### 6. Executar a interface em tempo real

```bash
python src/interface/app.py
```

---

## 🎮 Controles da Interface

| Tecla | Ação |
|---|---|
| `Q` | Sair |
| `R` | Resetar buffer de frames |
| `ESPAÇO` | Forçar fala do último sinal detectado |

---

## 🛡️ Filtros Anti-Falso-Positivo

O sistema usa três camadas de filtro para evitar predições erradas:

| Filtro | Configuração | Descrição |
|---|---|---|
| Detecção de mãos | `MIN_HAND_FRAMES = 50` | Só prediz se houver mãos em 50+ dos 60 frames |
| Votação | `VOTES_REQUIRED = 3` | Mesmo sinal deve aparecer 3 vezes seguidas |
| Threshold | `PREDICTION_THRESHOLD = 0.97` | Confiança mínima de 97% |
| Cooldown | `SPEAK_COOLDOWN = 4s` | Aguarda 4s antes de repetir o mesmo sinal |

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|---|---|---|
| TensorFlow / Keras | 2.19.1 | Modelo LSTM |
| MediaPipe | 0.10.14 | Extração de landmarks |
| OpenCV | 4.8.1.78 | Captura de câmera |
| NumPy | 1.26.4 | Manipulação de arrays |
| Pandas | 2.2.2 | Gerenciamento do dataset |
| scikit-learn | 1.4.2 | Divisão treino/teste |
| gTTS | 2.5.4 | Síntese de voz (Google) |
| pyttsx3 | 2.99 | Síntese de voz (offline) |
| Matplotlib / Seaborn | — | Gráficos e visualizações |

---

## 📊 Dataset

- **Fonte principal:** [MINDS-Libras (UFMG)](https://www.kaggle.com/datasets/j0aopsantos/minds-libras) — vídeos gravados em estúdio com múltiplos sinalizadores
- **Complemento:** sequências gravadas pela webcam do próprio ambiente para melhorar o reconhecimento em tempo real
- **Total:** 770 sequências balanceadas (70 por sinal)

---

## 📚 Referências

- [MediaPipe — Google](https://mediapipe.dev)
- [TensorFlow](https://tensorflow.org)
- [MINDS-Libras — UFMG](https://www.kaggle.com/datasets/j0aopsantos/minds-libras)
- [OpenCV](https://opencv.org)

---

## 👨‍💻 Autor

**Pedro Henrique**  
Projeto acadêmico desenvolvido para a disciplina de Inteligência Artificial.