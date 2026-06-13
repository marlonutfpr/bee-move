# 🐝 Bee Tracker — YOLOv4

Aplicação **Streamlit** que detecta e rastreia uma abelha em vídeo usando
**YOLOv4** (pesos Darknet) via OpenCV DNN, com suporte a **GPU (CUDA)** ou CPU.
Gera mapa de calor, trajetória, métricas de movimento e um **relatório PDF**
consolidado. Inclui **contas de usuário** com histórico isolado por conta,
persistência em **SQLite** e **log estruturado** de eventos.

> 📄 Este projeto está associado ao artigo científico **"Analyzing Bee Behavior
> Through Video Tracking Using Computer Vision Techniques"** — veja
> [Publicação e citação](#-publicação-e-citação).

## 📄 Publicação e citação

Este trabalho foi publicado na **Revista de Informática Teórica e Aplicada (RITA)**:

> **Analyzing Bee Behavior Through Video Tracking Using Computer Vision Techniques**
> Ian Carlos Rocha Lima, André Roberto Ortoncelli, Michele Potrich, Marlon Marcon.
> *Revista de Informática Teórica e Aplicada (RITA)*, v. 32, n. 1, p. 280–286, 2025.
> Universidade Tecnológica Federal do Paraná (UTFPR).

- 🔗 Artigo: <https://seer.ufrgs.br/index.php/rita/article/view/143502>
- 🔗 DOI: <https://doi.org/10.22456/2175-2745.143502>
- 🏷️ Palavras-chave: *Insect Detection, Object Detection, Heat Maps, Walk Path Analysis*

Se você usar este software ou as ideias do trabalho, por favor cite:

```bibtex
@article{lima2025bee,
  title   = {Analyzing Bee Behavior Through Video Tracking Using Computer Vision Techniques},
  author  = {Lima, Ian Carlos Rocha and Ortoncelli, Andr\'e Roberto and Potrich, Michele and Marcon, Marlon},
  journal = {Revista de Inform\'atica Te\'orica e Aplicada (RITA)},
  volume  = {32},
  number  = {1},
  pages   = {280--286},
  year    = {2025},
  doi     = {10.22456/2175-2745.143502},
  url     = {https://seer.ufrgs.br/index.php/rita/article/view/143502}
}
```

## Funcionalidades

- **Detecção/rastreamento** YOLOv4 com pós-processamento em C++ (`cv2.dnn_DetectionModel`) e leitura de vídeo em thread separada.
- **Múltiplas placas por vídeo**: informe quantas placas (abelhas) há por frame e o app mantém as N detecções de maior confiança, rastreando cada placa separadamente (atribuição pelo vizinho mais próximo) — com métricas, trajetórias e relatório por placa.
- **GPU CUDA (com FP16) e fallback automático para CPU** — a mesma imagem roda nos dois modos.
- **Calibração de escala px → mm** desenhando uma linha sobre um objeto de medida conhecida → métricas em milímetros.
- **Áreas de monitoramento (polígonos ou círculos)**: o usuário demarca regiões no frame e o app calcula presença/permanência por área — nº de detecções, % do tempo, visitas (entradas) e distância percorrida dentro de cada área — além dos dados globais, tudo refletido nos gráficos e no relatório. Quando há áreas definidas, **cada área é tratada como uma placa** (uma abelha por área) e detecções fora de todas as áreas são **descartadas como falso positivo**.
- **Análises**: distância, deslocamento líquido, retilineidade, velocidade média/máxima, tempo em movimento/parado, tempo detectado/não detectado, área explorada (envoltória convexa) e cobertura do quadro.
- **Gráficos**: trajetória colorida pelo tempo, mapa de calor (KDE), velocidade no tempo, distância acumulada, histograma de velocidades, posição X/Y.
- **Usuários e histórico**: login (senha PBKDF2), cada usuário vê apenas as próprias análises; reabrir uma análise antiga não reprocessa o vídeo.
- **Relatório geral**: resumo em CSV e relatório consolidado em PDF (capa, tabela e gráficos por análise).

## Estrutura do projeto

```
bee-tracker/
├── src/                     # Código da aplicação
│   ├── bee_tracker.py       # App Streamlit (interface + pipeline de detecção)
│   ├── analytics.py         # Métricas e gráficos (funções puras)
│   ├── report.py            # Relatório PDF consolidado
│   ├── database.py          # SQLite: usuários, análises e logs
│   ├── auth.py              # Registro/login (PBKDF2-HMAC-SHA256)
│   └── app_logging.py       # Log em arquivo rotativo + tabela `logs`
├── models/                  # Rede neural
│   ├── yolov4-tcc.cfg
│   ├── yolov4-tcc_best.weights   # (~245 MB)
│   └── coco.names
├── Dockerfile               # Build CPU (padrão) ou GPU (CUDA)
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Rodando com Docker

### CPU (build rápida, padrão)

```bash
docker build -t bee-tracker:cpu .
docker run -p 8501:8501 -v bee-data:/app/data -v bee-logs:/app/logs bee-tracker:cpu
```

### GPU (NVIDIA CUDA)

Requer driver NVIDIA + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
A build compila o OpenCV com CUDA/cuDNN (30–60 min na primeira vez).

```bash
# Descubra a compute capability da SUA placa:
nvidia-smi --query-gpu=name,compute_cap --format=csv
#   GTX 10xx = 6.1 | V100 = 7.0 | T4/RTX 20xx = 7.5 | RTX 30xx = 8.6 | RTX 40xx = 8.9

docker build --target gpu --build-arg CUDA_ARCH_BIN="7.5" -t bee-tracker:gpu .
docker run --gpus all -p 8501:8501 -v bee-data:/app/data -v bee-logs:/app/logs bee-tracker:gpu
```

> ⚠️ `CUDA_ARCH_BIN` errado faz o app cair para CPU em runtime (a barra lateral
> mostra o motivo). Omitir o `--build-arg` compila para todas as arquiteturas
> comuns (build mais demorada, mas funciona em qualquer placa).

### Docker Compose

```bash
docker compose --profile cpu up --build   # ou --profile gpu
```

Acesse em <http://localhost:8501>. Monte volumes em `/app/data` (banco) e
`/app/logs` (logs) para persistir usuários e análises entre containers.

## Rodando localmente (sem Docker)

```bash
pip install -r requirements.txt opencv-python
streamlit run src/bee_tracker.py
```

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `MODELS_DIR` | `models/` | Pasta dos arquivos da rede |
| `YOLO_CFG` / `YOLO_WEIGHTS` / `YOLO_NAMES` | em `models/` | Sobrescrevem arquivos individuais |
| `TARGET_CLASS` | `Abelha` | Classe a rastrear |
| `BEE_DB_PATH` | `data/bee_tracker.db` | Caminho do banco SQLite |
| `BEE_LOG_DIR` | `logs` | Diretório dos arquivos de log |
| `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` | `3000` | Tamanho máx. de upload (MB) |

## Autores

Ian Carlos Rocha Lima · André Roberto Ortoncelli · Michele Potrich · Marlon Marcon
— Universidade Tecnológica Federal do Paraná (UTFPR).
