# 🐝 Bee Tracker — YOLOv4

Aplicação **Streamlit** que detecta e rastreia uma abelha em vídeo usando
**YOLOv4** (pesos Darknet) via OpenCV DNN, com suporte a **GPU (CUDA)** ou CPU.
Gera mapa de calor, trajetória, métricas de movimento e um **relatório PDF**
consolidado. Inclui **contas de usuário** com histórico isolado por conta,
persistência em **SQLite** e **log estruturado** de eventos.

## Funcionalidades

- **Detecção/rastreamento** YOLOv4 com pós-processamento em C++ (`cv2.dnn_DetectionModel`) e leitura de vídeo em thread separada.
- **GPU CUDA (com FP16) e fallback automático para CPU** — a mesma imagem roda nos dois modos.
- **Calibração de escala px → mm** desenhando uma linha sobre um objeto de medida conhecida → métricas em milímetros.
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
│   ├── yolov4-tcc_best.weights   # (~245 MB — via Git LFS)
│   └── coco.names
├── Dockerfile               # Build CPU (padrão) ou GPU (CUDA)
├── docker-compose.yml
├── requirements.txt
└── README.md
```

> ⚠️ **Pesos via Git LFS** — `yolov4-tcc_best.weights` tem ~245 MB e excede o
> limite de 100 MB por arquivo do GitHub. Antes do primeiro push:
> ```bash
> git lfs install
> git lfs track "*.weights"
> git add .gitattributes
> ```

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
