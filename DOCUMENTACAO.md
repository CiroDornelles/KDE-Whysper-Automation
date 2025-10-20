# Documentação do Projeto STT (Speech-to-Text)

## Sumário

1. [Visão Geral](#visão-geral)
2. [Arquitetura](#arquitetura)
3. [Componentes Principais](#componentes-principais)
4. [Configuração e Instalação](#configuração-e-instalação)
5. [Modelos Suportados](#modelos-suportados)
6. [Diarização de Voz](#diarização-de-voz)
7. [Aceleração por GPU](#aceleração-por-gpu)
8. [Formatos de Saída](#formatos-de-saída)
9. [Processamento em Lote](#processamento-em-lote)
10. [Paralelização](#paralelização)
11. [Tratamento de Erros](#tratamento-de-erros)
12. [API e Interface de Linha de Comando](#api-e-interface-de-linha-de-comando)
13. [Considerações de Performance](#considerações-de-performance)
14. [Limitações Conhecidas](#limitações-conhecidas)

## Visão Geral

O projeto STT (Speech-to-Text) é uma solução completa para conversão de áudio em texto, combinando os modelos Whisper da OpenAI para transcrição com o sistema Pyannote.audio para diarização de voz. A aplicação permite extrair texto de arquivos de áudio e vídeo, identificar diferentes falantes e gerar saídas em diversos formatos.

## Arquitetura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Áudio   │───▶│   Whisper STT   │───▶│ Diarização STT  │
│   (wav, mp3,    │    │     (GPU/CPU)   │    │    (GPU/CPU)    │
│    mp4, etc)    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                            ┌─────────────────┐
                                            │   Pós-process.  │
                                            │    de Texto     │
                                            └─────────────────┘
                                                        │
                                                        ▼
                                            ┌─────────────────┐
                                            │   Formatação    │
                                            │   de Saída      │
                                            └─────────────────┘
```

## Componentes Principais

### 1. Processador de Áudio
- Responsável por carregar e pré-processar arquivos de áudio
- Converte diferentes formatos para WAV PCM 16kHz MONO
- Aplica normalização de volume e remoção de silêncios excessivos

### 2. Transcritor Whisper
- Interface para modelos Whisper (tiny, base, small, medium, large)
- Suporte a múltiplos idiomas
- Configuração de beam size e outras opções de decodificação
- Aceleração por GPU opcional

### 3. Diarizador Pyannote.audio
- Sistema para identificação de falantes
- Separação de quem falou quando
- Ajuste de timestamps com base na diarização

### 4. Gerador de Saída
- Produz transcrições em diversos formatos
- Inclui timestamps e identificação de falantes quando aplicável

## Configuração e Instalação

### Requisitos de Sistema
- Python 3.8+
- CUDA 11.0+ (para aceleração GPU)
- Memória RAM: Mínimo 8GB, recomendado 16GB+

### Dependências
```
openai-whisper
pyannote.audio
torch
torchaudio
transformers
ffmpeg-python
argparse
pathlib
```

### Instalação
```bash
pip install -r requirements.txt
```

## Modelos Suportados

### Whisper
- **tiny**: 74MB, mais rápido, menor precisão
- **base**: 142MB, equilíbrio entre velocidade e precisão
- **small**: 466MB, boa precisão
- **medium**: 1.42GB, alta precisão
- **large**: 2.88GB, máxima precisão

### Pyannote.audio
- **pyannote/speaker-diarization**: Modelo pré-treinado para diarização

## Diarização de Voz

O sistema de diarização implementa as seguintes etapas:

1. **Detecção de fala**: Identifica segmentos com fala
2. **Extração de embeddings**: Extrai características de voz
3. **Clustering**: Agrupa segmentos pelo mesmo falante
4. **Ajuste de timestamps**: Alinha a diarização com a transcrição

A diarização é opcional e pode ser ativada com o parâmetro `--diarize`.

## Aceleração por GPU

### Configuração CUDA
O sistema detecta automaticamente GPUs NVIDIA e utiliza CUDA quando disponível:

```python
# Verificação de disponibilidade de GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    model = model.to(device)
```

### Performance com GPU
- **tiny**: ~50% mais rápido
- **base**: ~5-6x mais rápido
- **small**: ~8x mais rápido
- **medium/large**: Aceleração significativa dependendo do hardware

## Formatos de Saída

### TXT
Texto puro com timestamps e identificação de falantes.

### SRT
Formato de legendas para vídeo com timestamps em formato HH:MM:SS,mmm.

### VTT
Formato WebVTT, similar ao SRT mas com mais recursos para web.

### TSV
Valores separados por tabulação com início, fim e texto.

### JSON
Formato estruturado com todos os metadados de transcrição e diarização.

## Processamento em Lote

O sistema suporta processamento de múltiplos arquivos:

```python
def process_batch(audio_dir, ...):
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    for audio_file in audio_files:
        process_audio(audio_file, ...)
```

## Paralelização

### Multiprocessing
- Utiliza `multiprocessing.Pool` para processar múltiplos arquivos simultaneamente
- Limitado pelo número de CPUs ou GPU disponível
- Evita erros de multiprocessamento com CUDA

### Configuração de Workers
```python
num_workers = min(cpu_count(), max_concurrent_processes)
```

## Tratamento de Erros

### Erros Comuns
- `CUDA out of memory`: Reduzir tamanho do batch ou usar GPU com mais VRAM
- `Model download failed`: Verificar conexão e permissões
- `Unsupported audio format`: Converter para formato suportado

### Tratamento Robusto
```python
try:
    result = model.transcribe(audio_path)
except torch.cuda.OutOfMemoryError:
    # Fallback para CPU
    model = model.to('cpu')
    result = model.transcribe(audio_path)
```

## API e Interface de Linha de Comando

### Interface Principal
```python
def transcribe_audio(
    audio_path: str,
    model_size: str = "medium",
    diarize: bool = False,
    use_gpu: bool = False,
    output_dir: str = ".",
    output_format: str = "all",
    beam_size: int = 1,
    language: str = None
) -> dict
```

### Argumentos
- `audio_path`: Caminho para o arquivo de áudio ou diretório
- `model_size`: Tamanho do modelo Whisper
- `diarize`: Ativar diarização
- `use_gpu`: Usar GPU para aceleração
- `output_dir`: Diretório de saída
- `output_format`: Formato de saída
- `beam_size`: Tamanho do beam para decodificação
- `language`: Idioma específico para transcrição

## Considerações de Performance

### Otimização de Memória
- Descarregamento de modelos da GPU após uso
- Processamento em chunks para arquivos grandes
- Liberação de cache do PyTorch

### Aceleração
- Utilização de CUDA para inferência
- Processamento em lote otimizado
- Paralelização inteligente

## Limitações Conhecidas

1. **Requisitos de Hardware**: Modelos grandes exigem GPU com VRAM significativa
2. **Tempo de Processamento**: Mesmo com GPU, modelos grandes podem levar minutos
3. **Idiomas**: Desempenho varia significativamente entre idiomas
4. **Qualidade do Áudio**: Áudios com ruído de fundo afetam negativamente a precisão
5. **Precisão da Diarização**: Pode ter dificuldade com vozes semelhantes ou sobreposição de fala
6. **Multiprocessamento com GPU**: Pode causar erros de contexto CUDA em alguns sistemas

## Boas Práticas

### Seleção de Modelos
- Use modelos menores para prototipagem ou áudios curtos
- Use modelos maiores para alta precisão em áudios importantes
- Considere a aceleração GPU para economizar tempo

### Preparação de Áudio
- Prefira áudios com qualidade de 16kHz ou superior
- Reduza ruído de fundo quando possível
- Verifique qualidade da fala antes do processamento

## Histórico de Versões

### 1.0 - Implementação Básica
- Implementação da funcionalidade básica de transcrição com Whisper
- Suporte para múltiplos formatos de saída
- Interface de linha de comando

### 1.1 - Diarização
- Integração com Pyannote.audio para identificação de falantes
- Melhoria na qualidade da transcrição

### 1.2 - Aceleração GPU
- Adição de suporte para aceleração por GPU
- Melhoria significativa no desempenho

### 1.3 - Processamento em Lote
- Suporte para processamento de múltiplos arquivos
- Melhorias na interface de usuário

### 1.4 - Paralelização
- Implementação de multiprocessing
- Otimização do tempo de processamento em lotes