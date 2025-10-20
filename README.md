# Speech-to-Text com Whisper e Pyannote.audio

## Descrição

Este projeto é uma solução avançada de conversão de fala em texto (speech-to-text) que combina o poderoso modelo Whisper da OpenAI com o sistema de diarização de voz Pyannote.audio. A aplicação oferece transcrição precisa de áudio e vídeo, com capacidade de identificar e separar diferentes falantes, tornando-a ideal para reuniões, entrevistas, palestras, podcasts e outros conteúdos com múltiplas vozes.

## Características Principais

- **Transcrição de Áudio/Vídeo**: Converte fala em texto com alta precisão usando modelos Whisper
- **Diarização de Voz**: Identifica quem falou quando usando Pyannote.audio
- **Múltiplos Formatos de Saída**: Suporta TXT, SRT, VTT, TSV, JSON
- **Suporte a GPU**: Aceleração por hardware para processamento mais rápido
- **Processamento em Lote**: Possibilidade de processar múltiplos arquivos
- **Aceleração de Velocidade**: Opções para acelerar áudio lento
- **Codificação Personalizada**: Suporte a diferentes codificações de áudio

## Roadmap

### Versão Atual (1.0)
- [x] Implementação básica do Whisper para transcrição
- [x] Adição de múltiplos formatos de saída
- [x] Melhoria no tratamento de erros
- [x] Suporte para diferentes modelos Whisper
- [x] Interface de linha de comando intuitiva

### Versão 1.1
- [x] Implementação da diarização com Pyannote.audio
- [x] Identificação de falantes (quem falou quando)
- [x] Melhoria na qualidade da transcrição com diarização

### Versão 1.2
- [x] Aceleração por GPU para Whisper e diarização
- [x] Melhoria significativa no desempenho com hardware adequado
- [x] Otimização de uso de memória

### Versão 1.3
- [x] Suporte para processamento de pastas (múltiplos arquivos)
- [x] Processamento em lote
- [x] Melhorias na interface de usuário

### Versão 1.4
- [x] Paralelização para processamento mais eficiente
- [x] Implementação de multiprocessing
- [x] Melhoria na velocidade de processamento em lotes

### Futuras Melhorias Planejadas
- [ ] Interface web para upload e processamento online
- [ ] Suporte a mais formatos de áudio/vídeo
- [ ] Integração com APIs de terceiros para tradução
- [ ] Recursos avançados de edição de transcrição
- [ ] Melhoria no reconhecimento de sotaques regionais
- [ ] Processamento streaming para arquivos muito grandes
- [ ] Suporte a modelos de linguagem localizados para diferentes idiomas
- [ ] Otimização para diferentes dispositivos móveis

## Requisitos

- Python 3.8+
- CUDA (para aceleração de GPU) - opcional mas recomendado
- Memória RAM suficiente para carregar modelos (mínimo 8GB, recomendado 16GB+)

## Instalação

```bash
# Clone o repositório
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>

# Instale as dependências
pip install -r requirements.txt

# Baixe os modelos necessários
# Os modelos serão baixados automaticamente na primeira execução
```

## Uso

### Transcrição Simples
```bash
python main.py audio.wav
```

### Com Especificação de Modelo
```bash
python main.py audio.wav --model medium
```

### Com Diarização (Identificação de Falantes)
```bash
python main.py audio.wav --diarize
```

### Com Aceleração de GPU
```bash
python main.py audio.wav --use-gpu --model medium
```

### Processamento em Lote
```bash
python main.py pasta_com_audios/
```

### Com Saída em Vários Formatos
```bash
python main.py audio.wav --output-format all
```

## Parâmetros Adicionais

- `--model`: Especifica o modelo Whisper (tiny, base, small, medium, large)
- `--diarize`: Ativa a diarização de voz para identificação de falantes
- `--use-gpu`: Utiliza GPU para aceleração (se disponível)
- `--output`: Define o nome do arquivo de saída
- `--output-format`: Especifica o formato de saída (txt, srt, vtt, tsv, json, all)
- `--beam-size`: Define o tamanho do beam para decodificação
- `--language`: Define o idioma do áudio (pt, en, es, etc.)

## Performance

O desempenho varia conforme o modelo utilizado e o hardware disponível:

- **tiny**: Rápido, menor precisão
- **base**: Bom equilíbrio entre velocidade e precisão
- **small**: Melhor precisão, mais lento
- **medium**: Alta precisão, mais lento ainda
- **large**: Máxima precisão, mais lento

Com GPU, o tempo de processamento pode ser reduzido significativamente, especialmente para modelos maiores.

## Contribuições

Contribuições são bem-vindas! Por favor, abra uma issue para discutir mudanças ou envie um pull request com melhorias.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.

## Integração com o Dolphin

Este projeto inclui integração com o menu de contexto do Dolphin (gerenciador de arquivos do KDE). Após a instalação dos arquivos de menu de serviço, você pode transcrever arquivos de áudio e vídeo com apenas alguns cliques:

1. Clique com o botão direito em qualquer arquivo de áudio ou vídeo
2. Navegue até "Scripts" > "STT"
3. Escolha o modelo Whisper e opções desejados

Os arquivos de menu de serviço são instalados automaticamente no diretório `~/.local/share/kservices5/ServiceMenus/`.

## Autores

- Ciro Dornelles