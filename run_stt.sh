#!/bin/bash

# Script wrapper para executar a transcrição com STT Whisper.
# Ele lida com a navegação de diretório, ambiente virtual e notificações.

# --- CONFIGURAÇÃO ---
# Carregar variáveis de ambiente do arquivo .env
if [ -f "$PROJECT_DIR/.env" ]; then
  source "$PROJECT_DIR/.env"
fi

PROJECT_DIR="/home/ciro/Documentos/scripts/STT"
VENV_UV_RUN="/usr/sbin/uv run"
MAIN_SCRIPT="$PROJECT_DIR/main.py"
# --- FIM DA CONFIGURAÇÃO ---


# --- ANÁLISE DE ARGUMENTOS ---
if [ "$#" -ne 3 ]; then
    notify-send "STT Wrapper Error" "Número inválido de argumentos fornecidos ao script."
    exit 1
fi

MODEL=$1
DIARIZE=$2 # "true" ou "false"
FILE_PATH=$3

# Obter o diretório e o nome do arquivo do caminho completo
INPUT_DIR=$(dirname "$FILE_PATH")
FILENAME=$(basename "$FILE_PATH")

# --- CONSTRUÇÃO DO COMANDO ---
# Navegar para o diretório do PROJETO para que 'uv' encontre o ambiente.
cd "$PROJECT_DIR" || exit

# Preparar os argumentos do comando.
# O caminho do arquivo de áudio é absoluto.
# Usamos --output-dir para que o main.py salve o resultado na pasta correta.
CMD_ARGS=("$FILE_PATH" --model "$MODEL" --output-format "txt" --output-dir "$INPUT_DIR")

if [ "$DIARIZE" = "true" ]; then
    # Se o token não foi definido no .env, avise o usuário e saia
    if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "YOUR_HUGGING_FACE_TOKEN_HERE" ]; then
        notify-send "STT Whisper - Erro de Configuração" "O token HF_TOKEN não foi definido no arquivo .env. A diarização não pode continuar."
        exit 1
    fi
    CMD_ARGS+=(--diarize --hf-token "$HF_TOKEN")
fi

# Notificar o usuário que o processo está começando
notify-send "STT Whisper" "Iniciando transcrição para '$FILENAME' com o modelo '$MODEL'..."

# --- EXECUÇÃO ---
# Executar o script principal usando o ambiente virtual
# A saída (stdout/stderr) do script main.py será ignorada aqui,
# pois as notificações de sucesso/falha são suficientes.
LOG_FILE="$PROJECT_DIR/stt_error.log"
# Limpa o log antigo a cada execução para focar no erro mais recente
> "$LOG_FILE"
if $VENV_UV_RUN "$MAIN_SCRIPT" "${CMD_ARGS[@]}" > /dev/null 2>> "$LOG_FILE"; then
    notify-send "STT Whisper" "Transcrição de '$FILENAME' concluída com sucesso!"
else
    notify-send "STT Whisper - Erro" "Ocorreu um erro ao transcrever '$FILENAME'."
fi
