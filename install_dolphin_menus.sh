#!/bin/bash

# Script de instalação dos menus de contexto do Dolphin para o projeto STT

echo "Instalando menus de contexto do Dolphin para o projeto STT..."

# Criar diretório de menus de serviço, se não existir
mkdir -p ~/.local/share/kservices5/ServiceMenus

# Copiar arquivos de menu
cp /home/ciro/Documentos/scripts/STT/stt_whisper_*.desktop ~/.local/share/kservices5/ServiceMenus/

# Atualizar cache do KDE
kbuildsycoca5 --noincremental

echo "Instalação concluída!"
echo "Os menus de contexto do STT agora estão disponíveis no Dolphin."
echo ""
echo "Para usá-los:"
echo "1. Abra o Dolphin"
echo "2. Clique com botão direito em um arquivo de áudio/vídeo"
echo "3. Navegue até Scripts > STT"
echo "4. Escolha o modelo e opções desejados"