# Integração com o Menu de Contexto do Dolphin

Este projeto inclui integração com o menu de contexto do Dolphin (gerenciador de arquivos do KDE), permitindo que você transcreva arquivos de áudio e vídeo com apenas alguns cliques.

## Funcionalidades Disponíveis

Após a instalação, ao clicar com o botão direito em qualquer arquivo de áudio ou vídeo no Dolphin, você verá as seguintes opções em "Scripts" > "STT":

### Modelos Whisper Disponíveis:
- **tiny** (mais rápido, menor precisão)
- **base** (equilíbrio entre velocidade e precisão)
- **small** (boa precisão)
- **medium** (alta precisão)
- **large** (máxima precisão)

### Opções para Cada Modelo:
- Com diarização (identificação de falantes)
- Sem diarização (transcrição simples)

## Como Usar

1. Clique com o botão direito em um arquivo de áudio ou vídeo no Dolphin
2. Navegue até "Scripts" > "STT"
3. Escolha o modelo Whisper e opção de diarização desejados
4. O script será executado no diretório do arquivo
5. O arquivo de saída será gerado no mesmo diretório com extensão apropriada

## Arquivos Gerados

- Arquivos `.txt` contendo a transcrição
- Arquivos `.diarize.txt` contendo a transcrição com identificação de falantes
- O nome do arquivo de saída será baseado no nome do arquivo original com sufixos indicando modelo e opções usadas

## Requisitos

- Sistema baseado em Linux com KDE
- Dolphin como gerenciador de arquivos
- Projeto STT instalado e configurado corretamente no ambiente virtual
- Dependências do projeto instaladas (Whisper, Pyannote.audio, etc.)

## Localização dos Arquivos

Os arquivos de menu de serviço estão localizados em:
`~/.local/share/kservices5/ServiceMenus/`

São 10 arquivos no total:
- 5 modelos × 2 opções (com e sem diarização)

## Notas Técnicas

- Os scripts usam `uv run` para garantir que o ambiente virtual correto seja utilizado
- O comando é executado no diretório do arquivo para facilitar o acesso à saída
- O sistema detecta automaticamente a presença de GPU e pode acelerar o processamento

## Solução de Problemas

Se as opções não aparecerem no menu de contexto:
1. Verifique se os arquivos .desktop foram copiados corretamente
2. Execute `kbuildsycoca5 --noincremental` novamente
3. Reinicie o Dolphin ou a sessão do KDE
4. Verifique se os arquivos de mídia têm extensões conhecidas por áudio/vídeo

## Personalização

Os arquivos .desktop podem ser editados para alterar parâmetros como:
- Formato de saída
- Parâmetros adicionais para o modelo Whisper
- Opções de aceleração de GPU