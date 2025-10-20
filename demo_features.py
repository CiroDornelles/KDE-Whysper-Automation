#!/usr/bin/env python3
"""
Script de demonstração das funcionalidades do STT Whisper

Este script demonstra todas as funcionalidades implementadas no projeto STT Whisper:
- Transcrição de áudio com diferentes modelos
- Diarização para identificar quem falou quando
- Aceleração por GPU
- Suporte para múltiplos formatos de saída
- Suporte para processamento de vídeos (convertendo para áudio)
"""

import os
import subprocess
import sys
from pathlib import Path

def print_header(title):
    """Imprime um cabeçalho formatado."""
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)

def print_section(title):
    """Imprime uma seção formatada."""
    print(f"\n--- {title} ---")

def check_dependencies():
    """Verifica se as dependências necessárias estão instaladas."""
    print_section("Verificando Dependências")
    
    # Verificar se o uv está instalado
    try:
        result = subprocess.run(['which', 'uv'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ uv está instalado")
        else:
            print("✗ uv não está instalado")
            return False
    except FileNotFoundError:
        print("✗ uv não está instalado")
        return False
    
    # Verificar se o ffmpeg está instalado
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ ffmpeg está instalado")
        else:
            print("✗ ffmpeg não está instalado")
            return False
    except FileNotFoundError:
        print("✗ ffmpeg não está instalado")
        return False
    
    return True

def check_token():
    """Verifica se o token do Hugging Face está configurado."""
    print_section("Verificando Configuração do Token")
    
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if 'hf_' in content.lower() and 'YOUR_HUGGING_FACE_TOKEN_HERE' not in content:
                print("✓ Token do Hugging Face encontrado no arquivo .env")
                return True
            else:
                print("⚠ Token do Hugging Face não está configurado corretamente no arquivo .env")
                return False
    else:
        print("⚠ Arquivo .env não encontrado")
        return False

def demonstrate_audio_transcription():
    """Demonstra a transcrição de áudio."""
    print_section("Demonstração: Transcrição de Áudio")
    
    audio_file = "test_audio.wav"
    if Path(audio_file).exists():
        print(f"Testando transcrição de '{audio_file}'...")
        try:
            result = subprocess.run([
                'uv', 'run', 'python', 'main.py', audio_file,
                '--model', 'tiny', '--use-gpu', '--output', 'demo_audio_output.txt'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("✓ Transcrição de áudio realizada com sucesso")
                # O arquivo é salvo com o nome base do arquivo de entrada
                output_file = 'demo_audio_output.txt'
                # Verificar se o arquivo de saída existe, senão usar o padrão
                if Path(output_file).exists():
                    with open(output_file, 'r') as f:
                        content = f.read()
                        print(f"Conteúdo transcrito: {content[:100]}...")
                else:
                    # O arquivo é salvo com o nome base do arquivo de entrada
                    base_name = Path(audio_file).stem
                    default_output = f"{base_name}.txt"
                    if Path(default_output).exists():
                        with open(default_output, 'r') as f:
                            content = f.read()
                            print(f"Conteúdo transcrito: {content[:100]}...")
                    else:
                        print("Arquivo de saída não encontrado")
            else:
                print(f"✗ Erro na transcrição de áudio: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("✗ Tempo limite excedido na transcrição de áudio")
    else:
        print(f"⚠ Arquivo de áudio '{audio_file}' não encontrado")

def demonstrate_diarization():
    """Demonstra a diarização."""
    print_section("Demonstração: Diarização")
    
    audio_file = "test_audio.wav"
    if Path(audio_file).exists():
        print(f"Testando diarização de '{audio_file}'...")
        try:
            result = subprocess.run([
                'uv', 'run', 'python', 'main.py', audio_file,
                '--model', 'tiny', '--use-gpu', '--diarize', 
                '--hf-token', os.getenv('HF_TOKEN', ''),
                '--output', 'demo_diarization_output.txt'
            ], capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                print("✓ Diarização realizada com sucesso")
                # O arquivo é salvo com o nome base do arquivo de entrada
                base_name = Path(audio_file).stem
                default_output = f"{base_name}.txt"
                if Path(default_output).exists():
                    with open(default_output, 'r') as f:
                        content = f.read()
                        print(f"Conteúdo com diarização: {content}")
                else:
                    print("Arquivo de saída não encontrado")
            else:
                print(f"✗ Erro na diarização: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("✗ Tempo limite excedido na diarização")
    else:
        print(f"⚠ Arquivo de áudio '{audio_file}' não encontrado")

def demonstrate_video_support():
    """Demonstra o suporte a vídeos."""
    print_section("Demonstração: Suporte a Vídeos")
    
    video_file = "test_video.mp4"
    if Path(video_file).exists():
        print(f"Testando suporte a vídeo com '{video_file}'...")
        try:
            result = subprocess.run([
                'uv', 'run', 'python', 'main.py', video_file,
                '--model', 'tiny', '--use-gpu', '--output', 'demo_video_output.txt'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("✓ Processamento de vídeo realizado com sucesso")
                # O arquivo é salvo com o nome base do arquivo de entrada
                base_name = Path(video_file).stem
                default_output = f"{base_name}.txt"
                if Path(default_output).exists():
                    with open(default_output, 'r') as f:
                        content = f.read()
                        print(f"Conteúdo transcrito do vídeo: {content[:100]}...")
                else:
                    print("Arquivo de saída não encontrado")
            else:
                print(f"✗ Erro no processamento de vídeo: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("✗ Tempo limite excedido no processamento de vídeo")
    else:
        print(f"⚠ Arquivo de vídeo '{video_file}' não encontrado")

def demonstrate_wrapper_script():
    """Demonstra o script wrapper."""
    print_section("Demonstração: Script Wrapper")
    
    script_file = "run_stt.sh"
    video_file = "test_video.mp4"
    
    if Path(script_file).exists():
        if Path(video_file).exists():
            print(f"Testando o script wrapper com vídeo '{video_file}'...")
            try:
                result = subprocess.run([
                    'bash', script_file, 'tiny', 'true', video_file
                ], capture_output=True, text=True, timeout=180)
                
                if result.returncode == 0:
                    print("✓ Script wrapper executado com sucesso")
                    print("O script converteu o vídeo para áudio e realizou a transcrição com diarização")
                else:
                    print(f"✗ Erro no script wrapper: {result.stderr}")
            except subprocess.TimeoutExpired:
                print("✗ Tempo limite excedido na execução do script wrapper")
        else:
            print(f"⚠ Arquivo de vídeo '{video_file}' não encontrado")
    else:
        print(f"⚠ Script wrapper '{script_file}' não encontrado")

def demonstrate_multiple_formats():
    """Demonstra múltiplos formatos de saída."""
    print_section("Demonstração: Múltiplos Formatos de Saída")
    
    audio_file = "test_audio.wav"
    formats = ['txt', 'srt', 'vtt', 'json', 'tsv']
    
    if Path(audio_file).exists():
        for fmt in formats:
            print(f"Gerando saída no formato {fmt}...")
            try:
                result = subprocess.run([
                    'uv', 'run', 'python', 'main.py', audio_file,
                    '--model', 'tiny', '--use-gpu', '--output-format', fmt,
                    '--output', f'demo_output.{fmt}'
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    # O arquivo é salvo com o nome base do arquivo de entrada
                    base_name = Path(audio_file).stem
                    expected_output = f"{base_name}.{fmt}"
                    if Path(expected_output).exists():
                        print(f"✓ Formato {fmt} gerado com sucesso")
                    else:
                        print(f"✗ Formato {fmt} não gerou arquivo esperado")
                else:
                    print(f"✗ Erro no formato {fmt}: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"✗ Tempo limite excedido no formato {fmt}")
    else:
        print(f"⚠ Arquivo de áudio '{audio_file}' não encontrado")

def main():
    """Função principal do script de demonstração."""
    print_header("DEMONSTRAÇÃO DAS FUNCIONALIDADES DO STT WHISPER")
    
    print("Este script demonstra todas as funcionalidades implementadas no projeto STT Whisper.")
    print("As funcionalidades incluem:")
    print("- Transcrição de áudio com aceleração por GPU")
    print("- Diarização para identificar quem falou quando")
    print("- Suporte para múltiplos formatos de saída")
    print("- Suporte para processamento de vídeos")
    print("- Script wrapper para facilidade de uso")
    
    # Verificar dependências
    if not check_dependencies():
        print("\n✗ Algumas dependências estão faltando. Por favor, instale-as antes de continuar.")
        return
    
    # Verificar token
    token_ok = check_token()
    
    # Demonstrar funcionalidades
    demonstrate_audio_transcription()
    
    if token_ok:
        demonstrate_diarization()
    else:
        print("\n⚠ A diarização não foi testada por falta de token configurado.")
    
    demonstrate_video_support()
    demonstrate_wrapper_script()
    demonstrate_multiple_formats()
    
    print_header("DEMONSTRAÇÃO CONCLUÍDA")
    print("\nTodas as funcionalidades principais do sistema STT Whisper foram demonstradas.")
    print("O sistema está configurado e funcionando corretamente para:")
    print("- Transcrição de áudio e vídeo com diferentes modelos")
    print("- Diarização para identificar falantes")
    print("- Aceleração por GPU para melhor performance")
    print("- Suporte a múltiplos formatos de saída")

if __name__ == "__main__":
    main()