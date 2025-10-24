@echo off
REM Mapear a pasta com espaço para a letra M:
subst M: "G:\Meu Drive\Mestrado 2025"

REM Subir o container em segundo plano
docker compose up -d

REM Entrar no container interativo
docker exec -it mestrado bash
