@echo off
REM Parar e remover o container
docker compose down

REM Desmontar o drive M:
subst /D M:
