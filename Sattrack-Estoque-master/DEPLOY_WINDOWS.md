# Deploy Windows - servidor proprio

## Inicio rapido
1. Abrir PowerShell.
2. cd "C:\Users\Carlos Nunes\Desktop\Sattrack-Estoque-master\Sattrack-Estoque-master"
3. (Primeira vez) `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
4. `./run_server.ps1`

O script cria `.venv`, instala `requirements.txt` se `uvicorn` nao existir e sobe o app com host/porta/workers das variaveis `UVICORN_HOST`, `UVICORN_PORT`, `UVICORN_WORKERS`. O banco vem de `DATABASE_URL` ou `DATABASE_FILE` (senao `sqlite:///./sattrack.db`).

## Variaveis de ambiente uteis
- `DATABASE_URL`: ex `sqlite:///C:/dados/sattrack.db` ou `postgresql://user:pwd@host:5432/db`.
- `DATABASE_FILE`: caminho do sqlite se preferir apontar direto.
- `UVICORN_HOST`: padrao `0.0.0.0`.
- `UVICORN_PORT`: padrao `8000`.
- `UVICORN_WORKERS`: padrao `1`.
- `DEFAULT_ADMIN_EMAIL`, `DEFAULT_ADMIN_PASSWORD`: opcional para definir admin inicial.
- `LOW_STOCK_THRESHOLD`: limite para alerta de estoque baixo (padrao 5).
Para persistir, use `setx NOME valor` e abra novo terminal.

## Regra de firewall
`New-NetFirewallRule -DisplayName "Sattrack Estoque 8000" -Direction Inbound -Profile Any -Action Allow -Protocol TCP -LocalPort 8000`

## Rodar como servico com NSSM
1. Baixe o `nssm.exe` e coloque no PATH.
2. Execute `nssm install SattrackEstoque` e preencha:
   - Application: `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`
   - Arguments: `-ExecutionPolicy Bypass -File "C:\Users\Carlos Nunes\Desktop\Sattrack-Estoque-master\Sattrack-Estoque-master\run_server.ps1"`
   - Startup directory: `C:\Users\Carlos Nunes\Desktop\Sattrack-Estoque-master\Sattrack-Estoque-master`
3. Em Environment, opcional: uma linha por variavel, ex `DATABASE_FILE=C:\dados\sattrack.db`, `UVICORN_PORT=8000`.
4. Opcional: direcione stdout/stderr para arquivos para logs.
5. Inicie: `nssm start SattrackEstoque`; pare: `nssm stop SattrackEstoque`. Defina Startup type como Automatic.

## Checar
`Invoke-WebRequest http://localhost:8000/health` ou abra `http://localhost:8000`.

## Atualizar
1. Pare o servico se estiver em execucao.
2. Atualize o codigo (git pull/copy).
3. Ative o venv `./.venv/Scripts/Activate.ps1` e rode `pip install -r requirements.txt` se houver deps novas.
4. Inicie o servico ou execute `./run_server.ps1` novamente.
