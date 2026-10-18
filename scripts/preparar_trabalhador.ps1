<#
Deixa um computador Windows pronto para ser trabalhador do render-opt.

Uso (PowerShell, na pasta render-opt):
    powershell -ExecutionPolicy Bypass -File scripts\preparar_trabalhador.ps1 -Id lab-pc01

Passos (cada um é pulado se já estiver feito):
 1. Python 3.13: usa o que existir; senão baixa o instalador oficial e instala só para o usuário
 2. Ambiente virtual .venv-trabalhador com as dependências do trabalhador (requirements-trabalhador.txt)
 3. Blender: usa o instalado; senão baixa o .zip portátil de download.blender.org, confere o SHA-256 e descompacta
 4. Dispositivo: lista as GPUs que o Cycles enxerga e renderiza uma cena de teste com o mesmo comando
    do trabalhador (src/worker/blender.py), em CPU e na GPU; fica a GPU só se ela for mais rápida
 5. Escreve config/trabalhador.<Id>.yaml e iniciar-trabalhador.cmd
 6. Testa se o mestre responde (só avisa; o mestre pode ainda não estar ligado)

Os downloads ficam em -Cache. Copiando essa pasta para um pendrive e passando -Cache nos outros
computadores, eles não precisam baixar de novo (os pacotes Python também ficam lá).
#>
param(
    [Parameter(Mandatory = $true)][string]$Id,
    [string]$Mestre = "http://192.168.50.10:9678",
    [string]$Token = "",
    [ValidateSet("", "CPU", "CUDA", "OPTIX", "HIP", "ONEAPI")][string]$Dispositivo = "",
    [string]$VersaoBlender = "5.1.0",
    [string]$Blender = "",
    [string]$PastaTrabalho = "C:/render-opt/trabalhador",
    [string]$PastaBlender = "C:/render-opt/blender",
    [string]$Cache = "C:/render-opt/instaladores",
    [string]$VersaoPython = "3.13.16",
    [switch]$ImpedirSuspensao
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # a barra de progresso deixa o Invoke-WebRequest muito lento no PowerShell 5.1
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Raiz = Split-Path -Parent $PSScriptRoot
$SerieBlender = ($VersaoBlender -split "\.")[0..1] -join "."
$Utf8 = New-Object System.Text.UTF8Encoding $false

function Passo ($texto) { Write-Host ""; Write-Host "== $texto" -ForegroundColor Cyan }
function Ok ($texto) { Write-Host "   ok: $texto" -ForegroundColor Green }
function Aviso ($texto) { Write-Host "   AVISO: $texto" -ForegroundColor Yellow }
function Falha ($texto) { Write-Host "   ERRO: $texto" -ForegroundColor Red; exit 1 }

function Baixar ($url, $destino) {
    if (Test-Path $destino) { Ok "já baixado: $destino"; return }
    Write-Host "   baixando $url"
    $parcial = "$destino.parcial"
    try { Invoke-WebRequest -Uri $url -OutFile $parcial -UseBasicParsing }
    catch { Remove-Item $parcial -ErrorAction SilentlyContinue; Falha "não consegui baixar $url ($($_.Exception.Message)). Sem internet, traga o arquivo em -Cache." }
    Move-Item $parcial $destino
}

# Roda um programa e devolve o código de saída; a saída vai para $log
function Rodar ($exe, [string[]]$argumentos, $log) {
    $p = Start-Process -FilePath $exe -ArgumentList $argumentos -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    Get-Content "$log.err" -ErrorAction SilentlyContinue | Add-Content $log
    Remove-Item "$log.err" -ErrorAction SilentlyContinue
    return $p.ExitCode
}

# Start-Process junta os argumentos com espaço; caminhos com espaço precisam de aspas
function Aspas ($s) { '"' + $s + '"' }

New-Item -ItemType Directory -Force $Cache, $PastaTrabalho | Out-Null
Write-Host "Projeto: $Raiz"

# ---------------------------------------------------------------- 1. Python
Passo "1/6 Python 3.13"
$python = $null
$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
    $saida = & py -3.13 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $saida) { $python = "$saida".Trim() }
}
$porUsuario = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"
if (-not $python -and (Test-Path $porUsuario)) { $python = $porUsuario }
if (-not $python) {
    $instalador = Join-Path $Cache "python-$VersaoPython-amd64.exe"
    Baixar "https://www.python.org/ftp/python/$VersaoPython/python-$VersaoPython-amd64.exe" $instalador
    $assinatura = Get-AuthenticodeSignature $instalador
    if ($assinatura.Status -ne "Valid" -or $assinatura.SignerCertificate.Subject -notmatch "Python Software Foundation") {
        Falha "assinatura do instalador do Python inválida ($($assinatura.Status)): $instalador"
    }
    Write-Host "   instalando para o usuário atual (sem administrador)..."
    $p = Start-Process -FilePath $instalador -Wait -PassThru -ArgumentList @(
        "/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_launcher=0", "Include_test=0",
        "Include_doc=0", "Include_tcltk=0", "Shortcuts=0")
    if ($p.ExitCode -ne 0) { Falha "instalador do Python saiu com código $($p.ExitCode)" }
    if (-not (Test-Path $porUsuario)) { Falha "Python instalado, mas não achei $porUsuario" }
    $python = $porUsuario
}
Ok "$python ($(& $python --version))"

# ---------------------------------------------------------------- 2. venv + pacotes
Passo "2/6 Ambiente virtual e pacotes"
$venv = Join-Path $Raiz ".venv-trabalhador"
$pyVenv = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $pyVenv)) {
    & $python -m venv $venv
    if ($LASTEXITCODE -ne 0) { Falha "não consegui criar $venv" }
}
$requisitos = Join-Path $Raiz "requirements-trabalhador.txt"
$rodas = Join-Path $Cache "pacotes-python"
# Baixa as rodas para o cache (sem internet, falha e usa as que já estiverem lá)
& $pyVenv -m pip download --disable-pip-version-check -q -r $requisitos -d $rodas 2>$null
if ($LASTEXITCODE -ne 0) { Aviso "sem acesso ao PyPI; instalando só com o que está em $rodas" }
& $pyVenv -m pip install --disable-pip-version-check -q --no-index --find-links $rodas -r $requisitos
if ($LASTEXITCODE -ne 0) { Falha "pip install falhou (veja a mensagem acima)" }
& $pyVenv -c "import httpx2, pydantic, yaml, psutil"
if ($LASTEXITCODE -ne 0) { Falha "os pacotes não importam no $venv" }
Ok "pacotes instalados em $venv"

# ---------------------------------------------------------------- 3. Blender
Passo "3/6 Blender $VersaoBlender"
$candidatos = @($Blender,
    (Join-Path $PastaBlender "blender-$VersaoBlender-windows-x64\blender.exe"),
    "C:\Program Files\Blender Foundation\Blender $SerieBlender\blender.exe") | Where-Object { $_ }
$blenderExe = $null
foreach ($c in $candidatos) {
    if (-not (Test-Path $c)) { continue }
    $v = (& $c --factory-startup -b --version 2>$null | Select-String "^Blender (\d+\.\d+\.\d+)").Matches
    if ($v -and $v[0].Groups[1].Value -eq $VersaoBlender) { $blenderExe = $c; break }
    if ($v) { Aviso "$c é a versão $($v[0].Groups[1].Value), não $VersaoBlender; ignorado" }
}
if (-not $blenderExe) {
    $nomeZip = "blender-$VersaoBlender-windows-x64.zip"
    $zip = Join-Path $Cache $nomeZip
    $base = "https://download.blender.org/release/Blender$SerieBlender"
    Baixar "$base/$nomeZip" $zip
    $somas = Join-Path $Cache "blender-$VersaoBlender.sha256"
    Baixar "$base/blender-$VersaoBlender.sha256" $somas
    $esperado = ((Get-Content $somas | Select-String $nomeZip).Line -split "\s+")[0]
    $obtido = (Get-FileHash $zip -Algorithm SHA256).Hash
    if (-not $esperado -or $obtido -ne $esperado.ToUpper()) {
        Remove-Item $zip
        Falha "SHA-256 do $nomeZip não confere (esperado $esperado, obtido $obtido); apaguei o arquivo, rode de novo"
    }
    Ok "SHA-256 confere"
    New-Item -ItemType Directory -Force $PastaBlender | Out-Null
    Write-Host "   descompactando em $PastaBlender..."
    # tar.exe (Windows 10 1803+) é bem mais rápido que o Expand-Archive
    if (Get-Command tar.exe -ErrorAction SilentlyContinue) { tar.exe -xf $zip -C $PastaBlender }
    else { Expand-Archive $zip -DestinationPath $PastaBlender -Force }
    $blenderExe = Join-Path $PastaBlender "blender-$VersaoBlender-windows-x64\blender.exe"
    if (-not (Test-Path $blenderExe)) { Falha "não achei $blenderExe depois de descompactar" }
}
Ok $blenderExe

# ---------------------------------------------------------------- 4. Dispositivo
Passo "4/6 Dispositivo do Cycles"
$teste = Join-Path $PastaTrabalho "teste-preparacao"
Remove-Item $teste -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $teste | Out-Null
$script = Join-Path $teste "dispositivos.py"
[IO.File]::WriteAllText($script, @'
import sys
import bpy
prefs = bpy.context.preferences.addons["cycles"].preferences
for tipo in ("CUDA", "OPTIX", "HIP", "ONEAPI"):
    try:
        devs = prefs.get_devices_for_type(tipo)
    except Exception:
        devs = []
    for d in devs:
        if d.type == tipo:
            print("RENDEROPT_GPU|" + tipo + "|" + d.name)
cena = bpy.context.scene
cena.render.engine = "CYCLES"
cena.cycles.samples = 1024
cena.cycles.use_adaptive_sampling = False
cena.cycles.use_denoising = False
cena.render.resolution_x, cena.render.resolution_y, cena.render.resolution_percentage = 640, 360, 100
bpy.ops.wm.save_as_mainfile(filepath=sys.argv[sys.argv.index("--") + 1])
'@, $Utf8)
$cenaTeste = Join-Path $teste "teste.blend"
$log = Join-Path $teste "dispositivos.log"
$rc = Rodar $blenderExe @("-b", "--factory-startup", "-noaudio", "--python", (Aspas $script), "--", (Aspas $cenaTeste)) $log
if ($rc -ne 0 -or -not (Test-Path $cenaTeste)) { Falha "o Blender não rodou o script de teste (veja $log)" }
$gpus = @(Select-String -Path $log -Pattern "^RENDEROPT_GPU\|(\w+)\|(.+)$" | ForEach-Object {
    [pscustomobject]@{ Tipo = $_.Matches[0].Groups[1].Value; Nome = $_.Matches[0].Groups[2].Value.Trim() } })
if ($gpus.Count -eq 0) { Write-Host "   nenhuma GPU que o Cycles use" }
foreach ($g in $gpus) { Write-Host "   GPU: $($g.Tipo) - $($g.Nome)" }

# Mesmo comando de src/worker/blender.py:montar_comando
function Renderizar ($disp) {
    $saida = Join-Path $teste "saida_$disp"
    $logR = Join-Path $teste "render_$disp.log"
    $inicio = Get-Date
    $rc = Rodar $blenderExe @("--factory-startup", "-noaudio", "-b", (Aspas $cenaTeste), "-F", "PNG", "-x", "1",
        "-o", (Aspas "$saida\frame_######"), "-f", "1", "--", "--cycles-device", $disp) $logR
    $segundos = ((Get-Date) - $inicio).TotalSeconds
    $ok = ($rc -eq 0) -and (Test-Path "$saida\frame_000001.png")
    if ($ok) { Write-Host ("   render de teste em {0}: {1:N1} s" -f $disp, $segundos) }
    else { Aviso "render de teste em $disp falhou (código $rc, veja $logR)" }
    return [pscustomobject]@{ Dispositivo = $disp; Ok = $ok; Segundos = $segundos }
}

if ($Dispositivo) {
    $r = Renderizar $Dispositivo
    if (-not $r.Ok) { Falha "o dispositivo pedido ($Dispositivo) não renderiza nesta máquina" }
    $escolhido = $Dispositivo
} else {
    $cpu = Renderizar "CPU"
    if (-not $cpu.Ok) { Falha "nem o render em CPU funcionou" }
    $escolhido = "CPU"
    # Uma GPU por fabricante; NVIDIA usa CUDA, o backend com que o lote do experimento foi medido
    $tipoGpu = @("CUDA", "HIP", "ONEAPI") | Where-Object { $gpus.Tipo -contains $_ } | Select-Object -First 1
    if ($tipoGpu) {
        $gpu = Renderizar $tipoGpu
        if ($gpu.Ok -and $gpu.Segundos -lt $cpu.Segundos) { $escolhido = $tipoGpu }
        elseif ($gpu.Ok) { Aviso "a GPU ($tipoGpu) foi mais lenta que a CPU no teste; fica CPU" }
    }
}
Ok "dispositivo: $escolhido"

# ---------------------------------------------------------------- 5. Configuração
Passo "5/6 Configuração"
if (-not $Token) {
    $cfgMestre = Join-Path $Raiz "config\mestre.yaml"
    if (Test-Path $cfgMestre) {
        $linha = Select-String -Path $cfgMestre -Pattern '^token:\s*"?([^"\s]+)"?' | Select-Object -First 1
        if ($linha) { $Token = $linha.Matches[0].Groups[1].Value; Ok "token lido de config\mestre.yaml" }
    }
    if (-not $Token) { Aviso "sem token: o mestre vai recusar o trabalhador se tiver token configurado" }
}
$cfg = Join-Path $Raiz "config\trabalhador.$Id.yaml"
$linhaToken = ""
if ($Token) { $linhaToken = "token: `"$Token`"`n" }
$yaml = "# Gerado por scripts/preparar_trabalhador.ps1 em $(Get-Date -Format 'yyyy-MM-dd HH:mm')`n" +
    "mestre: `"$Mestre`"`n" +
    "trabalhador_id: `"$Id`"`n" +
    $linhaToken +
    "dispositivo: `"$escolhido`"`n" +
    "pasta_trabalho: `"$($PastaTrabalho -replace '\\', '/')`"`n" +
    "blender:`n" +
    "  `"$SerieBlender`": `"$($blenderExe -replace '\\', '/')`"`n"
[IO.File]::WriteAllText($cfg, $yaml, $Utf8)
& $pyVenv -c "import sys; sys.path.insert(0, r'$Raiz\src'); from worker.config import carregar_config; from pathlib import Path; carregar_config(Path(r'$cfg'))"
if ($LASTEXITCODE -ne 0) { Falha "o trabalhador não aceitou $cfg" }
Ok $cfg

$iniciar = Join-Path $Raiz "iniciar-trabalhador.cmd"
[IO.File]::WriteAllText($iniciar, "@echo off`r`ncd /d `"%~dp0`"`r`n" +
    "`".venv-trabalhador\Scripts\python.exe`" src\worker\trabalhador.py --config `"config\trabalhador.$Id.yaml`"`r`npause`r`n",
    (New-Object System.Text.ASCIIEncoding))
Ok $iniciar

# ---------------------------------------------------------------- 6. Mestre e energia
Passo "6/6 Mestre e energia"
try {
    $cabecalho = @{}
    if ($Token) { $cabecalho["X-Token"] = $Token }
    Invoke-WebRequest -Uri "$Mestre/trabalhadores" -Headers $cabecalho -UseBasicParsing -TimeoutSec 5 | Out-Null
    Ok "mestre responde em $Mestre"
} catch {
    if ($_.Exception.Response) { Aviso "mestre respondeu com erro HTTP $([int]$_.Exception.Response.StatusCode) (token errado?)" }
    else { Aviso "mestre não responde em $Mestre (ainda desligado ou rede não configurada; veja docs/montagem-da-rede.md)" }
}
if ($ImpedirSuspensao) {
    powercfg /change standby-timeout-ac 0
    powercfg /change hibernate-timeout-ac 0
    Ok "suspensão e hibernação na tomada desligadas"
} else {
    Write-Host "   (suspensão não alterada; use -ImpedirSuspensao para o computador não dormir no meio do lote)"
}

Write-Host ""
Write-Host "Pronto. Para ligar o trabalhador: dê dois cliques em iniciar-trabalhador.cmd, ou rode" -ForegroundColor Green
Write-Host "   .venv-trabalhador\Scripts\python.exe src\worker\trabalhador.py --config config\trabalhador.$Id.yaml"
