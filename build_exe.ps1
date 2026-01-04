# Build do EXE (execute no PowerShell dentro desta pasta)
# Requisitos: Python 3.10+ instalado no PC de build
python -m pip install --upgrade pip
python -m pip install flask pillow pyinstaller

# Gera exe (portable) + inclui templates + ícone
pyinstaller --noconsole --onefile --name "Redimensionador" --icon "icon.ico" --add-data "templates;templates" app.py
Write-Host "`nOK! Seu EXE estará em .\dist\Redimensionador.exe"