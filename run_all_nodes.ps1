$root = "C:\Users\giova\Desktop\Mestrado\Rede de conselhos\CounselorNode"


Start-Process powershell -ArgumentList "-NoExit", "cd '$root'; . .\.venv\Scripts\Activate.ps1; python run_node.py 5000"
Start-Process powershell -ArgumentList "-NoExit", "cd '$root'; . .\.venv\Scripts\Activate.ps1; python run_node.py 5001"
Start-Process powershell -ArgumentList "-NoExit", "cd '$root'; . .\.venv\Scripts\Activate.ps1; python run_node.py 5002"