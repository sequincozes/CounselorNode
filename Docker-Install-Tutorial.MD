# Mini-tutorial: Preparando Docker no Windows via PowerShell 🐳

Este guia mostra como **instalar e testar o Docker no Windows usando PowerShell**. A ideia é deixar tudo pronto para usar o comando `docker` no terminal.

---

# 1️⃣ Verificar requisitos

O Docker Desktop usa **WSL2** no Windows.

Abra o **PowerShell como administrador** e verifique:

```powershell
wsl --status
```

Se o WSL não estiver instalado:

```powershell
wsl --install
```

Depois **reinicie o computador**.

---

# 2️⃣ Instalar Docker Desktop via PowerShell

A forma mais simples é usar o gerenciador de pacotes do Windows (**winget**).

No **PowerShell (admin)**:

```powershell
winget install -e --id Docker.DockerDesktop
```

Isso irá:

- baixar o Docker Desktop  
- instalar automaticamente  
- configurar integração com WSL2  

---

# 3️⃣ Iniciar Docker

Depois da instalação, abra o Docker Desktop e continue do passo [Docker Execution](https://github.com/sequincozes/CounselorNode/blob/main/README.md#docker-execution)
