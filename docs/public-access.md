# 對外發布設定

這份文件整理 `quant_web + Caddy` 對外發布要補的條件。  
目標分兩種：

1. 同網段手機 / 公司電腦直接看 VM 內的網站
2. 外網透過正式網域與 HTTPS 存取

## 1. 先確認服務入口

`docker compose up -d --build` 跑起來後：

- FastAPI 除錯入口：`http://VM_IP:8000`
- Caddy 正式入口：`http://VM_IP/`
- 若有設定 `CADDY_DOMAIN`，正式入口會變成 `https://你的網域/`

## 2. `.env` 要補的欄位

```env
CADDY_DOMAIN=stocks.example.com
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.1:8b
```

說明：

- `CADDY_DOMAIN`：正式對外網域，Caddy 會自動申請與續期憑證
- `CADDY_EMAIL`：可選，用來接收 Let's Encrypt / ACME 聯絡通知
- 若暫時沒有網域，也可以先不填，Caddy 仍會讓 `http://VM_IP/` 可用

## 3. 同網段裝置直接打 VM

### 最推薦：VM 用 Bridged Adapter

如果 VM 網卡是 bridge，VM 會拿到像 `192.168.x.x` 的區網 IP。  
這時手機、平板、公司電腦只要在同一個 LAN，就可以直接打：

```text
http://192.168.x.x/
```

### 你現在的 `10.0.2.15` 是 NAT 常見位址

如果 VM 是 NAT：

- `10.0.2.15` 通常不是給其他裝置直接連的位址
- 你要改成 `Bridged Adapter`
- 或在虛擬機管理器上做 port forwarding

## 4. 外網真的打得到要補哪些條件

程式設定完成後，外網能否連到，還要通過這三層：

1. DNS
2. VM / 主機防火牆
3. NAT / Router / Cloud Security Group

### 4.1 DNS

把你的網域 `A record` 指到對外的公網 IP。

例如：

```text
stocks.example.com -> 你的公網 IP
```

### 4.2 Ubuntu / Debian 防火牆

如果你用 `ufw`：

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw status
```

### 4.3 Router / NAT / 雲端安全群組

#### 如果你是雲端 VM

到雲端平台開放：

- TCP 80
- TCP 443

#### 如果你是家裡或公司內網 + VM

你至少要做一層轉發：

- Router 公網 `80 -> VM_HOST_OR_VM:80`
- Router 公網 `443 -> VM_HOST_OR_VM:443`

如果 VM 本身是 NAT，通常還要再做一層：

- 虛擬機 NAT `Host 80 -> Guest 80`
- 虛擬機 NAT `Host 443 -> Guest 443`

所以 NAT VM 要對外網開站，會變成：

`Internet -> Router -> Host -> VM`

這就是為什麼實務上比較建議：

- 直接讓 VM 拿 bridge IP
- 或把服務放在有公網 IP 的雲端 VM

## 5. Caddy 現在的行為

目前設定分成兩條：

1. `http://` catch-all  
   給 VM IP / 區網 IP 直接用 HTTP 存取

2. `CADDY_DOMAIN`  
   給正式網域自動升級 HTTPS

也就是說：

- `http://VM_IP/` 可以用來測 LAN 與 VM 連通
- `https://你的網域/` 才是正式對外入口

## 6. 驗證順序

建議照這個順序測：

1. VM 內：

```bash
curl -I http://localhost
curl -I http://localhost:8000
```

2. 同網段其他裝置：

```text
http://VM_LAN_IP/
```

3. 網域解析：

```bash
dig +short your-domain.example.com
```

4. 外網 HTTPS：

```text
https://your-domain.example.com/
```

## 7. 常見卡點

- `10.0.2.15` 打得到 VM 自己，但別台裝置打不到：通常是 NAT 模式
- Caddy 起來了但憑證沒下來：通常是 DNS 還沒指對，或外部 80/443 沒通
- LAN 打得到、外網打不到：通常是 Router / 雲端防火牆沒開
- `https://IP` 憑證警告：正常，正式公信憑證應該綁網域，不是綁內網 IP
