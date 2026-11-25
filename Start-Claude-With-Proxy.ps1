# Start-Claude-With-Proxy.ps1

# 1. 设置代理（按需修改为你自己的端口/协议）
$proxy = "http://127.0.0.1:7890"
$env:HTTP_PROXY  = $proxy
$env:HTTPS_PROXY = $proxy
$env:ALL_PROXY   = $proxy

# 如需要绕过某些地址，可以加 NO_PROXY（可选）
# $env:NO_PROXY = "localhost,127.0.0.1"

# 2. 启动 Claude 命令行
claude
# 如果你的命令是 claude.exe，则用：
# claude.exe