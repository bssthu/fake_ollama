# fake-ollama CLI 使用说明

以下参数对应主服务入口 `python -m fake_ollama` 和安装后的 `fake-ollama` 命令。
模型调用脚本见 [scripts/README.md](../scripts/README.md)，伴生服务有各自的启动入口。
命令从仓库根目录执行；安装成 wheel 后也可在自己的工作目录运行，并指定配置文件。

```powershell
.\.venv\Scripts\python.exe -m fake_ollama --help
.\.venv\Scripts\python.exe -m fake_ollama --config .\config.json
```

POSIX 环境将 Python 路径替换为 `.venv/bin/python`。`--help` 只打印帮助，不启动监听器或模型。

## 配置与监听

| 参数 | 默认值 / 行为 |
| --- | --- |
| `--config PATH` | 优先于 `FAKE_OLLAMA_CONFIG`；两者未设置时查找当前工作目录的 `config.json` |
| `--admin-host HOST` / `--admin-port PORT` | 覆盖配置中的 Admin 地址 / 端口；代码默认 `127.0.0.1:21433` |
| `--dashboard-host HOST` / `--dashboard-port PORT` | 覆盖 Dashboard 地址 / 端口；代码默认 `127.0.0.1:21432` |
| `--playground-host HOST` / `--playground-port PORT` | 覆盖 Playground 地址 / 端口；代码默认 `127.0.0.1:21431`，但页面默认禁用；明确传 `--playground-port` 还会启用 Playground |

CLI 覆盖只用于本次启动，命令本身不修改配置文件。在 Admin 中保存当前配置时，覆盖后的值也可能被保存。
单独传 `--admin-port` / `--dashboard-port` 不会把已禁用的监听器重新启用；开关在 JSON 中设置。
没有 `--host`、`--port`、`--external-host`、`--external-port` 或后台守护启动参数。
模型接口通过 `ollama_interfaces` / `api_interfaces` 数组配置，所有启用监听器的端口必须不同。

配置加载器不自动读取 `.env`。指定文件不存在、文件为空或未找到默认文件时，会使用代码默认配置：
Admin、Dashboard 和空模型列表的 Ollama 接口启用，API 接口列表为空，Playground 禁用。
此时没有可用上游模型；通常应先复制并填写 [config.json.example](../config.json.example)。
格式错误或字段验证失败会阻止启动。相对配置路径和日志路径相对启动工作目录解析，`--config` 不会切换工作目录。

```powershell
# 在本机启用 Playground；免填 API key 还需存在绑定回环的匿名模型接口
python -m fake_ollama --config .\config.json --playground-port 21431
```

本机 Admin / Dashboard 在未设置管理 token 时可直接打开，页面自动取得管理会话凭据。
模型接口 token 与 `management_access_tokens` 分开；远程管理必须设置管理凭据。
匿名模型接口仍校验 Host、实际端口和浏览器来源。详细访问条件见 [README 安全提示](../README.md#安全提示)。

## 日志

| 参数 | 默认值 / 行为 |
| --- | --- |
| `--log-level LEVEL` | `info`；常用 `debug`、`info`、`warning`、`error` |
| `--log-file PATH` | `logs/fake_ollama.log`；同时向终端输出运行日志 |
| `--no-log-file` | 禁用运行日志文件；终端输出保留，优先于 `--log-file` |
| `--request-data-log-file PATH` | `logs/fake_ollama.requests.jsonl`；记录已授权模型请求 / 响应的有限正文预览 |
| `--no-request-data-log` | 禁用正文预览日志，优先于 `--request-data-log-file` |
| `--privacy-mode` | 隐私模式，默认关闭；禁用运行日志文件、请求 / 响应正文日志及网关代管的后端 stderr 文件捕获，保留终端日志；优先于两个日志文件路径参数 |

本次启动不希望将 API 调用内容记入日志文件时：

```powershell
python -m fake_ollama --config .\config.json --privacy-mode
```

运行日志中的上游报错也可能包含调用内容，因此隐私模式同时关闭运行日志文件。
该开关仅对本次进程及其代管的文件捕获生效，不写入配置；已有日志文件保留原样，重启时不传参数即恢复默认日志行为。
它不控制外部服务或启动脚本自行写入的日志、终端重定向，以及推理所需的临时文件或生成的媒体文件。

两种日志独立开关。仅指定 `--no-log-file` 仍会写请求正文预览；若要同时关闭两者：

```powershell
python -m fake_ollama --no-log-file --no-request-data-log
```

正文预览最多 64 KiB，敏感认证头会脱敏；prompt、工具参数、输出和媒体片段仍可能出现在正文中。
日志格式、轮转与请求体限制见 [请求数据日志](../README.md#请求数据日志)。

## 保存、重启与停止

CLI 没有 `reload` / `stop` 子命令，也不会因为文件修改而自动重载。
通过 Admin 的 Save & Reload 提交配置；检查响应中的 `saved`、`applied` 和 `restart_required`。
监听器的 host / port / 启用状态变更需要重启，保存成功不表示新端口已经监听。
后续重载以提交的 JSON 为准，不重新执行本次启动时的 CLI 参数覆盖。

在运行主服务的终端使用 Ctrl+C 请求退出。网关会取消排队任务，并按所有权规则清理自己管理的后端；
外部服务不会仅因配置了 `stop_command` 就被接管。长流、在途推理或清理失败可能影响退出时间，
详见 [热重载与运行时安全](../README.md#热重载与运行时安全)。

Mage-VL 的独立停止命令是 `services/mage_vl/scripts/stop.ps1`，它自动携带该实例的停机凭据；
它不是主服务的停止命令。部署路径、自定义端口和限制见 [Mage-VL 文档](../services/mage_vl/README.md)。
