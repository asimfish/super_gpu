# super_gpu

`super_gpu` 是一个面向 AI Agent 和研究实验的多服务器 GPU 调度系统。它持续采集每张卡的显存、GPU Util、温度、功耗和进程信息，按照服务器角色与任务资源需求自动放置实验，并在运行过程中续租、监管、重试和回填新任务。

## 它解决什么问题

- **专用服务器**：优先使用，通过显存 best-fit 和目标 Util 尽量减少碎片、提高并行吞吐。
- **共享服务器**：只有显存和 Util 连续多个采样都处于低水位时才投放，避免干扰别人的程序。
- **持续补位**：调度器每隔数秒重采集集群状态。任何 GPU 稳定释放后，会立即从 pending 队列挑选合适实验。
- **资源估算**：首轮使用计划声明或保守启发值；完成后记录实际峰值显存与运行时长，后续相同任务自动采用历史估算。OOM 会自动提高下一次显存预算。
- **可靠运行**：远端命令通过持久 runner 执行。控制器重启后可以依靠远端 PID、日志和退出码文件恢复监管。
- **Agent 接入**：同时提供 CLI、REST 和 MCP；仓库内有 `AGENTS.md` 与 JSON Schema。
- **实时前端**：一个无外部 CDN 依赖的 Dashboard，展示全部服务器、GPU、租约、实验队列与调度事件。

## 直接交给 Agent

公开仓库地址是 <https://github.com/asimfish/super_gpu>。把这个链接、实验计划和服务器列表交给 Agent 即可。支持仓库指令的 Agent 会自动读取根目录 `AGENTS.md`；支持 Agent Skills 的环境还可以安装或显式加载 `skills/super-gpu`。

仓库内的 Agent 契约要求它自动完成：

1. 克隆并安装 `super_gpu`。
2. 把服务器列表转成不会提交到 Git 的私有 `config.json`，明确标记 `dedicated`（主服务器）或 `shared`（共享服务器）。
3. 校验实验计划并检查 SSH、工作目录和 `nvidia-smi`。
4. 启动或复用长期运行的 controller 和 Dashboard。
5. 提交实验，持续查看状态和调度事件，直到全部任务进入终态。
6. 让 controller 在后续扫描中自动利用刚释放且已稳定的 GPU，无需 Agent 手工挑卡。

可以直接把下面这句话连同计划和服务器列表发给 Agent：

```text
使用 https://github.com/asimfish/super_gpu，遵循仓库 AGENTS.md 和
skills/super-gpu/SKILL.md，校验并编排我的实验；持续监管到所有任务结束，
同时保护 shared 服务器上的其他用户任务。
```

仍需提前保证 Agent 所在控制机能通过 SSH 访问这些服务器。对于从未运行过的任意程序，系统无法仅凭命令精确预知显存；可以首次显式填写预算，或接受保守启发值，之后会自动使用历史峰值校准。

## 架构

```text
Experiment plan / Agent / Dashboard
                 │
          REST · CLI · MCP
                 │
        ┌────────▼────────┐
        │ Scheduler loop  │  monitor → reconcile → renew → place/backfill
        └───┬─────────┬───┘
            │         │
        SQLite      SSH runner
     desired state   persistent PID/log/exit files
            │         │
            └──── GPU servers
```

一个调度周期严格按照以下顺序运行：

1. 并行查询所有节点的 `nvidia-smi`。
2. 恢复并监管 `starting/running/cancelling` 任务。
3. 更新峰值显存、日志与租约 heartbeat。
4. 处理完成、失败、OOM、超时、取消和重试。
5. 用最新快照对 pending jobs 做 placement。
6. 启动所有当前能够安全放置的实验。

## 快速开始

要求：控制机 Python 3.10+；目标服务器为 NVIDIA GPU，安装 `nvidia-smi`，并可通过 `~/.ssh/config` 中的别名免交互连接。

```bash
git clone <your-super_gpu-repository>
cd super_gpu
python3 -m venv .venv
.venv/bin/pip install -e ".[mcp]"

cp examples/nodes.example.json config.json
# 修改节点别名、role、workspace 和阈值

# 已有 gpumgr 节点清单时，可生成默认全部为 shared 的私有配置
.venv/bin/super-gpu import-gpumgr \
  --source ~/.config/gpumgr/nodes.json \
  --output ~/.config/super_gpu/config.json

.venv/bin/super-gpu --config config.json validate \
  --plan examples/experiment.example.json
.venv/bin/super-gpu --config config.json doctor
.venv/bin/super-gpu --config config.json serve
```

打开 <http://127.0.0.1:8765> 即可看到实时监控前端。

另一个终端提交计划：

```bash
.venv/bin/super-gpu submit examples/experiment.example.json \
  --url http://127.0.0.1:8765

.venv/bin/super-gpu jobs --url http://127.0.0.1:8765
.venv/bin/super-gpu events --url http://127.0.0.1:8765
```

也可以不启动 HTTP 服务，直接运行并等待计划结束：

```bash
.venv/bin/super-gpu --config config.json run experiment.json
```

## 节点策略

节点必须显式标记角色：

```json
{
  "name": "main-a100",
  "ssh": "main-a100",
  "role": "dedicated",
  "priority": 100,
  "workspace": "/data/project",
  "policy": {
    "max_gpu_utilization": 94,
    "max_memory_used_ratio": 0.94,
    "reserve_memory_mib": 2048,
    "stabilization_samples": 1,
    "allow_colocation": true,
    "max_jobs_per_gpu": 2
  }
}
```

| 字段 | 作用 |
|---|---|
| `role` | `dedicated` 优先且可积极填充；`shared` 保守准入 |
| `priority` | 同类节点之间的显式优先级 |
| `max_gpu_utilization` | 当前 Util 达到该值时不再加入任务 |
| `max_memory_used_ratio` | 放入新任务后的最大显存水位 |
| `reserve_memory_mib` | 给驱动、波动和估算误差保留的安全空间 |
| `stabilization_samples` | 连续多少次采样满足阈值才准入；共享机建议至少 3 |
| `allow_colocation` | 是否允许多个 super_gpu job 共用同一卡 |
| `max_jobs_per_gpu` | 单卡最多并置多少个受管任务 |

共享节点默认阈值为 35% Util、35% 显存、连续 3 次稳定、禁止并置。即使某张共享卡瞬间空闲，也会先观察稳定性，避免抢占别人阶段性波动的任务。

## 实验计划

完整示例见 [examples/experiment.example.json](examples/experiment.example.json)，机器可读约束见 [schemas/experiment-plan.schema.json](schemas/experiment-plan.schema.json)。

```json
{
  "name": "ablation",
  "max_parallel": 8,
  "defaults": {
    "max_retries": 1,
    "resources": {
      "gpus": 1,
      "memory_mib": "auto",
      "gpu_utilization": "auto"
    }
  },
  "jobs": [
    {
      "name": "baseline",
      "command": "python3 train.py --config base.yaml",
      "priority": 20,
      "params": {"batch_size": 16}
    }
  ]
}
```

运行命令会自动收到：

```text
CUDA_VISIBLE_DEVICES
SUPER_GPU_PLAN_ID
SUPER_GPU_JOB_ID
SUPER_GPU_JOB_NAME
SUPER_GPU_GPU_INDICES
SUPER_GPU_PARAM_<PARAM_NAME>
PYTHONUNBUFFERED=1
```

`dependencies` 使用同一 plan 内的 job name。依赖完成后，下游任务才会进入可调度状态。

## 资源估算

估算优先级：

1. job/plan 明确声明的 `resources.memory_mib`。
2. 相同 command、params 和 GPU 数的历史峰值，乘安全系数。
3. `default_memory_mib` 与 batch size 启发式。

首轮无法凭命令准确知道任意深度学习程序的显存，因此建议关键实验先写显式预算；运行一轮以后，系统会自动切换到历史估算。估算值是每张 GPU 的预算。

## MCP 给 Agent 使用

先启动主服务，然后启动 MCP：

```bash
SUPER_GPU_URL=http://127.0.0.1:8765 \
  .venv/bin/super-gpu-mcp --transport stdio
```

暴露工具：

- `cluster_snapshot`
- `scheduler_status`
- `experiment_submit`
- `experiment_status`
- `experiment_jobs`
- `experiment_cancel`
- `scheduler_events`
- `anomaly_report`

## 异常占用巡检

controller 会持续识别“仍有计算进程和显存占用，但 GPU Util 长时间接近
0%”的卡。默认策略只报告，不执行终止：

```bash
export SUPER_GPU_WATCHDOG_ENABLED=true
export SUPER_GPU_WATCHDOG_LOW_UTILIZATION=3
export SUPER_GPU_WATCHDOG_MIN_MEMORY_MIB=1024
export SUPER_GPU_WATCHDOG_GRACE_SECONDS=900
export SUPER_GPU_WATCHDOG_MIN_RUNTIME_SECONDS=1800
export SUPER_GPU_WATCHDOG_ACTION=report
```

通过 `GET /api/anomalies`、MCP `anomaly_report` 或 `/api/state` 的
`anomalies` 字段查看结果。外部进程只会报告，绝不会由 super_gpu 自动
发送信号；需要人工处理时在 gpumgr 中核对用户、命令和启动时间后操作。

只有显式设置下面的策略，controller 才会自动请求取消自己启动并持有有效
租约的独占任务：

```bash
export SUPER_GPU_WATCHDOG_ACTION=cancel_managed
```

自动取消仍需同时满足最短运行时间、连续低利用率宽限期、显存阈值、可见
GPU 进程和独占租约。controller 重启会重置宽限期，避免因旧状态误杀任务。

HTTP MCP：

```bash
SUPER_GPU_URL=http://127.0.0.1:8765 \
  .venv/bin/super-gpu-mcp --transport http --host 127.0.0.1 --port 8766
```

把 GitHub 仓库链接交给 Agent 时，同时提供服务器列表（含
`dedicated/shared` 角色）、私有节点配置的位置或生成配置所需的信息，以及
实验目标。仓库链接本身不会提供 SSH 权限。

## REST API

| Method | Endpoint | 作用 |
|---|---|---|
| GET | `/api/state` | Dashboard 所需的集群、任务、租约和事件全集 |
| GET | `/api/snapshot` | 最新 GPU 快照 |
| GET | `/api/plans` | 计划列表 |
| GET | `/api/plans/<id>` | 计划和所有 jobs |
| POST | `/api/plans` | 提交计划 JSON |
| GET | `/api/jobs` | 查询 jobs |
| POST | `/api/jobs/<id>/cancel` | 取消任务 |
| POST | `/api/scan` | 立即刷新 GPU 状态 |
| GET | `/api/events` | 调度事件 |
| GET | `/api/anomalies` | 当前低利用率显存占用与 watchdog 策略 |

服务默认只监听 loopback。绑定 `0.0.0.0` 时必须设置：

```bash
export SUPER_GPU_API_TOKEN='<long-random-token>'
super-gpu --config config.json serve --host 0.0.0.0
```

仍建议放在 VPN、SSH tunnel 或私网之后，详见 [SECURITY.md](SECURITY.md)。

## 当前边界

- 首版监控后端面向 NVIDIA `nvidia-smi`。
- GPU Util 无法可靠按单进程拆分，因此共享节点采用总 Util 保守准入。
- 显存历史峰值在并置任务场景是保守估计，宁可少放任务也避免 OOM。
- 单个 SQLite 数据库只允许一个活动调度控制器；这正是防止双重调度的安全约束。

## 测试

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
```
