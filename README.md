# cf-gradio

把**本机 ComfyUI** 的工作流封装成一个进程：

- 🎨 **Gradio 可视化界面**：生图 / 生视频 / TTS / 数字人，按工作流 sidecar 自动生成参数表单。
- 🔌 **OpenAI 兼容接口**：`/v1/images/generations`、`/v1/images/edits`、`/v1/videos`、`/v1/audio/speech`、`/v1/models`。
- 🌐 **一个公网地址搞定两边**：`launch(share=True)` 生成的 `*.gradio.live` 链接同时暴露 UI 和 `/v1` API，适合 Kaggle / 百度飞桨 / 启智社区等不能做反向代理的免费 GPU 环境。
- 🧩 **工作流格式完全兼容 Comfyui2api**：直接复用 ComfyUI `File → Export (API)` 导出的 JSON 和 `.comfyui2api/*.params.json` 参数映射。
- 🪶 **单实例简化版**：不需要实例池、数据库、模型目录，只连本机 `127.0.0.1:8188` 的 ComfyUI。

---

## 工作原理

```
┌────────────────────────── 一个 Python 进程（一个端口） ──────────────────────────┐
│                                                                                │
│  Gradio UI  ◄──┐                                                               │
│                ├── 同一个 FastAPI app ──► gradio.live 公网隧道                   │
│  /v1 OpenAI ◄──┘         │                                                     │
│  /files 产物             │ httpx + WebSocket                                   │
│                          ▼                                                     │
│                   ComfyUI 127.0.0.1:8188                                       │
│                   ├─ /prompt  提交工作流                                        │
│                   ├─ /history 轮询完成（WS 只负责实时进度）                      │
│                   ├─ /upload/image 上传图片/音频                                │
│                   └─ /view    下载产物                                          │
└────────────────────────────────────────────────────────────────────────────────┘
```

Gradio 6 的 `launch()` 会重建内部 FastAPI app，因此本项目包装了
`gradio.routes.App.create_app`，把 `/v1`、`/files`、`/health` 注入同一个 app。
`share=True` 的隧道代理整个端口，所以页面和 API 天然同域同链接。

---

## 快速开始

### 1. 准备 ComfyUI

确保 ComfyUI 已经在本机运行：

```bash
python main.py --listen 127.0.0.1 --port 8188
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 建议 Python 3.10+；本项目在 Python 3.11 + Gradio 6.31 上验证。

### 3. 放入工作流

`workflows/` 里已经附带 5 个示例工作流（来自 Comfyui2api 仓库）。每个工作流必须由 ComfyUI
的 **File → Export (API)** 导出，参数映射放在 `workflows/.comfyui2api/<文件名>.params.json`。

### 4. 启动

```bash
# 本地使用（http://127.0.0.1:7860）
python app.py

# 生成公网地址（同时暴露 UI 和 /v1）
python app.py --share

# 指定端口 / 页面登录
python app.py --share --port 7860 --auth me:secret
```

启动后终端会打印：

```text
本地页面 : http://127.0.0.1:7860
公网页面 : https://xxxx.gradio.live
OpenAI  : https://xxxx.gradio.live/v1
健康检查 : https://xxxx.gradio.live/health
```

只检查工作流是否能加载：

```bash
python app.py --check
```

---

## 在免费 GPU 平台使用

### Kaggle

```python
# 1) 启动 ComfyUI（按你的 Kaggle 启动脚本）
# 2) 安装本项目和依赖
!git clone <你的仓库地址> cf-gradio
%cd cf-gradio
!pip install -q -r requirements.txt

# 3) 启动（推荐在单元格内直接调用，保持 notebook 运行）
import os
os.environ["GRADIO_SHARE"] = "1"
os.environ["GRADIO_AUTH"] = "me:secret"
os.environ["API_TOKEN"] = "sk-my-token"
from cf_gradio import launch
launch()            # 阻塞当前单元格；链接打印在输出里
# 或者 launch(block=False) 让单元格返回，内核继续托管服务
```

也可以在终端里运行（进程会一直阻塞，直到你停止它）：

```bash
!python app.py --share --port 7860
```

单元格会阻塞，这是正常的；保持 notebook 运行即可。公网链接有效期由 Gradio 决定（通常最长约 1 周，提前失效也正常）。

### 百度飞桨 / 启智社区

只要能访问外网、能运行 Python 进程，步骤与 Kaggle 相同。若平台禁止 frpc / gradio.live：

- 用 `--auth` 保护页面，`API_TOKEN` 保护接口；
- 或改用平台自带的端口转发/内网穿透，把 `--port 7860` 暴露出去（本项目不依赖 share 也能工作）。

> ⚠️ **公网安全**：`share` 链接就是访问凭证。未设置 `GRADIO_AUTH` / `API_TOKEN` 时，
> 任何拿到链接的人都能打开页面或调用接口、消耗你的 GPU 额度。启动日志会给出警告。

---

## Gradio 界面

| 标签页 | 能做什么 | 工作流 kind |
|---|---|---|
| 生图 | 文生图 / 图生图切换 | `txt2img` / `img2img` |
| 生视频 | 文生视频 / 图生视频切换 | `txt2video` / `img2video` |
| TTS 语音 | 输入文本合成语音 | `txt2audio` |
| 数字人 | 参考图片 + 驱动音频 → 视频 | `digitalhuman` |

- 工作流下拉会自动过滤；AIO 工作流（同一张图既支持文生图又支持图生图）会同时出现在两种模式里。
- 文生图 / 文生视频模式不要求上传图片；AIO 工作流会把 `inputImgCount` 自动设为 `0`，走文生分支。
- 参考图片支持**多张上传**（最多 8 张），按顺序映射为主图、`image2`、`image3`…；`inputImgCount`
  由运行时自动计算，首尾帧类视频工作流可一次传两张。
- 表单参数来自 sidecar：`size/steps/cfg/seed/voice/speed/...` 自动生成控件；留空即使用工作流内置默认值。
- 生成结果支持图片画廊、视频/音频播放器和产物下载。
- 底部「最近任务」可查看任务状态与耗时。

---

## 工作流与 sidecar

### 工作流

在 ComfyUI 里调好流程 → **File → Export (API)** → 保存到 `workflows/*.json`。

支持的 kind 由节点自动检测：

| 检测条件 | kind |
|---|---|
| 有 `SaveImage` | `txt2img`；若同时有 `LoadImage` 则也支持 `img2img` |
| 有 `SaveVideo` | `txt2video`；若同时有 `LoadImage` 则也支持 `img2video` |
| 有 `SaveVideo` + `LoadAudio` | `digitalhuman`（数字人） |
| 有 `SaveAudio` | `txt2audio` |

模型 id 默认由文件名生成：`moss_image_aio_api.json` → `moss-image-aio`。
也可以在 sidecar 顶层用 `"model": "my-model"`、`"display_name": "我的模型"` 覆盖。

### sidecar 参数映射

`workflows/.comfyui2api/<同名>.params.json`，格式与 Comfyui2api 完全一致：

```json
{
  "version": 1,
  "kind": "img2video",
  "model": "my-flf2v",
  "display_name": "首尾帧生视频",
  "prompt_node": "297.value",
  "negative_prompt_node": "301.text",
  "image_node": "31.image",
  "audio_node": "12.audio",
  "parameters": {
    "size": {
      "type": "size",
      "maps": [
        {"target": "291.value", "part": "width"},
        {"target": "292.value", "part": "height"}
      ]
    },
    "duration": {
      "type": "float",
      "minimum": 1,
      "maximum": 10,
      "default": 4,
      "description": "视频时长（秒）",
      "maps": [{"target": "294.value"}]
    },
    "seed": {"type": "int", "maps": [{"target": "274.noise_seed"}]},
    "fps": {"type": "int", "maps": [{"target": "296.value"}]}
  }
}
```

- `node_id.input_key` 直接对应 API JSON 里的节点与输入字段。
- 支持 `type`: `size` / `int` / `float` / `bool` / `string` / `image`。
- 支持 `minimum` / `maximum` / `default` / `required` / `description`。
- `maps[].transform = "seconds_to_frames"` 可把秒数按 `fps` 自动换算成帧数。
- `maps[].selector` 支持按 `class_type` / 节点标题定位（节点 id 会变时更稳）。
- `negative_prompt_node_with_image` 用于 AIO 工作流：有图时走另一条负向编码分支。
- 额外图片会按顺序自动填入 sidecar 声明的 `image2`、`image3`… 参数。
- `inputImgCount` 由运行时按实际图片数自动填充（0 表示文生分支），不要手填。

### 数字人工作流

任何“参考图 + 驱动音频 → 视频”的 ComfyUI 流程都可以接入，只要它包含 `LoadImage`、`LoadAudio`
和 `SaveVideo` 节点。在 sidecar 里写：

```json
{
  "version": 1,
  "kind": "digitalhuman",
  "prompt_node": "3.text",
  "image_node": "1.image",
  "audio_node": "2.audio",
  "parameters": {}
}
```

放入 `workflows/` 后，数字人标签页会自动出现该工作流。

---

## OpenAI 兼容接口

Base URL 为公网或本地地址的 `/v1`，认证方式：

```http
Authorization: Bearer <API_TOKEN>
```

未设置 `API_TOKEN` 时不校验。所有错误使用 OpenAI 风格：

```json
{"error": {"message": "...", "type": "invalid_request_error", "code": "model_not_found"}}
```

### 模型列表

```bash
curl -s https://xxxx.gradio.live/v1/models
```

返回 `id` 即后续请求的 `model`。模型省略时，默认取该 kind 的第一个可用工作流。

### 文生图

```bash
curl -s -X POST https://xxxx.gradio.live/v1/images/generations \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moss-image-aio",
    "prompt": "a cute cat, pixel art",
    "size": "1024x1024",
    "steps": 8,
    "seed": -1,
    "n": 1
  }'
```

同步返回：

```json
{"created": 1700000000, "data": [{"url": "https://xxxx.gradio.live/files/..."}]}
```

- `response_format`: `url`（默认）或 `b64_json`。
- 异步：加 `x-comfyui-async: 1`，立即返回 `{"job_id": "...", "status": "pending"}`，
  再用 `GET /v1/jobs/{job_id}` 轮询。

Python（openai SDK）：

```python
from openai import OpenAI

client = OpenAI(base_url="https://xxxx.gradio.live/v1", api_key="sk-my-token")
result = client.images.generate(
    model="moss-image-aio",
    prompt="a cute cat, pixel art",
    size="1024x1024",
)
print(result.data[0].url)
```

### 图生图 / 图片编辑

```bash
curl -s -X POST https://xxxx.gradio.live/v1/images/edits \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "model=moss-image-aio" \
  -F "prompt=make it cinematic" \
  -F "image=@input.png"
```

也支持 JSON + 图片 URL / base64 / data URL：

```json
{
  "model": "moss-image-aio",
  "prompt": "make it cinematic",
  "image": "data:image/png;base64,...",
  "size": "1024x1024"
}
```

多图输入：重复 `image` / `input_reference` 字段，或传 `images` 数组。服务会按顺序映射
主图 → `image2` → `image3`…，并自动计算 `inputImgCount`（例如 qwen AIO 支持 3 张参考图）：

```bash
curl -s -X POST https://xxxx.gradio.live/v1/images/edits \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "model=qwen-image21-turbo-aio" \
  -F "prompt=combine these references" \
  -F "image=@first.png" \
  -F "image=@second.png"
```

AIO 工作流也可以直接用于 `/v1/images/generations`：不带图片时 `inputImgCount=0`，自动走文生分支。

### 视频

`POST /v1/videos` 创建异步任务，字段兼容 OpenAI Video API：

```bash
# 图生视频（可传多张：主图 + 尾帧）
curl -s -X POST https://xxxx.gradio.live/v1/videos \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "model=kaggle-ltx2-5-flf2v" \
  -F "prompt=camera pans right" \
  -F "seconds=4" \
  -F "input_reference=@first_frame.png" \
  -F "input_reference=@last_frame.png"

# 文生视频（不带 input_reference；AIO 视频模型会自动 inputImgCount=0）
curl -s -X POST https://xxxx.gradio.live/v1/videos \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"kaggle-minimax-h3-ti2v","prompt":"waves at sunset","seconds":4}'
```

返回：

```json
{"id": "video_xxx", "object": "video", "status": "queued", "progress": 0}
```

轮询与下载：

```bash
curl -s https://xxxx.gradio.live/v1/videos/video_xxx
curl -s -o out.mp4 https://xxxx.gradio.live/v1/videos/video_xxx/content
```

状态值：`queued` → `in_progress` → `completed` / `failed`。

### TTS

```bash
curl -s -X POST https://xxxx.gradio.live/v1/audio/speech \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm2-base",
    "input": "你好，欢迎使用语音合成。",
    "voice": "default",
    "speed": 1.0
  }' --output speech.wav
```

返回音频字节，实际格式由工作流的 `SaveAudio` 节点决定（wav/mp3/…），`response_format`
字段会被接受但不做转码。

### 数字人

```bash
curl -s -X POST https://xxxx.gradio.live/v1/digital-human \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "model=my-digital-human" \
  -F "image=@face.png" \
  -F "audio=@speech.wav" \
  -F "prompt=natural talking"
```

返回结构同 `/v1/videos`；用 `GET /v1/videos/{id}` 轮询、`/content` 下载。

### 高级参数

任何 sidecar 声明的参数都可以直接放在请求顶层（例如 `prompt_text`、`voice`、`fps`）。
需要精确改任意节点时用 `overrides`：

```json
{
  "model": "moss-image-aio",
  "prompt": "a cat",
  "overrides": {"72.seed": 123, "37.steps": 6, "46.steps": 6}
}
```

---

## 环境变量

复制 `.env.example` 为 `.env`，或直接设置环境变量（环境变量优先）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `COMFYUI_BASE_URL` | `http://127.0.0.1:8188` | 本机 ComfyUI 地址 |
| `COMFYUI_AUTH_TOKEN` | 空 | ComfyUI 网关的 Bearer token |
| `WORKFLOWS_DIR` | `./workflows` | 工作流目录 |
| `OUTPUTS_DIR` | `./outputs` | 任务产物目录 |
| `API_TOKEN` | 空 | 设置后 `/v1/*` 需要 Bearer |
| `PUBLIC_BASE_URL` | 空 | 覆盖返回媒体 URL 的前缀（默认用请求 Host，gradio.live 会自动正确） |
| `TIMEOUT_S` | `3600` | 单个任务最长等待时间 |
| `HTTP_TIMEOUT_S` | `600` | 访问 ComfyUI HTTP 的超时 |
| `POLL_INTERVAL_S` | `0.5` | `/history` 轮询间隔 |
| `MAX_UPLOAD_MB` | `30` | 上传文件大小上限 |
| `MAX_IMAGE_BYTES` | `20000000` | 图片 URL/上传大小上限 |
| `ALLOW_PRIVATE_IMAGE_URLS` | `0` | 放开内网图片 URL（仅调试） |
| `INPUT_SUBDIR` | `cfgradio` | 上传到 ComfyUI input 的子目录 |
| `GRADIO_SHARE` | `0` | 是否默认开启 gradio.live |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | 监听地址 |
| `GRADIO_SERVER_PORT` | `7860` | 监听端口 |
| `GRADIO_AUTH` | 空 | 页面登录，格式 `user:pass` |
| `GRADIO_AUTH_MESSAGE` | `请输入访问账号` | 登录提示 |

---

## 目录结构

```text
cf-gradio/
├── app.py                     # CLI 入口
├── cf_gradio/
│   ├── config.py              # 环境变量 / .env
│   ├── comfy_client.py        # ComfyUI HTTP + WebSocket
│   ├── workflow.py            # 工作流解析、kind 检测、prompt/image/audio 注入
│   ├── params.py              # sidecar 参数校验与映射
│   ├── registry.py            # 工作流扫描、模型 id
│   ├── jobs.py                # 任务提交、进度、产物下载、持久化
│   ├── openai_api.py          # OpenAI 兼容路由 + /files + /health
│   ├── ui.py                  # Gradio 页面（动态表单）
│   └── server.py              # 组装 + share 启动
├── workflows/                 # 工作流 JSON + .comfyui2api sidecar
├── outputs/                   # 运行时产物（gitignore）
├── tests/                     # unittest + fake ComfyUI
└── requirements.txt
```

---

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖：

- 5 个自带工作流与 sidecar 全部加载；
- kind 检测（含 AIO 多 kind、数字人）；
- prompt / image / audio 节点注入；
- 参数映射（size / seed / voice / seconds_to_frames）；
- 用进程内 fake ComfyUI 跑通生图、图生图、视频、TTS、数字人的完整 HTTP 链路。

无 GPU 想手动点 UI：

```bash
python tests/dev_server.py --port 7864
```

它会启动一个假 ComfyUI，界面和接口都能正常跑通。

---

## 限制与说明

- **单 ComfyUI 实例**：不包含多后端池、健康巡检、模型目录数据库；这些在 Comfyui2api 里。
- **无 chat/completions 兼容层**：只提供标准 images / videos / audio / models。
- **TTS 不转码**：返回工作流实际输出的音频格式。
- **产物链接**：`/files/<job_id>/<filename>` **有意不做鉴权**，方便浏览器/Agent 直接嵌入和下载产物；
  它只提供任务 outputs 中登记的实际产物，不会暴露 `job.json` / `prompt.json` / `history.json`。
  job_id 随机不可枚举；分享链接本身就是访问凭证，建议配合 `GRADIO_AUTH` / `API_TOKEN` 使用。
- **share 链接**：由 Gradio 官方 FRP 服务提供，有效期与可用性取决于 Gradio，不保证长期稳定。
- **任务重启**：服务重启后，未完成的任务会标记为失败；已完成任务的记录和产物会从
  `outputs/*/job.json` 恢复。

---

## 与 Comfyui2api 的关系

本项目复用了 Comfyui2api 中经过验证的工作流解析、sidecar 参数映射、ComfyUI 提交/轮询/产物
下载逻辑，并删除了多实例调度、SQLite 模型目录、React 管理台等复杂部分，改用 Gradio 动态表单
和单进程 OpenAI 路由，目标是在免费 GPU 平台上“一个格子跑起来、一个链接给出去”。
