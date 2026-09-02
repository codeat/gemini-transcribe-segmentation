# Gemini 3.5 Transcribe 离线转写：长段不断句问题的测试与优化

针对 `gemini-3.5-transcribe-preview`（Vertex AI）在离线会议转写场景中出现"几百上千字不断句大段"的问题，
本仓库给出复现步骤、测试脚本、结论与客户端优化方案。**全部测试均在开启说话人分离（diarization）与词级时间戳（word_timestamp）的前提下进行，优化方案不改动任何说话人标签。**

完整报告（PDF）见 [docs/](docs/)。

## 结论速览

| 发现 | 内容 |
|---|---|
| 分段单位 | 返回的每个 `part` 就是一个说话人轮次，轮次内部不按句切分；段长完全取决于说话人分离何时切换 |
| 触发条件 | 中文音频传入 `language_codes=["cmn-Hans-CN"]` 时稳定出现 272 秒 / 1232 字单段（重跑逐字相同，确定性）；传 `zh-CN` 更严重；不传语言提示则正常 |
| 与语速无关 | 长段语速 4.6 字/秒反而低于短段 5.3 字/秒；大段内部有 26 处 ≥1 秒停顿，缺的只是标签切换 |
| 文字提示无效 | 在音频旁附加文字指令（前 / 后 / 中 / 英）输出逐字相同，prompt token 不变，文本在服务端被丢弃；模型卡明确不支持系统指令与多轮 |
| 尾部漏转 | 开启词级时间戳时，24 分钟中文音频有 3 次运行在 18:59 提前结束（finish_reason=STOP，非长度上限）；10 分钟分块后覆盖完整 |

## 优化方案（可叠加）

1. **客户端二次切分（主方案，零成本、确定性）**：用词级时间戳切分，三条规则：① 停顿两档，≥1.5 秒无条件断开，0.6–1.5 秒只在有标点或当前句 ≥20 字时断开；② 句末标点随句断开；③ 超过 60 字回退到最近逗号断开。
   中文崩塌版 82 段 → 223 段，最长段 1232 字 → 60 字，272 秒 → 13.5 秒；英文 34 段 → 160 段，最长 890 → 128 字符。说话人标签原样保留。
2. **请求参数**：中文不传 `language_codes`；`diarization` 与 `word_timestamp` 保持同开（关词级时间戳后最长段反弹到 745 字）。
3. **分块与校验**：长音频按 10 分钟分块；调用后校验"末词 end_offset 与音频时长之差 < 30 秒"，不满足即补跑尾段。

## 脚本

| 文件 | 作用 |
|---|---|
| `scripts/run.py` | 调用模型，保存完整返回 JSON。参数：`--lang`、`--no-wordts`、`--no-diar`、`--prompt`、`--maxtok` |
| `scripts/analyze.py` | 段数、段长、语速、说话人分布统计，定位最长段 |
| `scripts/postprocess.py` | 二次切分（Pass A 本地切分；`--rerun` 对超长轮次截窗重跑） |
| `scripts/prompt_matrix.py` | 文字提示的多种放置方式对照实验 |

### 快速开始

```bash
pip install -r requirements.txt
export GOOGLE_CLOUD_PROJECT=your-project-id      # 需已 gcloud auth application-default login
python3 scripts/run.py meeting.wav results/meeting.json
python3 scripts/analyze.py results/meeting.json
python3 scripts/postprocess.py results/meeting.json --gap 0.6 --max 60 --out results/meeting_segments.json
```

`postprocess.py` 输出的每个 segment 都带 `spk`（说话人标签）、`start`、`end`、`text`，可直接用于小屏字幕或会议纪要。

### 调用示例

```python
from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
audio = types.Part.from_bytes(data=open("meeting.wav", "rb").read(), mime_type="audio/wav")
cfg = types.AudioTranscriptionConfig(
    mode="VERBATIM",      # SMART 模式与分离 / 时间戳互斥
    diarization=True,     # 说话人分离，保持开启
    word_timestamp=True,  # 词级时间戳，二次切分的依据
    language_codes=None,  # 中文建议不传
)
resp = client.models.generate_content(
    model="gemini-3.5-transcribe-preview", contents=[audio],
    config=types.GenerateContentConfig(audio_transcription_config=cfg))
for part in resp.candidates[0].content.parts:   # 一个 part = 一个说话人轮次
    at = part.audio_transcription
    print(at.speaker_label, at.words[0].start_offset, part.text)
```

## 测试环境

- 模型 `gemini-3.5-transcribe-preview`，Vertex AI `global` 区域，Google Gen AI SDK 2.20
- 音频：英文客服电话 9 分 57 秒、中文人物访谈 23 分 37 秒，16 kHz 单声道 PCM WAV
- 18 次对照运行，一次只改一个变量；关键组各重跑两次核对确定性
- 模型为公开预览版（2026 年 9 月），后续版本行为可能变化，上线前建议用自有音频回归

## 独立裁判对照：语言提示该不该加

用通用多模态模型 Gemini 3.8 Flash 直接听音频做独立裁判（`scripts/judge_speakers.py`），判定中文访谈全程为 **2 人**、211 轮；把裁判轮次按文本对齐到各运行的 part，统计说话人切换数、标签一致率与文本相似度（10–20 分钟窗口，裁判 51 次切换）：

| 配置 | 识别出的切换 | 标签一致率 | 与裁判文本相似度 |
|---|---|---|---|
| 整文件 · 不加语言提示 | 28 | 65% | 0.891 |
| 整文件 · 加 cmn-Hans-CN | 17 | 54% | 0.894 |
| 10 分钟分块 · 不加语言提示 | 52 | 90% | 0.936 |
| 10 分钟分块 · 加 cmn-Hans-CN | 44 | 94% | 0.934 |

结论：**中文不加 `language_codes`**。加了以后识别文本没有提升（相似度差值 < 0.005），说话人切换在四组对照中三组更少，整文件场景下更把 34 轮对话并成一段。**按 10 分钟分块**的收益最大：切换 28 → 52，标签一致率 65% → 90%。所有配置均开启说话人分离与词级时间戳。

## 边界

- 说话人分离开启时单次请求音频上限 30 分钟，更长文件需分块
- 访谈类短插话密集场景，模型给出的说话人标签可能串位；二次切分不改标签，需抽检
- 原始音频与完整转写文本不在本仓库公开
