# 羽毛球视频自动剪辑

本程序在本机分析羽毛球视频，利用球员姿态、动作和羽毛球检测估计回合边界，并将符合最短时长的片段合并导出。默认使用 context 模式；识别边界是视觉估计，建议先用 dry-run 核对。

程序本身没有视频上传步骤。setup_models.py 会从 Hugging Face 和 GitHub 下载并校验模型权重。

## 安装

建议使用 Python 3.10 及以上版本。Windows PowerShell：

    py -3.10 -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    & .\.venv\Scripts\python.exe setup_models.py

模型权重保存在 models/，该目录已排除在 Git 提交之外。

## 快速运行

不带参数运行时，读取仓库目录下的 input.mp4，分析原视频 [0, 450) 秒，输出到 output/：

    & .\.venv\Scripts\python.exe badminton_cut.py

也可以指定视频、分析区间和输出目录：

    & .\.venv\Scripts\python.exe badminton_cut.py ".\match.mp4" --start-seconds 120 --end-seconds 300 --output ".\results"

end-seconds 是原视频上的结束时间（不包含该秒），不是片段长度；超过视频长度时会截到视频结尾。输出结构：

    output/<视频名>/<运行时间>/highlights.mp4
    output/<视频名>/<运行时间>/merged_starts.log

日志同时记录原视频与合并后时间轴，例如：

    export=32.220-44.250s ; merged=0.000-12.030s

## 按视频调整的主要参数

| 参数 | 默认值 | 作用与修改方法 | 位置 |
|---|---:|---|---|
| 输入 | 仓库下 input.mp4 | 推荐命令行传入视频路径；或修改默认文件名 | auto_pipeline.py：DEFAULT_INPUT、build_parser() |
| 输出 | 仓库下 output/ | 可用 --output 指定 | auto_pipeline.py：DEFAULT_OUTPUT、build_parser() |
| 分析起点 | 0 秒 | 跳过片头等无关区段：--start-seconds | auto_pipeline.py：DEFAULT_START_SECONDS |
| 分析终点 | 450 秒 | 长视频需调大：--end-seconds；是原视频时间且右端不包含 | auto_pipeline.py：DEFAULT_END_SECONDS |
| 标注参考时刻 | 12 秒 | 固定机位选球场和球员清楚的帧：--setup-seconds；不改变分析区间 | auto_pipeline.py：DEFAULT_SETUP_SECONDS |
| pre-roll | 2.5 秒 | 估计开局前额外保留；固定机位用于发球准备；受分析起点限制 | Settings.pre_roll；--pre-roll |
| post-roll | 2 秒 | 估计结束后额外保留；不计入最短时长，且受捡球和场景边界约束 | Settings.post_roll；--post-roll |
| t-limit | 8 秒 | 最短保留时长；前滚计入、后滚不计入 | Settings.t_limit；--t-limit |
| frame-stride | 基础值 5 帧 | 姿态分析每隔多少个源视频帧采样一次；越小越慢但瞬间动作漏检更少 | Settings.frame_stride；config.auto.json 或 --frame-stride |

固定机位 context 模式在没有显式配置 frame-stride 时会自动改为 4。命令行或 config.auto.json 明确设置后则使用指定值；启动输出会显示最终值。

参数优先级：Python 常量提供默认输入、输出和分析区间；Settings 提供检测默认值；config.auto.json 覆盖 Settings；命令行选项覆盖对应配置。temporal 对象的默认值和校验在 temporal_events.py 的 TemporalSettings 中。

## 检测参数

一般先调分析区间、标注帧和前后滚动时间；需要时再改 config.auto.json。

| 参数（config.auto.json） | 当前值 | 作用 |
|---|---:|---|
| imgsz | 1280 | 羽毛球模型推理尺寸；提高会增加计算量 |
| pose_imgsz | 768 | 姿态模型尺寸；远处球员漏检时可尝试提高；须至少 320 且为 32 的倍数 |
| confidence | 0.18 | 羽毛球检测置信度下限；降低会保留更多候选，也可能增加误检 |
| max_players | 2 | 无人工标注时的自动跟踪人数上限 |
| shuttle_fps | 5（代码默认） | 羽毛球模型每秒最多检测次数；姿态频率由 frame-stride 控制 |

模型文件默认位于 models/：羽毛球模型为 shuttle_yolo11s.pt，姿态模型为 yolo11n-pose.pt；可用 --weights 和 --pose-weights 换路径。--device 默认为 cpu，可用 --device 0 选择可用的 NVIDIA GPU。

temporal 中与回合边界相关的默认值：

| 参数 | 值 | 作用 |
|---|---:|---|
| action_gap | 3.0 秒 | 动作证据允许的间隔；调大可能把停顿或回位并入回合 |
| rally_stitch_gap | 8.0 秒 | 尝试合并动作候选的最大间隔；长间隔还需要中间有对打证据 |
| rally_stitch_evidence_gap | 3.0 秒 | 长间隔合并时允许的中间证据空档 |
| reaction_backtrack | 0.6 秒 | 仅转播模式使用，从首次可靠动作前回退以包含接发球反应 |
| serve_backtrack | 0.3 秒（代码默认） | 固定机位从首次明显动作前回退，补入发球准备；位置在 temporal_events.py 的 TemporalSettings |
| terminal_flight | 0.8 秒 | 最后动作后的飞行余量 |
| end_confirm | 3.0 秒 | 结束确认时观察后续动作；须不小于 action_gap |
| arm_threshold | 0.8 | 手臂动作门限；降低会增加候选，也可能把走路摆臂误作击球 |
| low_hand_arm_threshold | 1.2 | 较低手位下的动作门限 |
| pickup_confirm | 0.2 秒 | 捡球姿态持续多久才作为捡球证据 |

这些是启发式参数，机位、分辨率和球员大小都会影响效果。一次只改少量参数，先 dry-run 检查；不要单纯降低动作门限来补漏，否则会增加走路误收。

analysis_fps、pose_fps、pose_confidence 和旧版 players 区域是兼容字段；当前姿态采样由 frame-stride 控制，调整这些字段不会改变当前检测行为。

## 标注与模式

默认打开人工标注界面，选择球场和 2–4 名球员。固定机位建议保留标注。

- --mode context：默认，结合姿态动作和羽毛球证据估计边界。
- --mode strict：要求更严格的羽毛球事件证据，可能漏掉球未检出的真实回合。
- --view-mode auto：默认视角模式。
- --view-mode fixed：固定机位。
- --view-mode broadcast：转播模式，需配合 --no-manual-setup。
- --no-manual-setup：跳过标注并自动检测球员。
- --dry-run：分析但不编码 highlights.mp4。
- --debug-events：写入 event_debug.json，内含逐采样点姿态和边界证据。
- --debug-video：生成调试视频。

完整选项：badminton_cut.py --help。

## 测试

    & .\.venv\Scripts\python.exe -m unittest discover -s . -p "test_*.py" -v

测试使用合成图像、姿态和短视频，不依赖个人比赛视频；临时数据会自动清理。

## 隐私检查

原视频、模型、虚拟环境、运行输出、调试 JSON 和本地开发资料均应留在本机；.gitignore 已排除这些类型。merged_starts.log 含回合时间点，event_debug.json 还含逐采样点的姿态证据，分享前先检查。不要提交令牌、密码、个人视频或人工标注样本。

公开前还要检查 Git 提交历史：修改当前文件不会清除旧提交中的本机路径或提交者邮箱。
