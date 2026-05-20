## 项目快速上手（为 AI 助手定制）

目标：帮助 AI 代理快速理解 AV-ALOHA 的架构、常用工作流、关键约定与可被调用的脚本，便于实现代码更改、调试与自动化任务。

1) 大体架构与组件
- 代码基于 ROS (Noetic, catkin) 与 PyTorch/LeRobot 训练代码混合。主要路径：
	- `src/av-aloha/`：项目根，包含 ROS-launch / data collection / Python 包（`lerobot`, `gym_guided_vision`）
	- `data_collection_scripts/`：记录/回放/可视化脚本，用于真实机器人与仿真的数据采集
	- `lerobot/lerobot/`：训练、评估与数据处理（hydra 配置、scripts）
	- `eval_scripts/`：离线/在线评估脚本

2) 常用开发与运行命令（精确示例）
- 构建 ROS 工作区：在仓库根（workspace）运行：
	- `cd ~/interbotix_ws && catkin_make`，然后 `source devel/setup.bash`
- Python 环境（推荐）：
	- `conda create -y -n lerobot python=3.10 && conda activate lerobot`
	- 安装依赖示例：`pip install -e gym_guided_vision && pip install -e lerobot && pip install -r requirements.txt`
- 训练示例（在仓库根）：
	- `python lerobot/lerobot/scripts/train.py device=cuda env=sim_sew_needle_3arms policy=zed_static_wrist_act hydra.run.dir=outputs/...`
- 记录仿真数据：
	- `python data_collection_scripts/record_sim_episodes --task_name sim_insert_peg --episode_idx 0`
- 在真实机器人上录制：
	- 先 `source data_collection_scripts/launch_robot.sh`，另开终端 `source data_collection_scripts/activate.sh && python record_episodes --task_name occluded_insertion --episode_idx 0`

3) 项目特有约定（谨记）
- 使用 ALOHA/AV-ALOHA 的设备命名与端口绑定（例如：手臂绑定到 `/dev/ttyDXL_puppet_middle`）。修改设备绑定请在 `data_collection_scripts` 中查找相关 `launch`/`env` 文件。
- 数据格式：episode 存为 HDF5，上传/下载与 Hugging Face hub 协作使用 `lerobot/lerobot/scripts/*`（例如 `push_dataset_to_hub.py`、`visualize_dataset.py`）。
- Hydra 配置主导训练/评估参数，常见配置目录：`lerobot/lerobot/configs`。训练/评估脚本通过 `hydra.run.dir` 指定输出目录。

4) 集成点与外部依赖
- ZED 相机：依赖 ZED Python API（必须额外安装，见 README 中的链接）。
- Hugging Face：用于数据集与模型存储（`huggingface-cli login` 后使用 `push_dataset_to_hub.py`）。
- Unity / WebRTC：远程 VR teleop 与视频 passthrough，信令配置位于 `data_collection_scripts/signalingSettings.json`，需要 `serviceAccountKey.json`（Firebase）。

5) 测试与 CI 线索
- `lerobot` 包含测试与 Github Actions（见 `lerobot/.github/workflows` 与 `lerobot/Makefile`）。本地运行示例：
	- `DATA_DIR=tests/data pytest -sx tests/test_something.py`（见 PR 模板示例）

6) 可参考与易修改的文件（热点）
- `src/av-aloha/README.md`：总体说明与命令示例（首要参考）。
- `data_collection_scripts/`：录制、回放、可视化与 launch 脚本（调真实设备与仿真）。
- `lerobot/lerobot/scripts/train.py`、`eval.py`：训练/评估入口，使用 hydra 参数化。
- `lerobot/lerobot/configs/`：配置集合，修改实验行为请优先在此处改动。

7) 风险与注意事项
- 真实硬件修改慎重：任何更改设备端口或启动脚本前都要告知并在仿真中先验证。
- CUDA/torch 版本与本地驱动需匹配（README 建议的安装命令是首选）。

如果需要，我可以：
- 将这份文件合并到仓库根的 `.github/copilot-instructions.md`（不是子目录）或把同样内容同步到其他包；
- 根据你想要的 agent 角色（修 bug / 写特性 / 生成数据）把说明调成更偏重的子集。请告诉我你更偏好的 agent 职能。 
