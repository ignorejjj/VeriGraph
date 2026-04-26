# VeriGraph RL 迁移计划（基于 `verl-datamind`，尽量少改）

## 目标

在 `verl-datamind` 现有的 DataMind multi-turn RL 链路上，最小改动接入我的 `VeriGraphAgent`。

对齐目标：

- rollout 走 multi-turn
- assistant 输出格式对齐当前 agent：`<think>...</think>n<code_interpreter>...</code_interpreter>`
- tool observation 对齐当前 agent：下一轮作为 `user` 消息注入 `<tool_response>...</tool_response>`
- rollout 结束条件对齐当前 agent：以 `submit_answer()` / executor 的 `submission_finished` 为准，不依赖 `<answer>`
- 第一版先不做 history compression
- 第一版继续复用 `recipe.dapo.main_dapo` 和 `multi.sh`，不新起一套 trainer

明确不建议第一版做的事：

- 不要继续往 `verl-datamind/agent/async_interpreter.py` 里塞 VeriGraph 逻辑
- 不要一开始就大改 `fsdp_workers.py` / `dapo_ray_trainer.py`
- 不要保留 DataMind 的 tool schema 注入；这会让 RL prompt 和当前 SFT agent 不一致

原因：

- `VeriGraph` 已经有更干净的持久化 Python 执行器：`agents.core.code_executor.CodeExecutor`
- `dapo_ray_trainer.py` 现在虽然有一些 DataMind 特化逻辑，但只要 rollout 返回它需要的 `trajectory`，第一版其实不用大动它
- 当前最关键的是先把 “prompt 对齐、rollout 状态机、reward、脚本联调” 路跑通

---

## 先说结论：一共分 4 个大步骤

1. 数据入口和 `multi.sh` 改成 VeriGraph 版本
2. rollout / schema 最小改造，接入 `CodeExecutor + submit_answer`
3. reward 改成 VeriGraph 的 final-claims judge
4. 小规模联调和验证，再决定要不要补 history compression

---

## 第 1 步：数据入口和 `multi.sh`

### 这一步要解决什么

把当前原始 VeriGraph 样本转成 `verl-datamind` 能直接读的 RL parquet，同时把 `multi.sh` 从 DataMind 的 SQL/CSV 配置改成 VeriGraph 的配置。

### 建议新增/修改的文件

- 新增 `verl-datamind/recipe/verigraph/preprocess_verigraph_rl.py`
- 修改 `verl-datamind/multi.sh`
- 修改 `verl-datamind/recipe/dapo/main_dapo.py`

### RL parquet 建议字段

沿用 `verl-datamind/verl/utils/dataset/rl_dataset.py` 的读法，样本建议输出成：

- `prompt`
- `data_source = "verigraph"`
- `ability = "verigraph"`
- `reward_model = {"style": "llm_judge", "ground_truth": ...}`
- `working_dir`
- `extra_info = {index, id, question, files, working_dir, context_dir, dir_id, judge_prompt_type, has_ground_truth}`

其中：

- `prompt` 只放初始消息，不放整条 SFT 轨迹
- `prompt` 结构直接对齐 `VeriGraphAgent.build_init_prompt()`：`system + user`
- `user` 内容继续用：`question + 可选 "(Attach files: ...)"` 后缀
- `working_dir` 指向当前样本的 context 目录
- `ground_truth` 优先从 `answers / ground_truth / answer / reference_answer / final_claims` 等字段里取

### 这里要注意的关键点

- RL 数据不需要预先塞 multi-turn assistant/tool 轨迹；online rollout 自己生成
- 当前 SFT 还不是 multi-turn，不影响 RL；RL 真正的 multi-turn 对齐发生在 rollout，而不是数据文件本身
- 如果一部分训练样本没有 `answers` / `ground_truth`，第一版建议在预处理时直接过滤掉，或者单独切分出去

### `multi.sh` 建议怎么改

保留 `recipe.dapo.main_dapo` 入口，只把 DataMind 私有变量替换掉。

建议删除或不再使用的 DataMind 变量：

- `pred_csv_result_dir_parent`
- `gold_csv_results_dir`
- `db_schema_data_path`
- `csv_folder`
- `working_path` / `working_tmp_path` 这种全局 SQL workspace 语义

建议新增的变量：

- `RAW_JSON`
- `CONTEXT_ROOT_DIR`
- `RL_DATA_DIR`
- `VERIGRAPH_JUDGE_MODEL`
- `VERIGRAPH_JUDGE_API_BASE`
- `VERIGRAPH_JUDGE_API_KEY`

训练命令里建议新增/改成：

- `reward_model.reward_manager=verigraph`
- `actor_rollout_ref.rollout.multi_turn.enable=True`
- `actor_rollout_ref.rollout.multi_turn.max_turns=${max_turns}`
- `actor_rollout_ref.rollout.multi_turn.tool_config_path=null`
- `+actor_rollout_ref.rollout.multi_turn.verigraph.enable=True`
- `+actor_rollout_ref.rollout.multi_turn.verigraph.max_tool_output_chars=1200`
- `+actor_rollout_ref.rollout.multi_turn.verigraph.tool_execution_timeout_seconds=120`
- `+reward_model.reward_kwargs.process_weight=0.5`
- `+reward_model.reward_kwargs.final_weight=0.5`
- `+reward_model.reward_kwargs.judge_model=${VERIGRAPH_JUDGE_MODEL}`
- `+reward_model.reward_kwargs.judge_api_base=${VERIGRAPH_JUDGE_API_BASE}`
- `+reward_model.reward_kwargs.judge_api_key=${VERIGRAPH_JUDGE_API_KEY}`

第一版建议把 `max_turns` 先压到一个比较保守的值做 smoke test，比如 `4~8`，因为暂时不做 history compression。

### `main_dapo.py` 为什么也要改

当前 `recipe/dapo/main_dapo.py` 只把 DataMind 的 CSV reward 配置传给 reward manager，没有把 `reward_model.reward_kwargs` 透传进去。

所以这里需要把：

- `config.reward_model.reward_kwargs`

一起 merge 进 reward manager 的初始化参数里，否则 `judge_model / judge_api_base / process_weight / final_weight` 这些都到不了新的 VeriGraph reward manager。


其他:
需要移除db_id这种字段。verigraph的data里面每个item有dir_id作为query所需要文件的存储位置。需要把原先的一些东西删掉。





---

## 第 2 步：rollout / schema 最小改造

### 这一步要解决什么

让 `verl-datamind` 的 multi-turn rollout 在 VeriGraph 模式下：

- 解析 `<code_interpreter>`
- 用持久化 Python executor 执行
- 把 `<tool_response>` 当成下一轮 `user` 消息
- 在 `submit_answer()` 后立刻结束 rollout

### 建议新增/修改的文件

- 新增 `verl-datamind/verl/utils/verigraph_rl.py`
- 修改 `verl-datamind/verl/workers/rollout/sglang_rollout/sglang_rollout.py`
- 修改 `verl-datamind/verl/workers/rollout/schemas.py`
- 修改 `verl-datamind/verl/trainer/ppo/ray_trainer.py`

### `verl/utils/verigraph_rl.py` 建议放什么

第一版只放最需要的几件事，不要把顶层 `verl/` 那套 async callback / history compression 全搬进来：

- `build_verigraph_messages(question, files)`
- `extract_last_code_block(text)`，解析 `<code_interpreter>...</code_interpreter>`
- `VeriGraphTrajectoryRuntime`，内部直接复用 `agents.core.code_executor.CodeExecutor`
- `format_tool_response(...)`，输出 `<tool_response>` 里的观察文本
- `extract_final_claims(...)`
- `is_submission_finished(...)`

这里直接参考顶层已有的 `verl/verl/utils/verigraph_rl.py` 即可，但第一版不要把 custom token rebuild / compression 当成必需项。

### 为什么不要改 `agent/async_interpreter.py`

因为那条链路是 DataMind 的 SQL/CSV interpreter 逻辑，里面有大量 VeriGraph 不需要的假设：

- `db_id`
- `task_id`
- `execute_sql`
- CSV/SQLite 输出路径

VeriGraph 需要的是：

- 持久化 Python 环境
- `submit_answer()` 结束信号
- claim / graph 导出

这和 `CodeExecutor` 是天然匹配的，直接复用更干净。

### `schemas.py` 至少要补两个字段

在 `AsyncRolloutRequest` 里加：

- `working_dir: Optional[str] = None`
- `trajectory_info: Dict[str, Any] = {}`

原因：

- rollout 执行时每个样本都要有自己的 workspace/context 目录
- reward 不能再从 `<answer>` 里抽结果，要从 rollout 产出的结构化信息里读 final claims / submit 状态

### `sglang_rollout.py` 里 VeriGraph 模式要改哪些点

建议加一个独立分支，比如：

- `_is_verigraph_mode()`

只在这个分支里走 VeriGraph 逻辑，DataMind 原来的 SQL/CSV 路径保持不动。

需要改的核心点：

1. 解析 assistant 输出

- DataMind 现在找的是 `<code>...</code>`
- VeriGraph 要改成找 `<code_interpreter>...</code_interpreter>`

2. 执行工具

- DataMind 现在 `_execute()` 里分 SQL / CSV
- VeriGraph 分支不要复用 `_execute()`，而是直接：
  - 从 request 里拿 `working_dir`
  - 创建 `VeriGraphTrajectoryRuntime`
  - 调 `runtime.aexecute(code)`

3. observation 回灌方式

- 第一版不要引入 `tool` role
- 继续沿用 datamind 当前的 multi-turn 方式：把 `<tool_response>...</tool_response>` 作为下一轮 `user` 消息写回

这样做的好处是：

- 和你现在的 `CodeActAgent.arun_multiturn()` 一致
- `dapo_ray_trainer.get_sft_inputs()` 当前只支持 `system/user/assistant`，不用为此大改 trainer

4. 停止条件

- VeriGraph 模式下，不看 `<answer>`
- 每次 code 执行后立刻检查 `submission_finished`
- 如果 `submit_answer()` 已触发：
  - `submitted_answer_present=True`
  - 结束 rollout

5. 非法轨迹判定

VeriGraph 模式下，如果模型在未 `submit_answer()` 的情况下：

- 直接停止生成
- 或输出纯文本但没有 code call

则应视为 invalid trajectory，而不是当成正常 answer 结束。

6. 结构化 rollout 结果

rollout 至少要往 `trajectory_info` 里写：

- `tool_stats = {total_calls, successful_calls, failed_calls, process_reward}`
- `final_claims`
- `submitted_answer_present`
- `valid_trajectory`
- `termination_reason`

第一版可以不把完整 `graph` 放进 batch，避免 non-tensor 变太大；如果后面 debug 需要，再加。

7. 返回 `trajectory`

`recipe/dapo/dapo_ray_trainer.py` 更新 actor 时吃的是：

- `batch.non_tensor_batch["trajectory"]`

而不是 `messages`。所以这里要保证 rollout 输出里有：

- `trajectory = 当前多轮 messages`

最好同时保留：

- `messages`
- `trajectory`

这样最稳，不用立刻去改 `dapo_ray_trainer.py`。

8. 解析 `working_dir`

在 `_preprocess_prompt_to_async_rollout_requests()` 里，从下面两个地方择一取值：

- 样本顶层 `working_dir`
- `extra_info["working_dir"] / extra_info["context_dir"]`

然后塞到 `AsyncRolloutRequest.working_dir`。

### `ray_trainer.py` 这一处必须改

当前 `verl-datamind/verl/trainer/ppo/ray_trainer.py` 在 multi-turn 模式下强制要求：

- `tool_config_path is not None`

但 VeriGraph 恰恰不应该走 tool schema 注入；否则 prompt 里会被偷偷塞进 `<tools>` 或 JSON schema，和你现在 SFT agent 的输入不一致。

所以这里要改成：

- 普通 datamind multi-turn：仍然要求 `tool_config_path`
- `verigraph.enable=True`：允许 `tool_config_path=null`

这是第一版里非常关键的一刀。

---

## 第 3 步：reward 改成 VeriGraph final-claims judge

### 这一步要解决什么

DataMind 现在的 reward 逻辑是围绕：

- `<answer>`
- SQL/CSV 执行结果
- template / answer / execution score

VeriGraph 不该复用这套逻辑。它需要的是：

- 过程奖励：tool 使用质量
- 最终奖励：judge `final_claims` 是否回答/覆盖了 ground truth

### 建议新增/修改的文件

- 新增 `verl-datamind/verl/workers/reward_manager/verigraph.py`
- 修改 `verl-datamind/verl/workers/reward_manager/__init__.py`
- 修改 `verl-datamind/recipe/dapo/main_dapo.py`

### reward 公式建议

第一版直接用最简单的一版：

- `process_reward = successful_calls / total_calls`
- 如果 `valid_trajectory == 0` 或 `total_calls == 0`，则 `process_reward = 0`
- `final_reward = judge(final_claims, ground_truth | reference answer | answers)`
- `total_reward = process_weight * process_reward + final_weight * final_reward`

推荐默认值：

- `process_weight = 0.5`
- `final_weight = 0.5`

### reward manager 读取什么输入

从 batch 里读：

- `trajectory_info.final_claims`
- `trajectory_info.submitted_answer_present`
- `trajectory_info.valid_trajectory`
- `trajectory_info.tool_stats`
- `extra_info.question`
- `extra_info.dir_id`
- `extra_info.judge_prompt_type`
- `reward_model.ground_truth`

### judge 建议

建议直接参考顶层已有的：

- `verl/verl/workers/reward_manager/verigraph.py`

保留两种 prompt：

- `qa`
- `research`

路由规则：

- `dir_id` 包含 `research` 时走 research judge
- 否则走 qa judge

### 这里不要做的事情

- 不要再从 `<answer>` 抽最终答案
- 不要沿用 `reward_score/sql.py`
- 不要把 DataMind 的 `template_score / answer_score / execution_score` 硬改成 VeriGraph 版本

最小做法就是新增一个干净的 reward manager，然后在 `multi.sh` 里切过去。

---

## 第 4 步：联调顺序和验收

### 第一轮联调建议顺序

1. 先跑数据预处理

- 确认 parquet 能被 `RLHFDataset` 正常读进来
- 抽 1 条样本检查 `prompt` 是否只有 `system + user`
- 确认 `working_dir` / `extra_info` / `ground_truth` 都在

2. 再只跑 rollout smoke test

- `rollout.n=1`
- `train_batch_size=1`
- `max_turns=3~4`
- 只抽 1~2 条样本

重点确认：

- prompt 里没有 tool schema
- assistant 的 `<code_interpreter>` 能被解析
- `<tool_response>` 确实回灌成下一轮 `user`
- `submit_answer()` 后能立刻停止
- rollout 输出里有 `trajectory` 和 `trajectory_info`

3. 再跑 reward smoke test

重点确认：

- `trajectory_info.final_claims` 能进入 reward manager
- judge prompt 里能拿到 `question + final_claims + ground_truth`
- `process_reward / final_reward / total_reward` 都能打出来

4. 最后再开一个极小训练

建议：

- `n=2`
- `train_batch_size=2`
- `max_turns=4~6`
- 先只跑几十 step

### 建议补的单测

建议新增：

- `verl-datamind/tests/utils/test_verigraph_rl_on_cpu.py`
- `verl-datamind/tests/workers/reward_manager/test_verigraph_reward_manager_on_cpu.py`

至少覆盖下面这些点：

- 能正确解析 `<code_interpreter>`
- `submit_answer()` 后 rollout 会停止
- `final_claims` 能正确导出
- reward manager 能把 `process_reward` 和 `final_reward` 混合
- `preprocess_verigraph_rl.py` 能处理 `answers`
- `dir_id -> judge_prompt_type` 路由正确
- prompt 中没有额外 tool schema

---

## 第一版建议改哪些文件，哪些先别改

### 第一版建议改

- `verl-datamind/recipe/verigraph/preprocess_verigraph_rl.py`
- `verl-datamind/multi.sh`
- `verl-datamind/recipe/dapo/main_dapo.py`
- `verl-datamind/verl/utils/verigraph_rl.py`
- `verl-datamind/verl/workers/rollout/sglang_rollout/sglang_rollout.py`
- `verl-datamind/verl/workers/rollout/schemas.py`
- `verl-datamind/verl/trainer/ppo/ray_trainer.py`
- `verl-datamind/verl/workers/reward_manager/verigraph.py`
- `verl-datamind/verl/workers/reward_manager/__init__.py`
- `verl-datamind/tests/utils/test_verigraph_rl_on_cpu.py`
- `verl-datamind/tests/workers/reward_manager/test_verigraph_reward_manager_on_cpu.py`

### 第一版先不要改

- `verl-datamind/agent/async_interpreter.py`
- `verl-datamind/verl/workers/fsdp_workers.py`
- `verl-datamind/recipe/dapo/dapo_ray_trainer.py`

说明：

- `dapo_ray_trainer.py` 先不动逻辑，只要 rollout 返回 `trajectory` 就够了
- `fsdp_workers.py` 不需要为了 VeriGraph 专门改
- history compression 等 RL 稳定后再做

---

## 这版 NOTE 相比旧 NOTE 补充了哪些关键点

旧 NOTE 里还缺的、但实际实现必须明确的点有：

- `tool_config_path` 必须允许为 `null`，否则 prompt 会被塞 tool schema
- `reward_model.reward_kwargs` 现在不会自动透传到 DAPO reward manager，需要补
- `dapo_ray_trainer.py` 训练时要的是 `trajectory`，不是 `messages`
- 需要 per-sample `working_dir`
- reward 不能再围绕 `<answer>` 设计，而要围绕 `final_claims + submit_answer`
- 第一版不应该去扩展 DataMind 的 SQL interpreter，而应该直接复用现有 `CodeExecutor`

---

## 推荐实现策略

直接参考顶层已经写过的 VeriGraph 版本，但只迁移其中最必要的部分：

- 可直接参考：`verl/verl/utils/verigraph_rl.py`
- 可直接参考：`verl/verl/workers/reward_manager/verigraph.py`
- 可直接参考：`verl/recipe/verigraph/preprocess_verigraph_rl.py`
- 可直接参考：`verl/verl/trainer/ppo/ray_trainer.py` 里对 `tool_config_path=null` 的放宽方式

不建议第一版直接搬顶层那套 async callback / custom token rebuild / history compression；先用 `verl-datamind` 当前 sync multi-turn 路线把最小链路跑通。
