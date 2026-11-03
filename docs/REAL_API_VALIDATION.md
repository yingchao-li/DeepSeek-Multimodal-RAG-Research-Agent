# 真实 DeepSeek API 端到端验证报告

> 本报告来自真实 `https://api.deepseek.com` 请求，不含 Mock API。密钥未写入报告或结果文件。

## 运行摘要

- 实际 HTTP/API 请求数：**10**（硬上限 12，SDK 自动重试 0）
- 配置文本模型：`deepseek-chat`
- 配置视觉模型：`deepseek-flash`
- Prompt tokens：**4961**
- Completion tokens：**1183**
- Total tokens：**6144**
- 费用估算：未配置可靠单价，仅报告 Token

## Test 1：真实文本聊天 — PASS

- 预期：应回答巴黎
- 真实回答：法国的首都是巴黎。
- 端到端耗时：1501.14 ms
- API 调用：
  - #1 请求 `deepseek-chat`，服务端返回模型 `deepseek-flash`：prompt 90 / completion 5 / total 95 tokens，1500.92 ms，ok=True

## Test 2：真实图片理解 — PASS

- 预期：蓝色柱子最高
- 真实回答：中间那根蓝色柱子最高。
- 端到端耗时：1131.66 ms
- image_blocks：`1`
- API 调用：
  - #2 请求 `deepseek-flash`，服务端返回模型 `deepseek-flash`：prompt 308 / completion 53 / total 361 tokens，1130.83 ms，ok=True

## Test 3：真实多轮视觉聊天 — PASS

- 预期：绿色柱子最低
- 真实回答：最低的柱子是绿色的。
- 端到端耗时：1295.61 ms
- image_blocks：`1`
- duplicate_images：`false`
- API 调用：
  - #3 请求 `deepseek-flash`，服务端返回模型 `deepseek-flash`：prompt 326 / completion 40 / total 366 tokens，1293.74 ms，ok=True

## Test 4：真实 Multimodal PDF RAG — PASS

- 预期：蓝色；synthetic-aster.pdf 第2页 Figure 3
- 真实回答：Figure 3 中最高的柱子是**蓝色（BLUE）**。

图中三根柱子从左到右依次标为 RED、BLUE、GREEN，其中中间蓝色柱子的高度明显最高 [S2]。该图被描述为一张含三根柱子的颜色性能图 [S1]。
- 端到端耗时：2090.24 ms
- image_count：`1`
- retrieved：`["synthetic-aster.pdf, p. 2, 2 Visual Results", "synthetic-aster.pdf, p. 2, Figure 3, 2 Visual Results", "synthetic-aster.pdf, p. 2, 3 Discussion", "synthetic-aster.pdf, p. 3, 4 Ablation", "synthetic-aster.pdf, p. 2, 2 Visual Results", "synthetic-aster.pdf, p. 2, 3 Discussion"]`
- visual_evidence：`["synthetic-aster.pdf, p. 2, Figure 3, 2 Visual Results"]`
- API 调用：
  - #4 请求 `deepseek-flash`，服务端返回模型 `deepseek-flash`：prompt 793 / completion 112 / total 905 tokens，2073.91 ms，ok=True

## Test 5：真实跨文档问答 — PASS

- 预期：Boreal 83.0，Aster 81.0，Boreal 高 2.0 AP
- 真实回答：**结论：Boreal 更高。**

- **Boreal**：small-object AP 为 **83.0** [S1][S2][S6]
- **Aster**：small-object AP 为 **81.0** [S3][S4][S5][S8]
- **差值**：83.0 − 81.0 = **2.0**

**可比性说明**：资料明确说明两者使用相同的日光数据集和相同的测试划分（“the same daylight dataset and test split”），因此这一比较是在受控条件下进行的 [S6][S8]。

**补充**：这些文件自述为“Synthetic integration fixture - not a real research result”（合成集成测试样例，并非真实研究结果）[S6][S8]，引用时需注意其性质。
- 端到端耗时：1629.21 ms
- retrieved_sources：`["synthetic-aster.pdf", "synthetic-boreal.pdf"]`
- API 调用：
  - #5 请求 `deepseek-chat`，服务端返回模型 `deepseek-flash`：prompt 737 / completion 171 / total 908 tokens，1608.05 ms，ok=True
- 证据：`[{"citation": "S1", "chunk_id": "16a38fed5fcc0dd54acd", "source": "synthetic-boreal.pdf", "page": 1, "section": "1 Boreal Method", "kind": "paragraph"}, {"citation": "S2", "chunk_id": "4294a5ec627e5e38ab84", "source": "synthetic-boreal.pdf", "page": 1, "section": "1 Boreal Method", "label": "Table 1", "kind": "table"}, {"citation": "S3", "chunk_id": "664e694d2aa722ac2f65", "source": "synthetic-aster.pdf", "page": 1, "section": "1 Introduction", "kind": "paragraph"}, {"citation": "S4", "chunk_id": "8c28c34d5c84b02dabc4", "source": "synthetic-aster.pdf", "page": 1, "section": "1 Introduction", "label": "Table 2", "kind": "table"}, {"citation": "S5", "chunk_id": "0311ac6d91078fd5ebb5", "source": "synthetic-aster.pdf", "page": 3, "section": "4 Ablation", "label": "Table 4", "kind": "table"}, {"citation": "S6", "chunk_id": "449e8b677b32b988af7f", "source": "synthetic-boreal.pdf", "page": 1, "section": "1 Boreal Method", "kind": "paragraph"}, {"citation": "S7", "chunk_id": "0d46845acb6874689e52", "source": "synthetic-boreal.pdf", "page": 1, "section": "1 Boreal Method", "kind": "heading"}, {"citation": "S8", "chunk_id": "5e652a1469ee246140b5", "source": "synthetic-aster.pdf", "page": 1, "section": "1 Introduction", "kind": "paragraph"}]`

## Test 6：真实 Agentic RAG — PASS

- 预期：Boreal 83.0 vs Aster 81.0；相差 2.0；夜间场景与 CPU 延迟未测
- 真实回答：**结论：Boreal 更高。**

- **Boreal：83.0 small-object AP** [S5][S6][S7]
- **Aster：81.0 small-object AP** [S1][S2][S5][S6]
- **差距：2.0 AP**（83.0 − 81.0）

**可比性说明：** 两份资料都说明，Aster 与 Boreal 使用的是同一套白天数据集和同一评测划分，因此这一比较属于受控对比 [S3][S7]。

**资料明确说明未进行的评测：**
- **夜间场景（night-time scenes）未测** [S4]
- **CPU 推理延迟（CPU inference latency）未测** [S4]

**补充（消融结果，非跨方法比较）：** Aster 去掉特征金字塔后，small-object AP 从 81.0 降到 74.0，说明特征金字塔对该指标有贡献 [S4]。

**注意：** 所给材料中两份文档均标注为“Synthetic integration fixture - not a real research result”（合成整合样例，非真实研究结果）[S3][S7]，引用这些数值时应保留这一限定。
- 端到端耗时：7568.83 ms
- retrieved_sources：`["synthetic-aster.pdf", "synthetic-boreal.pdf"]`
- API 调用：
  - #6 请求 `deepseek-chat`，服务端返回模型 `deepseek-flash`：prompt 124 / completion 27 / total 151 tokens，1053.87 ms，ok=True
  - #7 请求 `deepseek-chat`，服务端返回模型 `deepseek-flash`：prompt 512 / completion 145 / total 657 tokens，1158.82 ms，ok=True
  - #8 请求 `deepseek-chat`，服务端返回模型 `deepseek-flash`：prompt 473 / completion 59 / total 532 tokens，1093.6 ms，ok=True
  - #9 请求 `deepseek-chat`，服务端返回模型 `deepseek-flash`：prompt 721 / completion 257 / total 978 tokens，2258.1 ms，ok=True
  - #10 请求 `deepseek-chat`，服务端返回模型 `deepseek-flash`：prompt 877 / completion 314 / total 1191 tokens，1982.54 ms，ok=True
- Agent path：`agentic_rag`；tool calls：2；API calls：5
- Agent evidence：`[{"citation": "S1", "chunk_id": "664e694d2aa722ac2f65", "source": "synthetic-aster.pdf", "page": 1, "section": "1 Introduction", "kind": "paragraph"}, {"citation": "S2", "chunk_id": "8c28c34d5c84b02dabc4", "source": "synthetic-aster.pdf", "page": 1, "section": "1 Introduction", "label": "Table 2", "kind": "table"}, {"citation": "S3", "chunk_id": "5e652a1469ee246140b5", "source": "synthetic-aster.pdf", "page": 1, "section": "1 Introduction", "kind": "paragraph"}, {"citation": "S4", "chunk_id": "9c85441ba97498cd916b", "source": "synthetic-aster.pdf", "page": 3, "section": "4 Ablation", "kind": "paragraph"}, {"citation": "S5", "chunk_id": "16a38fed5fcc0dd54acd", "source": "synthetic-boreal.pdf", "page": 1, "section": "1 Boreal Method", "kind": "paragraph"}, {"citation": "S6", "chunk_id": "4294a5ec627e5e38ab84", "source": "synthetic-boreal.pdf", "page": 1, "section": "1 Boreal Method", "label": "Table 1", "kind": "table"}, {"citation": "S7", "chunk_id": "449e8b677b32b988af7f", "source": "synthetic-boreal.pdf", "page": 1, "section": "1 Boreal Method", "kind": "paragraph"}]`
- Agent warnings：`[]`

## 调用链与结论

- 文本与视觉请求均经过项目的 `ManagedClient`、`src.api.request` 与 `src.llm`。
- PDF 测试经过上传保存、PyMuPDF 结构解析、MiniLM/FAISS 建索引、BM25 + FAISS + RRF、视觉证据选择和 RAG 生成；没有手工绕过检索指定图片。
- Agent 测试使用现有 `run_agentic_rag`，关闭外部 Web，限制为 2 轮、6 个工具调用、6 个 API 调用。
- 多轮视觉请求只携带 1 个历史图片块；prompt tokens 从 308 增至 326，未发现重复图片或异常膨胀。
- 所有请求均按配置发出：文本调用请求 `deepseek-chat`，视觉调用请求 `deepseek-flash`；本次服务端响应对象对两类请求都报告模型为 `deepseek-flash`。
- JSON 证据保存在 `reports/phase3/real-api-results.json`；其中不含密钥和 base64 图片。

## Bug 修复与剩余问题

- 修复：API 记录现在保存服务端返回的模型名与 total_tokens，便于真实集成审计。
- 限制：DeepSeek 返回的模型标识未区分文本请求与视觉请求；报告同时保留请求模型和服务端模型，未擅自推断其内部路由。
- 若测试失败，错误分类及未解决限制记录在对应测试项；失败结果未由 Mock 数据替代。

## 复现命令

```bash
cd /Users/ingchaolee/PycharmProjects/PythonProject_ai
.venv/bin/python scripts/real_api_validation.py
```
