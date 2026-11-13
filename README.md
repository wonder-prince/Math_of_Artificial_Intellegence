# Math_of_Artificial_Intellegence

人工智能数学课程实验：收益序列异常变化的数学建模与数值验证。

本目录提供数学课程研究报告的完整实验代码、实际计算产物与复现说明。主体严格为：最小二乘回归、离散分箱 Shannon 熵、蒙特卡罗/残差块 bootstrap。合成数据均明确标识，不作为真实市场证据，不含事件新闻判断。未读取或采用旧 nvda-event-study 的任何数据。

## 运行

需要 Python 3.10+；本次实际环境与依赖版本记在 results.json。

```bash
cd Math_of_Artificial_Intellegence
source .venv/bin/activate
python run_experiments.py
python -m unittest discover -s tests -v
shasum -a 256 -c manifest.sha256
```

重建环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_experiments.py
```

安装依赖需要网络；安装完成后，run_experiments.py 的实验计算和已归档真实数据处理不访问网络。不依赖 pandas。行情下载由独立的 download_data.py 完成。每次运行实验更新本项目的 results.json、validation.json、图表、运行日志和 manifest.sha256，不触碰旧项目。不要在已有需要保存的实验产物目录中直接改配置后覆盖运行，先复制本项目（可不复制 .venv）。

下载 Yahoo Finance 的 NVDA/SPY 2023—2025 年日线复权价格：

```bash
.venv/bin/python download_data.py
.venv/bin/python run_experiments.py
```

下载器使用 curl_cffi 的浏览器兼容 HTTPS 请求，并读取 macOS 系统代理或 HTTPS_PROXY/ALL_PROXY 环境配置，遵循代理绕过配置。Yahoo 对中国大陆直连返回停止服务页面时，需要使用能够访问 Yahoo 的现有网络出口；程序不硬编码代理地址。HTTP 401/403 会停止并记录原因，429 会记录限流及服务端 Retry-After，不连续重试同一请求。

失败后可以再次运行同一下载命令：旧失败记录和未完成的响应原件移入 data/download_history/ 下的 UTC 时间戳目录，保留原字节及历史证据。已有 verified_prices.json 时先通过 load_verified 校验并复用，不再次联网或覆盖。下载成功会生成原始响应、download_bundle.json、verified_prices.json 和 download_validation.json；之后运行实验即可纳入真实数据。

## 文件

- config.json：唯一默认实验配置，随机种子 20261113。
- math_core.py：lstsq 回归、AR(1)、非循环块索引、残差 bootstrap、p 值、Wilson 区间、离散熵、时间切分。
- experiments.py：三组数学实验；不涉及真实行情取数。
- download_data.py：独立联网下载、原件归档、共同交易日对齐与输入验证；读取现有系统代理。
- real_data.py：可选真实价格归档及下载证据校验，离线滚动回归。
- plots.py：6 幅浅底英文图，每幅输出 PNG 和 SVG。
- run_experiments.py：先跑测试，再执行全部实验，产出结果与清单。
- tests/test_math.py：独立 unittest；包含真值回归、正交/SSE、熵边界与反例、p 值边界、块内时间连续且不绕回、实际重新拟合对照、未来 y 扰动不改变训练/零分布、日期顺序与缺口、证据门控；另测AR(1)解析协方差/累计方差、固定设计预测误差权重及交叉项符号、退化窗口继续滚动、非有限/点质量bootstrap拒绝和取数失败记录。数学测试的I/O夹具全部在内存中。
- tests/test_download.py：模拟HTTP响应，核对系统代理、原件字节与哈希、权限/限流、失败留档重跑和成功复用；文件夹具仅写入自动清理的临时目录。
- results.json：完整配置、版本、原始逐重复 p 值和 MC 标准误、拒绝率与 Wilson 区间、共线性逐次系数及 MSE、熵合成路径、固定设计和示意零分布。
- validation.json / tests.log / run.log：验证结果与真实运行日志。
- manifest.sha256：源码、配置、产物和可选归档数据的 SHA256；不含清单自身、.venv、字节码缓存。

## 1. 最小二乘回归

模型 y=X beta+e，X 包含截距及两个相关正态解释变量。真 beta=[0.3,1.2,-0.7]。一律调用 numpy.linalg.lstsq，不构造正规方程逆矩阵。

共线性实验的 rho 为 0、0.9、0.999，每组独立重复 200 次，每次训练 200、独立测试 1000。记录每个系数的 RMSE、三个系数合并 RMSE、设计矩阵奇异值条件数、样本外 MSE 及其重复间标准误。独立测试噪声方差为 1，测试解释变量与训练具有相同总体相关结构；系数不稳定与同分布预测误差不同，不据此保证分布外预测。

记录最大 |X' residual| 和含截距的 SST=SSR+SSE 数值误差。真值测试在无噪声数据上检查 beta 恢复；bootstrap 批量多右端项 lstsq 与逐样本重新拟合独立对照。

## 2. 离散分箱 Shannon 熵

使用 H=-sum(p log2 p)，单位 bits，0 log 0 约定为 0。只讨论离散分箱熵，不把它和连续微分熵混用。

精确反例共用固定边界 [-11,-5,0,5,11]：

- A：-1/+1 各 0.5，H=1、均值 0、方差 1。
- B：-10/+10 概率 0.9/0.1，H=0.4689955935892812、均值 -8、方差 36。

每组另独立生成 10000 样本，记录样本熵及 ddof=1 标准差。有限样本结果允许偏离精确概率值。另用 N(0,1) 与 N(0.8,4) 两条明确标识的合成路径，窗口 50/200/1000，固定公共范围 [-12,12]，箱数 2/4/8/16/32，记录两组熵、熵差及标准差。改变箱数即改变离散化；不承诺任何数据的熵差方向。超出有限公共箱的样本会抛错，不静默丢弃尾部。

## 3. 蒙特卡罗/残差块 bootstrap

默认 ntrain=200、horizon=20、M=300、B=999、alpha=0.05。模型 AR(1) rho=0 和 0.6；平稳初始化，创新方差 1-rho²，从而残差边际方差为 1。两种 AR 设定的全部重复共用一组固定 X，设计相关性为 0.5。推断条件于这组 X，重复概率对随机误差与模拟取平均。

每个重复：

1. 用训练 200 点拟合，训练残差减去其均值。
2. 测试 20 点与训练严格分开，固定测试 X，统计量 T=abs(sum(test_y-test_X @ beta_hat))。
3. 分别用 block=1（iid）、5、10，随机抽取原训练残差池内连续块。起点均匀抽自 0..n-block，块不绕回，独立拼接到 n+h；最后一块可截短。
4. 每条伪路径以 X_train beta_hat + e_train* 生成伪训练 y，调用 lstsq 实际重新拟合 beta*，再用 X_test beta_hat + e_test* - X_test beta* 构造测试残差。一次多右端项求解完成 B 次重新拟合；这包括 beta 参数不确定性。
5. 比较 >= observed 的次数 exceed，用 (1+exceed)/(B+1)；拒绝条件 p<=0.05，最小 p=0.001。
6. 测试段四场景：null、均值 +0.8、标准差乘 2（方差乘 4）、中间单点 +8。训练不受冲击。四场景共用同条基础误差及各方法的同一零分布，方便配对，但检出率估计彼此相关。

每个 AR×方法×场景输出 M 次 p 值、统计量、exceed、MC 标准误，以及拒绝数、拒绝率、95% Wilson 区间。Wilson 衡量有限 M 的拒绝率误差，不衡量 bootstrap 模型偏差，不做多重比较校正。

同一AR设定内，各方法共用每次重复的基础误差，误报率差是配对比较。results.json 的 paired_null_rate_differences 保存 first-minus-second 的配对拒绝率差、仅第一/仅第二拒绝的次数，以及 std(I_first-I_second,ddof=1)/sqrt(M) 标准误。不能将两个边际 Wilson 区间按独立样本比例差解读。所有报告的检出率均未做 size-adjustment；误报率不等时，不应据此声称同等实际显著性水平下某方法功效更好。

解析参考不使用模拟代替真值：平稳边际方差为1的AR(1)有 Cov(e_i,e_j)=rho^|i-j|，累计噪声方差为 h+2*sum_{k=1}^{h-1}(h-k)*rho^k，等价闭式为 h*(1+rho)/(1-rho)-2*rho*(1-rho^h)/(1-rho)^2。rho=0.6,h=20 时约72.500274。固定X下，未取绝对值的累计预测误差为 1'e_future-c'e_train，其中 c=Xtrain(Xtrain'Xtrain)^-1 Xtest'1。实现通过对 Xtrain.T 做 lstsq 求最小范数权重c，不构造逆矩阵。整体权重w=[-c,ones(h)]，方差=w'Cov w；rho=0 时等于h+||c||²。训练与测试交叉协方差项带负号，绝不能把训练/测试独立近似当作AR解析真值。解析结果记录在 analytic_variance_reference 中。

单个 p 值的条件 MC 标准误使用 sqrt(B*p*(1-p))/(B+1) 的平滑插值近似；alpha=0.05、B=999 时约 0.0069，不包含残差池估计误差。边界值附近该误差估计较粗糙；+1 修正主要避免零 p 值，不能自动使近似 bootstrap 成为精确检验。

限制：训练期仅 200，block=5/10 丢失跨接缝相关与更长滞后。由于 200 是两种块长的倍数，训练/测试边界是一个拼块接缝，真实 AR 的边界依赖未保留。这个标准非循环 MBB 作为有限样本方法比较，不声称块法必然达到 5%。也不条件于最后训练残差做 AR 预测。T 针对累计偏移，方差冲击检出不是专门的方差检验；正负偏移可能抵消。

## 可选真实数据接口

主代理完成独立来源验证后，可写入 data/verified_prices.json。必须有 status、source、rows，且要求下载证据的 URL、时间戳、归档原件、SHA256。结构如下，尖括号内容由真实下载替换；不要直接将模板标识为已验证数据：

```json
{
  "status": "verified_download",
  "source": "<提供者、价格口径及验证说明>",
  "download_evidence": {
    "url": "<真实HTTPS下载URL>",
    "retrieved_at": "<ISO8601带时区下载时间>",
    "artifact": "<相对于data目录的原始下载文件名>",
    "sha256": "<归档原件的SHA256>"
  },
  "rows": [
    {"date": "<YYYY-MM-DD>", "NVDA": 100.0, "SPY": 500.0}
  ]
}
```

模板数值纯为字段类型示范，不是行情。真实计算至少需要 221 个同步价格日期（200 个训练收益+20 个测试收益+首日价格）。NVDA、SPY 必须同日对齐，价格为正且有限，建议同一供应商的可比较复权口径。提供者应记录原始文件解析方式及拆股/股息处理。程序只核验证据存在、哈希和基本结构，不能独立认证上游供应商、rows 转换或事件真实性。

真实滚动：先算相邻日对数收益；回归 NVDA 收益对截距与 SPY 收益。每窗仅过去 200 收益训练，测试未来 20，步长 20；训练窗可以重叠、测试窗互不重叠。测试 SPY 已实现收益作为固定设计，是事后条件推断。熵的内部边界只从训练残差范围确定，首尾无限外侧箱防止测试尾部丢失。

训练200与测试20的插件经验熵存在不同程度的有限样本偏差，且训练残差来自已拟合模型、测试残差来自样本外预测；仅凭熵差不能当作显著性证据。本项目不进行真实熵显著性检验，每个窗口均明确记录 entropy_comparable_for_significance=false。

逐窗口status：正常为computed；解释变量恒定导致设计秩不足，或训练残差为常量/数值上全零，或bootstrap零分布为点质量，记录skipped_degenerate和具体reason，随后继续后续窗口。常量训练残差的点质量离散熵定义为H=0，但训练分箱尺度不可识别，所以test_entropy=null、bootstrap=null，不将其与测试熵比较。数值常量阈值为64倍双精度机器精度乘以max(1,参考数据尺度,数组绝对最大值)。

非有限观测/模拟统计量、浮点算术异常、受控线性代数失败记录skipped_numerical和reason。只有上述明确数值异常被窗口处理捕获；普通ValueError、KeyError、RuntimeError等编程/配置错误继续上抛。mc_pvalue和bootstrap_null对非有限值或退化零分布显式抛出专用异常，不返回貌似有效的p值。所有窗口都被跳过时整体为no_valid_windows，并提供各status计数。

重复、乱序日期或超过 7 自然日的缺口拒绝；其他缺失工作日列出供主代理对照交易所日历，不自动填充或跨缺口插值。周末/假日之间的收益是相邻观测收益，真实数据尚不等同于完全规则等时距 AR 过程。缺数据记录 not_provided；证据/结构不合格记录 rejected 并跳过真实计算。合成部分仍完整运行。

若有 data/download_validation.json，离线读取并在real_data.data_availability和validation.json中保存取数记录。失败时保留not_provided及具体HTTP错误；下载成功且价格通过证据门控后才参与真实计算。早期的Yahoo HTTP 403失败记录保留在data/download_history中，不把失败尝试视为可用数据。下载记录和历史原件均纳入实验生成的最终哈希清单。

本次保留NVDA/SPY双标的接口。自有滞后收益会使设计包含过去随机误差，在AR误差下不能直接沿用当前严格外生固定设计推断；因此未把单标的滞后模型作为未经校准的替代实证。

## 复现口径

四条 SeedSequence 子流分别负责 bootstrap、熵、共线性、真实数据；有无真实数据不影响合成结果。保存固定设计、配置、数值库及平台版本；同环境同种子数值可复现，跨 BLAS/平台有末位差异。validation.json 另记录数值指纹；若存在同环境、同配置、同输入的上次结果，自动逐字段核对三组实验和真实数据接口的数值是否完全一致（首次或不可比时为 null）。日志时间、elapsed_seconds、SVG 元数据、以及相应清单哈希不承诺字节一致；不要据此误判模拟随机性。清单应在不修改文件后验证。
