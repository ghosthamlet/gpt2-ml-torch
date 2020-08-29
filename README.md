
## Pytorch model for https://github.com/imcaspar/gpt2-ml

gpt2-ml的pytorch版本预训练模型下载及转换和运用。

目前转换好的只有15G语料和BertTokenizer(21128 tokens)的预训练模型，可自行用以下方法转换gpt2-ml的30G语料版本，需修改目录配置。

#### 运行环境：

在Ubuntu 16.10，Python 3.6.10，Transformers 2.11.0，Pytorch 1.4.0和Tensorflow 1.14.0环境测试通过，其他环境未测试，如果有需要，大家自己稍作修改modeling_gpt2.py应该就能运行。

#### Pytorch模型下载：
模型太大已用7z格式压缩，并分成多个文件，下载前先留意一下是否支持或已安装7z解压软件，
模型下载百度云盘链接: https://pan.baidu.com/s/1Xe3MGA-ELiT1rsrx_HGaHg 提取码: k6b3。

#### 安装：

`
git clone https://github.com/ghosthamlet/gpt2-ml-torch.git
`

如果之前没有安装Transformers 2.11.0，Pytorch 1.4.0和Tensorflow 1.14.0:

`
pip install -r requirements.txt
`

百度云下载的pytorch模型放入models/mega-bert-tok


#### 如果需要自己转换Pytorch模型，先下载https://github.com/imcaspar/gpt2-ml 的tensorflow模型，放入 models/mega-bert-tok-tf 目录， 确保models/mega-bert-tok-tf 目录包含文件model.ckpt-100000.index, model.ckpt-100000.meta, model.ckpt-100000.data-00000-of-00001，及mega.json，models/mega-bert-tok 目录包含vocab.txt，参考本代码库models/ 目录，运行:

`
python convert.py
`   
生成的pytorch模型在models/mega-bert-tok 目录


#### 生成文字:   
`
python generate.py --prompt 宇宙的意义是 --max_len 300 --n_seq 3
`

> 0. 宇宙的意义是 无 序 。 因 而 一 切 不 确 定 的 存 在 都 是 无 限 的 。 所 以 我 们 才 有 无 所 不 知 的 物 理 物 理 。 这 便 是 宇 宙 的 意 义 。 没 有 什 么 是 绝 对 的 或 者 完 全 的 。 也 没 有 任 何 >东 西 是 绝 对 的 。 我 们 都 是 普 通 人 ， 我 们 也 都 存 在 其 他 的 知 识 ， 我 们 也 存 在 宇 宙 的 全 部 面 向 ， 所 有 东 西 都 可 以 被 称 为 是 宇 宙 的 一 部 分 ， 只 是 在 以 上 内 容 的 框 架 之 下 >， 你 很 难 找 到 某 一 种 或 某 几 种 以 上 的 定 义 。 也 许 ， 你 会 认 为 这 些 在 我 们 的 意 识 之 中 都 存 在 ， 但 这 才 是 我 们 的 生 活 的 意 义 。 而 每 个 人 的 意 识 之 中 ， 都 是 有 自 己 的 意 >识 ， 也 就 是 在 相 对 稳 定 的 意 识 之 中 。 而 生 活 在 相 对 稳 定 的 意 识 之 中 的 人 ， 就 像 一 个 生 活 在 恒 久 不 变 的 行 星 里 的 人 ， 是 无 法 真 的 发 觉 任 何 东 西 的 。 那 么 我 们 还 有 什 >么 呢 ？
> 1. 宇宙的意义是 什 么 ？ 天 文 学 的 意 义 在 哪 里 ？ 首 先 宇 宙 的 本 质 是 什 么 ？ 宇 宙 不 是 某 种 科 学 理 论 的 预 言 ， 然 而 它 确 实 是 一 种 认 知 的 认 知 。 如 果 要 说 宇 宙 是 什 么 ， 那 >宇 宙 就 是 一 堵 无 法 逾 越 的 墙 ， 一 个 可 能 性 ， 一 个 认 知 的 概 念 。 它 就 是 那 个 能 够 被 你 认 知 和 预 测 的 对 象 。 宇 宙 之 外 的 一 切 如 同 空 旷 的 存 在 者 ， 任 何 认 识 都 只 能 是 你 >所 说 的 确 定 ， 而 无 法 永 恒 。 在 这 个 可 能 性 下 ， 人 的 认 知 不 再 是 确 实 ， 而 是 可 知 。 如 果 说 宇 宙 是 一 个 概 念 ， 它 确 实 存 在 于 这 个 世 界 ， 它 存 在 的 意 义 之 一 也 是 这 个 世 >界 不 再 仅 仅 是 一 个 概 念 ， 它 存 在 的 意 义 也 是 我 们 生 活 的 概 念 ， 比 如 它 可 以 是 我 们 认 知 世 界 的 认 知 方 式 。 我 认 为 它 与 我 们 的 生 活 就 密 不 可 分 ， 有 着 它 独 有 的 个 性。
> 2. 宇宙的意义是 在 一 个 特 殊 的 时 间 ， 通 过 无 数 的 小 行 星 在 不 同 的 位 置 相 遇 ， 让 人 类 对 这 个 世 界 的 理 解 由 地 球 变 成 了 太 阳 系 。 可 以 说 是 一 场 史 无 前 例 的 太 空 旅 行 ， >大 家 认 识 的 好 多 事 物 ， 其 实 都 不 过 是 太 阳 系 的 小 分 子 结 构 ， 只 是 因 为 太 阳 系 还 处 在 星 际 时 代 的 前 期 ， 就 像 一 团 无 数 星 际 尘 埃 ， 一 瞬 间 被 发 现 ， 成 为 了 宇 宙 的 大 创 >世 神 。 很 多 人 看 过 地 心 说 ， 也 知 道 银 河 系 与 其 他 行 星 相 比 ， 不 过 是 个 几 亿 岁 的 小 神 经 ， 但 是 ， 这 个 不 是 宇 宙 的 全 体 ， 大 爆 炸 之 后 ， 如 果 没 有 外 围 的 引 力 ， 银 河 系 >就 没 有 什 么 力 量 了 。 地 心 说 主 张 的 就 是 这 个 观 念 ， 地 心 说 是 一 个 最 强 的 、 最 正 统 的 理 论 体 系 ， 被 称 作 上 帝 的 一 切 智 慧 ， 它 从 宇 宙 起 源 出 发 ， 到 发 现 一 个 行 星 为 止 >， 没 有 说 过 以 后 就 可 以 毁 灭 任 何 一 个 行 星 了 。

    
`
python generate.py --prompt "【上联】悠悠柳岸落红霞 【下联】" --max_len 100 --n_seq 3
`

> 0. 【上联】悠悠柳岸落红霞 【下联】 思 无 邪 松 烟 绿 水 悠 悠 ， 柳 岸 落 红 。 红 雨 过 后 ， 湖 光 山 色 依 然 碧 蓝 ， 美 丽 的 湖 光 山 色 中 ， 三 月 桃 花 漫 山 遍 野 开 放 ， 湖 上 人 们 静 静 诉 说 着 传 说 中 的 樱 花 雨 ， 湖 面 泛 起 水 ， 柳 堤 上 散 落 着 樱 花 树 。
> 1. 【上联】悠悠柳岸落红霞 【下联】 万 里 莺 莺 啼 尽 红 尘 ， 山 河 倒 影 入 斜 阳 。 谁 能 凭 一 身 诗 意 把 灵 魂 纳 进 诗 篇 ， 谁 就 成 功 的 成 为 了 一 个 传 奇 ！ 古 人 不 乏 美 女 ， 而 她 们 更 能 让 你 看 懂 人 生 ， 看 透 世 道 ， 体 会 人 情 。
> 2. 【上联】悠悠柳岸落红霞 【下联】 苍 山 淡 墨 写 婵 娟 【 横 批 】 悠 悠 的 柳 岸 ， 一 弯 秋 水 ， 淡 淡 的 绿 红 。 出 自 《 三 朝 北 盟 会 编 · 章 文 正 公 奇 幻 世 界 》 【 开 本 】 长 安 书 帖 【 序 >言 】 不 仅 我 们 生 活 在 一 个 多 元 化 的 时 代 ， 许 多 人 也 已 是 多 元 的。

`
python generate.py --prompt "刘梅和李丽是好朋友，她们正在讨论吃饭的问题。[刘梅] 中午去哪里吃？ [李丽] 吃麦当劳怎么样？ [刘梅]" --max_len 200 --n_seq 1
`
> 0. 刘梅和李丽是好朋友，她们正在讨论吃饭的问题。[刘梅] 中午去哪里吃？ [李丽] 吃麦当劳怎么样？ [刘梅]  汉 堡 包 比 较 合 口 味 ， 还 是 炸 鸡 腿 堡 更 合 口 味 ？ [ 刘 梅 ] 在 哪 里 吃 ？ [ 李 丽 ] 在 中 国 >菜 网 订 餐 ， 可 以 享 受 汉 堡 和 炸 鸡 腿 堡 的 优 惠 ， 汉 堡 包 的 价 格 比 炸 鸡 腿 堡 要 低 ， 如 果 你 的 汉 堡 夹 着 炸 鸡 肉 的 话 ， 会 更 合 口 味 。 [ 刘 梅 ] 那 李 丽 是 做 什 么 的 ？ [ 李 丽 ] 我
>  们 是 一 个 餐 饮 网 站 、 厨 师 培 训 、 淘 宝 运 营 团 队 。 ( 刘 梅 ) 如 果 您 是 想 了 解 食 品 行 业 的 信 息 ， 请 关 注。


#### 调用：

from config import MODEL_PATH

from generate import generate

print(generate('中国人', MODEL_PATH, 1, 100))

> [{'generated_text': '中国人 对 所 有 文 化 都 喜 欢 分 个 好 恶 高 下 ： 如 果 中 国 人 很 喜 欢 韩 流 ， 那 韩 娱 的 受 众 肯 定 不 会 特 别 喜 欢 这 种 类 型 的 韩 国 综 艺 。 中 国 人 很 喜 欢 美 剧 ， 美 剧 和 韩 剧 的 受 众 都 不 会 特 别 喜 欢 这 种 类 型 的 美 剧 。 中 国 人 看 日 本 电 影 对 美 妆 都 不 懂 ，'}]

