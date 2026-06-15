# （二）拆解黑盒：从d2l-zh到nanoGPT

> 两个月时间，从PyTorch零基础到手写Transformer

---

上一篇文章，你用Scikit-learn跑通了第一条ML Pipeline，理解了"读数据→训练→评估"这个基本循环。但你可能隐隐觉得不对劲——调 `fit()` 和 `predict()` 的感觉，像是在用一台你不懂原理的机器。

你真正想搞懂的问题可能是：ChatGPT 里面到底在发生什么？

这篇文章就是来回答这个问题的。它不会让你在一天之内搞懂一切——但会给你一条清晰的两阶段路径，走完它，你能手写一个简化版的GPT。

---

## 支柱一：d2l-zh 是你的教科书

[《动手学深度学习》（d2l-zh）](https://github.com/d2l-ai/d2l-zh) 是这个阶段唯一需要从头到尾跟的教材。它的特点：每一章 = 理论 + PyTorch 代码 + 可运行的 Notebook。不需要切换浏览器去看另一个教程，所有的都在一个 Jupyter 文件里。

**优先级最高的章节**（按顺序）：

| 章节 | 为什么重要 | 预计时间 |
|------|-----------|---------|
| 第3章：线性神经网络 | 理解梯度下降的最简形式 | 2天 |
| 第4章：多层感知机 | 理解激活函数、反向传播 | 3天 |
| 第5章：深度学习计算 | 理解PyTorch的Module、Parameter机制 | 1天 |
| 第6章：卷积神经网络 | 理解参数共享、局部连接——后面会用到 | 3天 |
| 第9章：循环神经网络 | 理解序列建模——Transformer的前身 | 3天 |
| 第10章：注意力机制 | **核心中的核心** | 5天 |

**怎么读**：每一个Notebook，把代码从头敲一遍。不是复制粘贴，是逐行打出来。遇到看不懂的公式，去B站搜李沐的同名视频，他会在黑板上把公式推一遍。

这里有个重要的判断标准：如果你在第10章（注意力机制）看到Q、K、V这三个字母时不觉得陌生，说明前面的部分吃透了。如果觉得陌生，回去复习第3章——Attention里的矩阵乘法就是线性回归里那个 `y = Xw` 的高维版本。

---

## 支柱二：nanoGPT 是你的里程碑

d2l-zh 给你理论，[nanoGPT](https://github.com/karpathy/nanoGPT) 给你终点。它是 Andrej Karpathy 写的极简GPT实现——去掉了所有框架抽象，只保留Transformer Decoder的核心逻辑。

**你需要的东西**：

- 一张消费级显卡。RTX 3060 (12GB) 或更高。没有显卡也能跑，但训练会慢 10-20 倍。
- 一个完整的下午。第一次跑通大概需要 3-5 小时，注意是"第一次"——读代码、调参数、看 loss 下降。

**做什么**：

把 nanoGPT 克隆到本地，跟着下面的步骤一步步跑：

**第一步：克隆仓库**

```
git clone https://github.com/karpathy/nanoGPT.git
```

**第二步：进入目录并安装依赖**

```
cd nanoGPT
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

**第三步：数据预处理**

```
python data/shakespeare_char/prepare.py
```

这个脚本会自动下载莎士比亚数据，把它切成训练集和验证集，生成 `train.bin`、`val.bin` 和 `meta.pkl`。

**第四步：跑一次完整训练**

```
python train.py config/train_shakespeare_char.py
```

> Windows 用户注意：用文本编辑器打开 `config/train_shakespeare_char.py`，在文件末尾添加一行 `compile = False`。

训练过程中终端会实时输出 loss 值。以下是某次实际训练的输出：

```
step 2000: train loss 1.0611, val loss 1.4695
step 4250: train loss 0.6824, val loss 1.6472
```

当 loss 降到 2.0 以下时，生成的文本开始出现基本的句子结构。到 loss 1.0 左右时，输出类似这样：

```
To be or not my words.

BRAKENBURY:
'Tis fair of you, 'tis it not worthy:
I have done that I was the writing Pedla,
And from the rather now you threw such pity
...
```

虽然语法不完全正确（10M 参数的字符级模型能力有限），但已经有了句子结构、人名和古英语风格——这就是"模型开始学会语言规律"的实证。

**第五步：用训练好的模型生成文本**

在 `nanoGPT` 目录下创建 `generate.py`，写入以下代码：

```
import torch
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load('out-shakespeare-char/ckpt.pt', map_location=device)

from model import GPT

class Cfg: pass
cfg = Cfg()
for k, v in ckpt['model_args'].items():
    setattr(cfg, k, v)

model = GPT(cfg)
model.load_state_dict(ckpt['model'])
model.eval()
model.to(device)

with open('data/shakespeare_char/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
itos = meta['itos']
stoi = meta['stoi']

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join([itos[i] for i in ids])

context = torch.tensor([encode('To be or not')]).to(device)
with torch.no_grad():
    out = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)

print(decode(out[0].tolist()))
```

然后运行 `python generate.py`，就能看到你自己的模型生成的莎士比亚风格文本了。

**训练时间参考**（莎士比亚数据集）：

| 显卡 | 单次训练耗时 |
|------|-------------|
| RTX 4090 | 20-30分钟 |
| RTX 3060 | 60-90分钟 |
| M2 MacBook Air | 约4-6小时（MPS后端，较慢） |

**理解代码的路径**：

1. 从头读到尾读一遍 `model.py`——你会看到一个 `GPT` class，里面包含 `self.transformer` 和 `self.lm_head`。
2. 看 `GPT.forward()`——输入 token indices，经过 embedding → transformer blocks → layer norm → linear projection，输出 logits。
3. 找到 `Block` class——它包含一个 `SelfAttention` + 一个 `MLP`。这就是Transformer论文里的核心结构。
4. 看 `CausalSelfAttention`——这就是你从d2l-zh第10章学到的Scaled Dot-Product Attention，只是加了mask（让token看不到后面的token）。

如果你能独立走完这四步，你就理解了GPT为什么是"预测下一个词"——它做的事情就是从输入序列出发，每一步只看前面的词，预测下一个最可能出现的词。然后把这个词拼到输入里，再预测下一个。循环往复。

---

## 辅助材料

两个项目在手边备着，不用专门学，遇到困惑时翻：

**[The Annotated Transformer](https://github.com/harvardnlp/annotated-transformer)**：带着论文（"Attention Is All You Need"）一起读。论文里出现一个公式，代码里就有对应的实现。不用全看懂——重点盯Multi-Head Attention和Positional Encoding两部分。

**[Transformer Explainer](https://github.com/poloclub/transformer-explainer)**：当你读nanoGPT的 `CausalSelfAttention` 时，如果脑子里矩阵维度对不上，打开这个网页工具看数据流动的可视化。它把抽象的矩阵乘法变成了动画。

---

## 两个评判标准

走完这个阶段，你应该能用大白话回答下面两个问题：

1. **Transformer 和 RNN 的本质区别是什么？**（答案是：RNN 逐步处理，Transformer 一步到位。这不是优劣问题，是计算方式根本不同。）
2. **为什么 GPT 能生成看起来通顺的文本？**（答案是：它学会了"在给定上下文的前提下，哪个词最可能出现"。不是"理解"了意思，是拟合了一个极其高维的条件概率分布。）

如果你能用自己的话把这两个问题讲清楚，你对Transformer的理解已经超过95%的"用过ChatGPT"的人。

---

## 参考项目

[d2l-zh](https://github.com/d2l-ai/d2l-zh) 第3-10章：主线教材，覆盖PyTorch基础到注意力机制 
[nanoGPT](https://github.com/karpathy/nanoGPT)：里程碑项目：读懂并跑通一个迷你GPT 
[Annotated Transformer](https://github.com/harvardnlp/annotated-transformer)：论文代码对照参考——遇到困惑时翻 
[Transformer Explainer](https://github.com/poloclub/transformer-explainer)：可视化辅助——矩阵维度对不上时看 

上面四个项目，前两个是主线要花时间跟的，后两个是参考工具随时翻的。不需要再多。

---

## 到了这里，你会想"所以怎么用这些模型"

当你跑通了nanoGPT，看着自己训练的模型从乱码逐渐生成有意义的句子，恭喜你——你已经是一个理解了Transformer底层的人。你会开始想一个更实际的问题：nanoGPT能写莎士比亚，但它不知道我今天的工作文档在哪里。它怎么回答关于我自己的东西的问题？这就是下一篇文章的内容。

---

> 本系列 Notebook 及配套文章：[github.com/ASPIRINH/hands-on-llm](https://github.com/ASPIRINH/hands-on-llm)

---

<details>
<summary><b>常见问题</b></summary>

**Q: 没有 GPU，nanoGPT 能跑吗？**
A: 能，但很慢。把 `config/train_shakespeare_char.py` 里的 `device` 改为 `'cpu'`，训练时间从 1 小时变成 10-20 小时。建议先下载预训练权重（GitHub Releases 里有）直接做推理，理解代码逻辑，有余力再从头训练。

**Q: d2l-zh 的 Notebook 里 `import torch` 报错？**
A: 没装 PyTorch。去 pytorch.org 选你的配置，复制生成的 `pip install` 命令。注意：不要装 CPU-only 版本如果你有 NVIDIA 显卡——装 CUDA 版本速度差 10 倍以上。

**Q: 训练 loss 不降？**
A: 最常见的原因是学习率太大或 batch size 太小。检查 `config/train_shakespeare_char.py` 里的 `learning_rate` 是否在 `1e-4` 到 `5e-4` 之间。如果改了其他参数，先恢复默认配置。

**Q: 看 Annotated Transformer 时被维度推导卡住了？**
A: 把 paper 里的 `d_model=512` 代入代码，在每一行后面加 `print(x.shape)`。你会看到每一步的 tensor 形状变化——这是理解 Transformer 最高效的方式。

</details>
