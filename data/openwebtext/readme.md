## OpenWebText 解説

このディレクトリは、`train.py` に入力するための OpenWebText データを前処理して  
`train.bin` / `val.bin` を作るためのものです。

### 何が作られるか

`prepare.py` を実行すると以下を生成します。

- `train.bin`（約 17GB）
- `val.bin`（約 8.5MB）

目安のトークン数:

- train: 約 9B tokens（9,035,582,198）
- val: 約 4M tokens（4,434,897）

元データは約 8,013,769 文書です。

### `prepare.py` の処理内容

1. Hugging Face Datasets で `openwebtext` を読み込む  
2. `train` を `train/val` に分割（`test_size=0.0005`）  
3. GPT-2 BPE（`tiktoken`）でトークン化  
4. 各サンプルの末尾に `eot_token` を追加  
5. 全サンプルを連結し、`uint16` の memmap で `*.bin` に保存

### 学習時との関係

`train.py` は `data/openwebtext/train.bin` と `val.bin` を `np.memmap` で読み込み、  
ランダムな位置から `block_size` 長の系列を切り出してミニバッチを作ります。  
`x` が入力、`y` は 1 トークン右にずらした教師データです（次トークン予測）。

### 実行例

```bash
cd /tmp/workspace/tatuki-1106-cloud/nanoGPT
python data/openwebtext/prepare.py
```

### 参考

- GPT-2 paper: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset
