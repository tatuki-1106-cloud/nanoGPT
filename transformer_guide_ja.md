# nanoGPTで学ぶTransformer入門（日本語）

この資料は、**BERT/GPTなどの用語は知っている初学者**向けに、`nanoGPT`の実装を使ってTransformerの仕組みを理解するためのメモです。

## 1. このリポジトリで学ぶ目的

- Transformerの主要部品を、数式より先に**コード対応で理解**する
- 「学習時に何をしているか」「推論時に何をしているか」を把握する
- GPT系モデル（自己回帰型）の最低限の理解を得る

> 注意: このリポジトリはREADMEにある通り古めですが、実装が短く読みやすく、学習用として有用です。

---

## 2. Transformer（GPT）の全体像

GPTは次の流れで動きます。

1. 入力トークンIDを埋め込みベクトルに変換
2. 位置埋め込みを加算
3. Transformer Blockを複数層通す
   - LayerNorm
   - Causal Self-Attention
   - Residual接続
   - MLP
4. 最後に語彙次元へ射影して次トークン確率を出す
5. 正解トークン（1つ先）とのCross Entropyで学習

---

## 3. 用語を最短で理解する

- **Self-Attention**: 各位置の情報を、系列中の他位置を参照して更新する仕組み
- **Multi-Head Attention**: 複数の注意の見方を並列で使う
- **Causal Mask**: 未来トークンを見ない制約（GPTの核）
- **MLP**: 各位置ごとの特徴変換（非線形）
- **Residual接続**: `x + f(x)` で勾配を流しやすくする
- **LayerNorm**: 活性値を安定化して学習を助ける

---

## 4. nanoGPTコード対応表（最重要）

### 4.1 `model.py`

- `GPT.forward`
  - `wte`: token embedding
  - `wpe`: position embedding
  - `tok_emb + pos_emb` を入力としてBlockへ
- `Block.forward`
  - `x = x + self.attn(self.ln_1(x))`
  - `x = x + self.mlp(self.ln_2(x))`
  - つまり「Pre-LN + Residual」構成
- `CausalSelfAttention.forward`
  - `q, k, v` を作成
  - `scaled_dot_product_attention(..., is_causal=True)` で因果制約付き注意
- `lm_head`
  - 最終隠れ状態を語彙次元に変換し、次トークン分布を作る

### 4.2 `train.py`

- `get_batch(split)`
  - `x`: ある区間のトークン列
  - `y`: `x`を1トークン先にずらした正解列（次トークン予測）
- 学習ループ
  - `logits, loss = model(X, Y)`
  - `loss.backward()` で勾配計算し、optimizerで更新

### 4.3 `config/train_shakespeare_char.py`

- 小さめ設定（6層、6ヘッド、埋め込み384）で、学習挙動を掴みやすい
- `block_size=256` は「最大文脈長」

---

## 5. BERTとGPTの違い（混同しやすい点）

- **GPT**: 左から右へ次トークン予測（自己回帰、causal）
- **BERT**: マスク穴埋め中心（双方向文脈）
- このリポジトリはGPTのみを扱う

---

## 6. 最短ハンズオン

1. データ準備  
   `python data/shakespeare_char/prepare.py`
2. 学習  
   `python train.py config/train_shakespeare_char.py`
3. 生成  
   `python sample.py --out_dir=out-shakespeare-char`

学習直後の生成は崩れることもありますが、**「次トークン予測で言語らしさが生まれる」**という中核体験が得られます。

---

## 7. 読む順番（おすすめ）

1. `config/train_shakespeare_char.py`（モデル規模の感覚）
2. `train.py`（学習入出力）
3. `model.py`（Transformer本体）
4. `sample.py`（推論）

---

## 8. 初学者向けチェックポイント

- Attentionだけ覚えて満足しない（MLP・Residual・LayerNormも同じくらい重要）
- `block_size` はGPUメモリと精度のトレードオフ
- 低lossでも出力品質が常に高いとは限らない
- 小さい設定でまず一周し、理解後に規模を上げる

---

## 9. まとめ

`nanoGPT`は、Transformer/GPTの仕組みを**最短で「実装として」理解する教材**として優秀です。  
まずは小規模設定で学習→生成を体験し、`model.py`の対応箇所を追うことで、BERT/GPTの用語知識を実装理解に変換できます。
