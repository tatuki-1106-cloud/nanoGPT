# STEP 0: 全体像と用語の定着

## 到達目標

- GPTの処理を「入力→Block群→出力→損失」の一本線で説明できる
- Self-Attention / Causal Mask / Residual / LayerNorm / MLP の役割を短く言える

## 学ぶ順番

1. `/home/runner/work/nanoGPT/nanoGPT/transformer_guide_ja.md` の「2. Transformer（GPT）の全体像」を読む
2. 同ファイルの「3. 用語を最短で理解する」を読む

## 手を動かす

- ノートに以下を1行ずつ書く
  - 入力トークンID → 埋め込み
  - 位置埋め込み加算
  - Blockを複数回
  - 語彙次元へ射影
  - 次トークン予測の損失計算

## 確認質問

- なぜGPTではCausal Maskが必要ですか？
- AttentionだけでなくMLPが必要なのはなぜですか？（直感で可）
