# STEP 3: Transformer本体を読む（model.py）

## 到達目標

- `GPT.forward` のデータフローを説明できる
- `Block.forward` の Pre-LN + Residual 構造を説明できる
- Causal Self-Attentionで未来情報を見ない実装を説明できる

## 対象ファイル

- `/home/runner/work/nanoGPT/nanoGPT/model.py`

## 見るポイント

1. `wte`, `wpe`, `tok_emb + pos_emb`
2. `x = x + self.attn(self.ln_1(x))`
3. `x = x + self.mlp(self.ln_2(x))`
4. `scaled_dot_product_attention(..., is_causal=True)`
5. `lm_head` で語彙次元へ射影

## 手を動かす

- 上記5点が出る行を自分で見つけて、1行説明を付ける
- 「1トークン先予測」に必要な最小構成を図にする

## 確認質問

- Pre-LN構成のメリットは何ですか？
- なぜAttentionの後にResidualを足すのですか？
