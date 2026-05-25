# STEP 2: 学習ループを読む（train.py）

## 到達目標

- `x` と `y` が1トークンずれた教師データになっている理由を説明できる
- `logits, loss = model(X, Y)` から更新までの流れを追える

## 対象ファイル

- `/home/runner/work/nanoGPT/nanoGPT/train.py`

## 見るポイント

1. `get_batch(split)` の `x` / `y` 生成
2. Forwardでlossを計算している箇所
3. `loss.backward()` と optimizer.step() の更新処理

## 手を動かす

1. 学習を実行
   - `python train.py config/train_shakespeare_char.py`
2. ログを見て、次を記録
   - 何iterごとに評価しているか
   - train/val lossがどう推移するか

## 確認質問

- なぜ `y` は `x` を1つ右にずらした列ですか？
- 学習が進んでも生成品質が不安定なことがあるのはなぜですか？
