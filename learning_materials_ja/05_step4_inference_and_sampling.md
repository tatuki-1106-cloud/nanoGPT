# STEP 4: 推論と生成を読む（sample.py）

## 到達目標

- 学習済み重みから、次トークンを繰り返し生成する流れを説明できる
- 学習（train.py）と推論（sample.py）の違いを整理できる

## 対象ファイル

- `/home/runner/work/nanoGPT/nanoGPT/sample.py`

## 見るポイント

1. モデル読み込み（`out_dir` / 事前学習モデル）
2. 入力プロンプト処理
3. 生成ループ（次トークンを順次追加）
4. 生成長やサンプル数などの制御引数

## 手を動かす

1. 生成を実行
   - `python sample.py --out_dir=out-shakespeare-char`
2. 引数を変えて比較
   - `--num_samples`
   - `--max_new_tokens`

## 確認質問

- 学習時と違い、推論時にlossが不要なのはなぜですか？
- 生成が崩れるとき、まずどの設定や状態を疑いますか？
