# STEP 1: 設定ファイルからモデル規模を掴む

## 到達目標

- `block_size`, `n_layer`, `n_head`, `n_embd` が何を制御するか説明できる
- 小さく試して理解し、後で大きくする方針を持てる

## 対象ファイル

- `/home/runner/work/nanoGPT/nanoGPT/config/train_shakespeare_char.py`

## 見るポイント

1. モデルサイズ関連パラメータ（層数・ヘッド数・埋め込み次元）
2. 学習反復や評価頻度などの実行設定
3. `out_dir` など成果物の出力先

## 手を動かす

1. 次のコマンドでデータ準備
   - `python data/shakespeare_char/prepare.py`
2. 設定ファイルを読み、主要パラメータを表にまとめる
   - 例: `block_size=256` は「最大文脈長」

## 確認質問

- `block_size` を増やすと、何と何のトレードオフが発生しますか？
- `n_layer` と `n_embd` を上げると、一般に何が増えますか？
