# train.py 読解ガイド（日本語）

この資料は `train.py` を「コードの塊ごと」に追って、**何をしているか**を理解するための学習メモです。

---

## 1. ファイル先頭: 目的と実行モード

- 先頭のdocstringで、単一GPU実行とDDP実行の起動例を示しています。
- `import` 群で、
  - 標準ライブラリ（`os`, `time`, `math`, `pickle`）
  - 数値計算（`numpy`）
  - 学習基盤（`torch`, `DDP`, 分散初期化/終了）
  - モデル定義（`GPTConfig`, `GPT`）
  を読み込みます。

---

## 2. デフォルト設定ブロック

`out_dir` から `compile` までの変数定義は、学習設定の初期値です。

- I/O・評価・保存方針（`eval_interval`, `always_save_checkpoint` など）
- データ設定（`dataset`, `batch_size`, `block_size`）
- モデル設定（`n_layer`, `n_head`, `n_embd`, `dropout`）
- Optimizer設定（`learning_rate`, `weight_decay`, `beta1`, `beta2`）
- 学習率スケジュール設定（`warmup_iters`, `lr_decay_iters`, `min_lr`）
- 実行環境設定（`device`, `dtype`, `compile`）

その後、

- `config_keys = ...` で設定候補を列挙
- `exec(open('configurator.py').read())` でCLI/設定ファイル上書き
- `config = { ... }` で最終設定を辞書化

します。

---

## 3. 実行初期化（DDP対応を含む）

`ddp = int(os.environ.get('RANK', -1)) != -1` で分散実行か判定します。

### DDP時

- `init_process_group` で分散初期化
- rank情報（`RANK`, `LOCAL_RANK`, `WORLD_SIZE`）取得
- GPUデバイス固定
- `master_process`（ログ/保存担当）をrank0に限定
- `gradient_accumulation_steps` をworld sizeで割って整合性を取る

### 単一プロセス時

- `master_process=True`, `ddp_world_size=1`

最後に `tokens_per_iter` を計算して、1ステップあたり処理トークン数を表示します。

---

## 4. 乱数・dtype・autocast文脈

- 出力ディレクトリ作成（masterのみ）
- 乱数seed設定（rankごとにoffset）
- TF32許可設定
- `device_type` と `ptdtype`（float32/bfloat16/float16）決定
- `ctx` を構築（CPUなら`nullcontext`、GPUなら`torch.amp.autocast`）

この `ctx` は forward 時に使われ、混合精度学習を統一的に扱います。

---

## 5. データ読み出し: `get_batch(split)`

ここはミニバッチ生成の中心です。

1. `train.bin` or `val.bin` を `np.memmap` で開く
2. ランダム開始位置 `ix` をサンプリング
3. 入力 `x`（長さ`block_size`）を作る
4. 正解 `y` を1トークン右にずらして作る（次トークン予測）
5. CUDA時は `pin_memory()` + `non_blocking=True` でGPU転送を効率化

返り値は `(x, y)` です。

---

## 6. チェックポイント再開準備用の初期値

- `iter_num = 0`
- `best_val_loss = 1e9`

を置いて、scratch/resumeどちらでも同じ変数で扱えるようにします。

---

## 7. 語彙サイズ推定（`meta.pkl`）

- `data/<dataset>/meta.pkl` があれば読み込み
- `meta['vocab_size']` を取得して `meta_vocab_size` に保存

データセット側で語彙情報がある場合、モデル初期化に反映できます。

---

## 8. モデル初期化（3分岐）

`model_args` を基準に、`init_from` で分岐します。

### `init_from == 'scratch'`

- 新規モデルを構築
- `meta_vocab_size` が無ければ 50304（GPT-2語彙を効率化した値）

### `init_from == 'resume'`

- `out_dir/ckpt.pt` 読み込み
- 互換性必須項目（層数や埋め込み次元など）をチェックポイント側に合わせる
- state_dict読み込み時に `_orig_mod.` 接頭辞があれば除去
- `iter_num`, `best_val_loss` も復元

### `init_from.startswith('gpt2')`

- OpenAI GPT-2重みから初期化
- 読み込んだモデルの構成値を `model_args` に反映

その後、必要なら `crop_block_size` で文脈長を短縮し、`model.to(device)` します。

---

## 9. Optimizer・GradScaler・compile・DDPラップ

- `GradScaler`（float16時有効）を初期化
- `configure_optimizers(...)` でoptimizer構築
- resume時はoptimizer状態も復元
- `compile=True` なら `torch.compile(model)`
- DDP時は `model = DDP(model, device_ids=[ddp_local_rank])`

---

## 10. 評価関数 `estimate_loss()`

`@torch.no_grad()` で勾配なし評価。

- `train`/`val` それぞれ `eval_iters` 回バッチを回す
- loss平均を返す
- `model.eval()` と `model.train()` を切り替え

定期評価やチェックポイント保存判定に使われます。

---

## 11. 学習率関数 `get_lr(it)`

3区間のスケジュールです。

1. warmup区間: 線形増加
2. decay終了後: `min_lr` 固定
3. その間: cosine decay

`decay_lr=False` の場合は固定 `learning_rate` を使います。

---

## 12. W&B初期化

`wandb_log and master_process` のときだけ `wandb.init(...)` します。

---

## 13. 学習ループ本体

`while True:` の中で以下を繰り返します。

### 13.1 学習率更新

各param groupの`lr`を現在ステップ値へ更新。

### 13.2 定期評価とチェックポイント保存

`iter_num % eval_interval == 0 and master_process` のとき:

- `estimate_loss()` 実行
- lossを表示
- W&Bへ指標送信（有効時）
- 改善時または常時保存設定時に `ckpt.pt` 保存

`eval_only=True` なら初回評価後に終了。

### 13.3 勾配蓄積付き学習ステップ

`for micro_step in range(gradient_accumulation_steps):`

- DDP時、最後のmicro stepだけ同期するよう `require_backward_grad_sync` を制御
- `with ctx:` でforwardし `loss` 計算
- `loss /= gradient_accumulation_steps` でスケーリング
- 次バッチを先読み
- `scaler.scale(loss).backward()` で逆伝播

### 13.4 勾配クリップ・最適化

- 必要なら `clip_grad_norm_`
- `scaler.step(optimizer)` / `scaler.update()`
- `optimizer.zero_grad(set_to_none=True)`

### 13.5 ログ出力

- 反復時間 `dt` を計測
- `lossf`（蓄積前相当のloss）を表示
- 数ステップ後から `estimate_mfu` でMFU（GPU利用効率目安）を平滑化表示

### 13.6 終了判定

- `iter_num > max_iters` でbreak
- ループ終了後、DDPなら `destroy_process_group()`

---

## 14. このスクリプトの学習上の要点

- 目的は一貫して**次トークン予測**（`y`は`x`の1トークン先）
- 実運用向け要素（DDP、mixed precision、grad accumulation、checkpoint、lr schedule）が短いコードにまとまっている
- `train.py`は「学習運用の骨格」、モデルの中身は`model.py`が担当

---

## 15. 読解チェック（理解確認）

1. `gradient_accumulation_steps` を増やすと、実効バッチサイズはどう変わるか？
2. DDP時に `require_backward_grad_sync` を最後だけ有効にする理由は？
3. `eval_only=True` はどのタイミングで終了するか？
4. `resume` 時に `model_args` の一部を強制上書きする理由は？

この4問に答えられれば、`train.py`の主要フローは把握できています。
