# 香港賽馬量化分析系統 (hkjc-quant)

全自動喺 GitHub 雲端運行:唔使開住自己部電腦。每日自動攞馬會公開數據、重新訓練模型、
分析下一個賽馬日,賽馬日每 5 分鐘更新一次,結果顯示喺 GitHub Pages 網頁(香港可直接瀏覽)。

## 功能

| 部分 | 內容 |
|---|---|
| 數據 | 由 racing.hkjc.com 公開頁面攞排位表、賽果、沿途走位、分段時間、官方評分、配備、走勢評述、事件報告、試閘 |
| 速度指數 | 按跑道/路程自動校準標準時間,再按場地狀況同負磅調整 |
| 步速分析 | 由沿途走位計每匹馬嘅跑法(領放/前列/中游/後上),再按今場前領馬數目推算步速快慢,判斷邊類跑法著數 |
| 場地偏差 | 每條跑道/賽道/路程嘅歷史檔位統計 + 跑法偏差 |
| 即日場地走勢 | 賽馬日每跑完一場,即時計算今日前置定後上有利、內檔定外檔有利,用落之後嘅場次 |
| 模型 | Conditional logit(每場 softmax),以時間順序重播歷史數據訓練,冇偷睇未來數據;最後 20% 作回測 |
| AI Agent | Claude 讀取量化結果,寫廣東話賽事分析(步速、場地、首選、風險) |
| 往績 | 賽前鎖定預測,賽後自動對賽果,計首選勝出率同平注回報 |

## 設定(一次過)

1. **Settings → Pages → Source 揀 "GitHub Actions"**
2. **Settings → Secrets and variables → Actions → New repository secret**
   - `ANTHROPIC_API_KEY`:你嘅 Claude API key(冇都照行,只係冇 AI 文字分析)
   - `HKJC_PROXY`(可選):如果馬會封鎖 GitHub 伺服器 IP,填一個香港 HTTP proxy,例如 `http://user:pass@host:port`
3. **Actions → update → Run workflow** 行第一次

之後全自動:ticker 喺賽事期間每 5 分鐘更新,其餘時間(08:00–23:59 HKT)每 30 分鐘更新,每晚自我檢測一次。

## 本機運行

```bash
pip install -r requirements.txt
python -m hkjc backfill 120     # 下載過去 120 個賽馬日
python -m hkjc run              # 完整更新一次(--no-agent 唔叫 AI)
```

然後開 `docs/index.html`(需要經 web server,例如 `python -m http.server -d docs`)。

## 檔案

```
hkjc/scraper.py   爬馬會公開頁面
hkjc/model.py     特徵、訓練、回測
hkjc/analyze.py   將模型輸出變成可解釋分析
hkjc/agent.py     Claude AI 分析(有快取,數據冇變唔會重複收費)
hkjc/__main__.py  流程
data/             歷史賽果、排位表(由 workflow 自動 commit)
docs/             網頁 + 分析 JSON
```

## 重要說明

- 任何模型都冇辦法「保證」準確;網頁會同時顯示回測同實戰往績,包括同大熱門比較,請以數據判斷。
- 即時投注賠率嘅 API 有白名單限制,本系統冇使用;只用馬會公開賽馬資訊頁。
- 未滿十八歲人士不得投注。切勿沉迷賭博,平和基金熱線 1834 633。
