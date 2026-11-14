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

## 測試

```bash
pip install pytest
python -m pytest
```

`tests/` 入面嘅檢查大約一分鐘行完,唔會連馬會網站,亦唔會改 `data/` 同 `docs/`(全部喺臨時資料夾做)。
每次改 `hkjc/`、`docs/index.html`、`tests/` 或 workflow,GitHub 會自動行一次(Actions → tests);紅燈即係有改動整壞咗嘢。
部分測試要 Chrome(經 Playwright)、bash 同 jq:本機冇就自動跳過,雲端就一定要行。

涵蓋:
- 開跑時鎖定預測、開跑前賠率嘅讀取時間、賽果快出
- 模擬倉落注同結算(包括退出馬退款)、彩池測試、新資料測試點樣判斷、自我學習紀錄、實戰往績
- 模型冇偷睇未來數據
- 用真瀏覽器讀賠率頁(本機假頁面,同馬會頁面用同一批 id)
- 用真瀏覽器開 `docs/index.html`,數據由真實程式產生,檢查每個部分有冇正常顯示
- 雲端時鐘幾時 5 分鐘、2 分鐘、30 分鐘更新一次,同埋幾時發佈網站(直接行 workflow 入面嗰段 shell)
- 馬會賽果頁、分段時間頁、排位表嘅解析(`tests/fixtures/pages` 係按真實頁面結構砌嘅縮細版)
- `tests/test_meeting_1111.py` 用 2026-11-11 嗰日儲低嘅真實檔案重播:模擬倉結餘同彩池測試數字一改就會紅燈
- 馬會取消一場賽事(2026-11-14 沙田第 6 場):認得馬會嘅取消通告、唔再等賽果同讀賠率、
  網頁顯示「取消」、模擬倉當冇落注。取消嘅場次記喺 `data/cancelled.json`,唔會寫入賽果歷史

測試睇唔到馬會幾時改版,呢樣由每晚自我檢測負責:「馬會頁面」會重讀最近一場已儲存嘅賽果同分段時間再對比,
「賽前賠率」喺賽馬日朝早檢查賠率頁有冇讀到。

## 檔案

```
hkjc/scraper.py   爬馬會公開頁面
hkjc/model.py     特徵、訓練、回測
hkjc/analyze.py   將模型輸出變成可解釋分析
hkjc/agent.py     Claude AI 分析(有快取,數據冇變唔會重複收費)
hkjc/__main__.py  流程
hkjc/trackwork.py 賽前儲存晨操同傷患紀錄(暫時只收集,未入模型)
hkjc/aerial.py    讀馬會「模擬鳥瞰圖」,儲低每匹馬喺 800/400/200 米同終點嘅位置(暫時只收集,未入模型)
hkjc/watch.py     賽馬日監察:`python -m hkjc.watch` 睇系統而家有冇正常運作(只讀;加 --act 可以重新觸發更新或時鐘)
tests/            自動測試(見上面「測試」)
data/             歷史賽果、排位表(由 workflow 自動 commit)
docs/             網頁 + 分析 JSON
```

## 重要說明

- 任何模型都冇辦法「保證」準確;網頁會同時顯示回測同實戰往績,包括同大熱門比較,請以數據判斷。
- 即時投注賠率嘅 API 有白名單限制,本系統冇使用;只用馬會公開賽馬資訊頁。
- 未滿十八歲人士不得投注。切勿沉迷賭博,平和基金熱線 1834 633。
