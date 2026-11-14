# -*- coding: utf-8 -*-
"""
Tests for International Gold (COMEX GC=F) and commodity support in US market review.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from data_provider.base import BaseFetcher, DataFetcherManager
from data_provider.yfinance_fetcher import YfinanceFetcher
from data_provider.yfinance_fundamental_adapter import _convert_to_yf_symbol
from src.core.market_profile import US_PROFILE, CN_PROFILE
from src.market_analyzer import (
    MarketAnalyzer,
    MarketIndex,
    MarketOverview,
    _CHINESE_SECTION_PATTERNS,
    _ENGLISH_SECTION_PATTERNS,
)


class TestYfinanceFetcherCommodities(unittest.TestCase):
    def setUp(self):
        self.fetcher = YfinanceFetcher()

    def test_non_us_region_returns_none(self):
        self.assertIsNone(self.fetcher.get_commodities(region="cn"))
        self.assertIsNone(self.fetcher.get_commodities(region="hk"))

    @patch("yfinance.Ticker")
    def test_get_commodities_success(self, mock_ticker_cls):
        # 30 days of synthetic price data ending with today's action
        dates = pd.date_range("2026-10-01", periods=25, freq="D")
        closes = [2600.0 + i * 5 for i in range(25)]
        closes[-2] = 2700.0  # prev close
        closes[-1] = 2727.0  # today close (+1.00%)
        df = pd.DataFrame(
            {
                "Open": [2690.0] * 24 + [2705.0],
                "High": [2710.0] * 24 + [2735.0],
                "Low": [2680.0] * 24 + [2695.0],
                "Close": closes,
                "Volume": [100000] * 25,
            },
            index=dates,
        )

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_ticker_cls.return_value = mock_ticker

        items = self.fetcher.get_commodities(region="us")
        self.assertIsNotNone(items)
        self.assertEqual(len(items), 1)
        gold = items[0]

        self.assertEqual(gold["code"], "GOLD")
        self.assertEqual(gold["symbol"], "GC=F")
        self.assertEqual(gold["name"], "国际黄金 (COMEX)")
        self.assertEqual(gold["name_en"], "Gold Futures (COMEX)")
        self.assertEqual(gold["unit"], "USD/oz")
        self.assertEqual(gold["current"], 2727.0)
        self.assertEqual(gold["prev_close"], 2700.0)
        self.assertEqual(gold["change"], 27.0)
        self.assertEqual(gold["change_pct"], 1.0)
        self.assertEqual(gold["open"], 2705.0)
        self.assertEqual(gold["high"], 2735.0)
        self.assertEqual(gold["low"], 2695.0)
        # amplitude = (2735 - 2695) / 2700 * 100 = 40 / 27 = 1.48%
        self.assertAlmostEqual(gold["amplitude"], 1.48, places=2)
        # 5d change: index -6 was closes[-6] = closes[19] = 2695.0; (2727 - 2695) / 2695 * 100
        self.assertIsNotNone(gold["change_5d_pct"])
        self.assertEqual(gold["high_20d"], 2735.0)
        self.assertIsNotNone(gold["ma20"])

    @patch("yfinance.Ticker")
    def test_get_commodities_handles_empty_or_nan(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        self.assertIsNone(self.fetcher.get_commodities(region="us"))

    @patch("yfinance.Ticker")
    def test_get_commodities_exception_fail_open(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = RuntimeError("network down")
        self.assertIsNone(self.fetcher.get_commodities(region="us"))

    def test_us_dotted_ticker_translation(self):
        self.assertEqual(self.fetcher._convert_stock_code("BRK.B"), "BRK.B")
        self.assertEqual(self.fetcher._to_yf_ticker_symbol("BRK.B"), "BRK-B")
        self.assertEqual(self.fetcher._to_yf_ticker_symbol("AAPL"), "AAPL")
        self.assertEqual(_convert_to_yf_symbol("BRK.B"), "BRK-B")
        self.assertEqual(_convert_to_yf_symbol("AAPL"), "AAPL")


class TestDataFetcherManagerCommodities(unittest.TestCase):
    def test_manager_get_commodities_non_us_returns_empty(self):
        manager = DataFetcherManager()
        self.assertEqual(manager.get_commodities(region="cn"), [])
        self.assertEqual(manager.get_commodities(region="hk"), [])

    def test_manager_get_commodities_us(self):
        manager = DataFetcherManager()
        mock_fetcher = MagicMock(spec=BaseFetcher)
        mock_fetcher.name = "YfinanceFetcher"
        mock_fetcher.get_commodities.return_value = [{"code": "GOLD", "current": 2700.0}]
        manager._fetchers = [mock_fetcher]

        res = manager.get_commodities(region="us")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["code"], "GOLD")


class TestMarketAnalyzerGoldIntegration(unittest.TestCase):
    def setUp(self):
        self.analyzer = MarketAnalyzer(region="us")
        self.original_language = getattr(self.analyzer.config, "report_language", "zh")
        self.analyzer.config.report_language = "zh"

    def tearDown(self):
        self.analyzer.config.report_language = self.original_language

    def test_overview_includes_commodities_for_us(self):
        with patch.object(self.analyzer.data_manager, "get_main_indices", return_value=[]), \
             patch.object(self.analyzer.data_manager, "get_sector_rankings", return_value=([], [])), \
             patch.object(
                 self.analyzer.data_manager,
                 "get_commodities",
                 return_value=[{
                     "code": "GOLD",
                     "name": "国际黄金 (COMEX)",
                     "name_en": "Gold Futures (COMEX)",
                     "current": 2720.0,
                     "change_pct": 0.5,
                     "unit": "USD/oz",
                 }],
             ):
            overview = self.analyzer.get_market_overview()
            self.assertEqual(len(overview.commodities), 1)
            self.assertEqual(overview.commodities[0]["current"], 2720.0)

    def test_overview_ignores_unmocked_magic_mock(self):
        self.analyzer.data_manager = MagicMock()
        # Unmocked get_commodities returns MagicMock, which must not be assigned as commodities
        self.analyzer._get_commodities(MarketOverview(date="2026-03-05"))
        ov = MarketOverview(date="2026-03-05")
        self.analyzer._get_commodities(ov)
        self.assertEqual(ov.commodities, [])

    def test_build_commodities_block_zh_and_en(self):
        overview = MarketOverview(
            date="2026-11-14",
            commodities=[{
                "code": "GOLD",
                "name": "国际黄金 (COMEX)",
                "name_en": "Gold Futures (COMEX)",
                "current": 2750.5,
                "change_pct": 0.85,
                "change_5d_pct": 2.1,
                "low": 2730.0,
                "high": 2760.0,
                "amplitude": 1.1,
                "low_20d": 2680.0,
                "high_20d": 2780.0,
                "ma20": 2710.0,
                "unit": "USD/oz",
            }],
        )

        # Chinese block
        self.analyzer.config.report_language = "zh"
        block_zh = self.analyzer._build_commodities_block(overview)
        self.assertIn("#### 国际黄金与避险资产", block_zh)
        self.assertIn("国际黄金 (COMEX)", block_zh)
        self.assertIn("2750.50 美元/盎司", block_zh)
        self.assertIn("+0.85%", block_zh)
        self.assertIn("+2.10%", block_zh)
        self.assertIn("2730.00 - 2760.00 (1.10%)", block_zh)

        # English block
        self.analyzer.config.report_language = "en"
        block_en = self.analyzer._build_commodities_block(overview)
        self.assertIn("#### International Gold & Safe-Haven Assets", block_en)
        self.assertIn("Gold Futures (COMEX)", block_en)
        self.assertIn("2750.50 USD/oz", block_en)

    def test_build_commodities_block_empty(self):
        overview = MarketOverview(date="2026-11-14")
        self.assertEqual(self.analyzer._build_commodities_block(overview), "")

    def test_inject_data_into_review_with_commodities_and_emoji_headings(self):
        overview = MarketOverview(
            date="2026-11-14",
            indices=[MarketIndex(code="SPX", name="标普500", current=6000.0, change_pct=0.4)],
            top_sectors=[{"name": "科技", "change_pct": 1.2}],
            bottom_sectors=[{"name": "能源", "change_pct": -0.8}],
            commodities=[{
                "code": "GOLD",
                "name": "国际黄金 (COMEX)",
                "name_en": "Gold Futures (COMEX)",
                "current": 2750.0,
                "change_pct": 0.5,
                "unit": "USD/oz",
            }],
        )

        review = """## 2026-11-14 美股大盘复盘

### 📈 一、盘面总览
美股今日总体温和收涨。

### 📉 二、指数结构
标普500领涨。

### 🧱 三、板块主线
科技板块领跑。

### 🪙 四、国际黄金与避险风向
黄金小幅走强，反映避险需求稳健。

### ⚡ 五、消息催化
宏观叙事平静。
"""
        injected = self.analyzer._inject_data_into_review(review, overview)

        # Indices table injected under 二
        self.assertIn("| 指数 | 最新 |", injected)
        # Sectors table injected under 三
        self.assertIn("#### 行业板块领涨 Top 5", injected)
        # Commodities table injected under 四
        self.assertIn("#### 国际黄金与避险资产", injected)
        # Zero duplicate section headings
        self.assertEqual(injected.count("### 🧱 三、板块主线"), 1)
        self.assertEqual(injected.count("### 4. Sector Highlights"), 0)

    def test_inject_data_into_review_appends_gold_when_heading_drifts(self):
        overview = MarketOverview(
            date="2026-11-14",
            commodities=[{
                "code": "GOLD",
                "name": "国际黄金 (COMEX)",
                "name_en": "Gold Futures (COMEX)",
                "current": 2750.0,
                "change_pct": 0.5,
                "unit": "USD/oz",
            }],
        )
        review = """## 2026-11-14 美股大盘复盘

### 自由格式点评
整体市场平淡。
"""
        injected = self.analyzer._inject_data_into_review(review, overview)
        self.assertIn("### 四、国际黄金走势", injected)
        self.assertIn("#### 国际黄金与避险资产", injected)

    def test_build_market_review_payload_includes_commodities(self):
        overview = MarketOverview(
            date="2026-11-14",
            commodities=[{"code": "GOLD", "current": 2750.0}],
        )
        payload = self.analyzer.build_market_review_payload(overview, [], "## Report")
        self.assertIn("commodities", payload)
        self.assertEqual(len(payload["commodities"]), 1)
        self.assertEqual(payload["commodities"][0]["current"], 2750.0)

    def test_template_review_includes_commodities(self):
        overview = MarketOverview(
            date="2026-11-14",
            indices=[MarketIndex(code="SPX", name="标普500", current=6000.0, change_pct=0.4)],
            commodities=[{
                "code": "GOLD",
                "name": "国际黄金 (COMEX)",
                "name_en": "Gold Futures (COMEX)",
                "current": 2750.0,
                "change_pct": 0.5,
                "unit": "USD/oz",
            }],
        )
        report = self.analyzer._generate_template_review(overview, [])
        self.assertIn("### 四、国际黄金与避险风向", report)
        self.assertIn("国际黄金 (COMEX)", report)

    def test_search_market_news_us_yfinance_news_fallback(self):
        # When search_service has no results, US analyzer falls back to yfinance market news
        self.analyzer.search_service = MagicMock()
        self.analyzer.search_service.search_stock_news.return_value = None

        with patch.object(
            self.analyzer,
            "_fetch_us_yfinance_market_news",
            return_value=[{"title": "Gold reaches new high", "snippet": "Macro news", "url": ""}],
        ):
            news = self.analyzer.search_market_news()
            self.assertEqual(len(news), 1)
            self.assertEqual(news[0]["title"], "Gold reaches new high")


if __name__ == "__main__":
    unittest.main()
