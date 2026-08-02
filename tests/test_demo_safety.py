import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from ensemble_strategies import (
    EnsembleSystem,
    StrategySignal,
    StrategyType,
)
from kraken_bot_v4_advanced import (
    TradingBotV4,
    TradingPair,
    has_required_kraken_credentials,
)


class DemoSafetyTests(unittest.TestCase):
    def make_bot(self, dry_run=True):
        bot = TradingBotV4.__new__(TradingBotV4)
        bot.config = SimpleNamespace(
            DRY_RUN=dry_run,
            SIMULATION_BALANCE=1000.0,
            REGIME_LOOKBACK=30,
            USE_VOLATILE_ENTRY_FILTER=True,
            USE_SMA200_FILTER=True,
            BASE_STOP_LOSS=4.0,
        )
        bot.kraken = Mock()
        bot.telegram = Mock()
        bot.trades_history = []
        bot._save_trades_history = Mock()
        bot._send_v4_notification = Mock()
        return bot

    @staticmethod
    def make_params(capital=100.0, price=100.0, signal='BUY', leverage=1):
        return {
            'capital': capital,
            'price': price,
            'leverage': leverage,
            'data': pd.DataFrame(),
            'analysis': {
                'final_signal': signal,
                'confidence': 0.8,
                'reasons': ['test'],
                'v4_data': {},
            },
        }

    def test_paper_uses_virtual_funds_without_private_calls(self):
        bot = self.make_bot(dry_run=True)

        funds = bot._get_account_funds()

        self.assertEqual(funds, (1000.0, 1000.0, 'EUR'))
        bot.kraken.get_balance.assert_not_called()
        bot.kraken.get_available_margin.assert_not_called()

    def test_live_uses_private_account_funds(self):
        bot = self.make_bot(dry_run=False)
        bot.kraken.get_balance.return_value = (250.0, 'EUR')
        bot.kraken.get_available_margin.return_value = 200.0

        self.assertEqual(bot._get_account_funds(), (250.0, 200.0, 'EUR'))
        bot.kraken.get_balance.assert_called_once_with()
        bot.kraken.get_available_margin.assert_called_once_with()

    def test_credentials_are_required_only_in_live_mode(self):
        paper = SimpleNamespace(DRY_RUN=True, KRAKEN_API_KEY='', KRAKEN_API_SECRET='')
        live_missing = SimpleNamespace(DRY_RUN=False, KRAKEN_API_KEY='', KRAKEN_API_SECRET='')
        live_ready = SimpleNamespace(DRY_RUN=False, KRAKEN_API_KEY='key', KRAKEN_API_SECRET='secret')

        self.assertTrue(has_required_kraken_credentials(paper))
        self.assertFalse(has_required_kraken_credentials(live_missing))
        self.assertTrue(has_required_kraken_credentials(live_ready))

    def test_entry_filter_flags_control_volatile_and_sma200(self):
        bot = self.make_bot()
        data = pd.DataFrame({
            'Close': [100.0] * 200,
            'High': [101.0] * 200,
            'Low': [99.0] * 200,
        })

        with patch('kraken_bot_v4_advanced.RegimeDetector.detect', return_value='VOLATILE'):
            self.assertEqual(bot._get_entry_filter_rejection('BUY', data, 90.0), 'volatile')
            bot.config.USE_VOLATILE_ENTRY_FILTER = False
            self.assertEqual(bot._get_entry_filter_rejection('BUY', data, 90.0), 'sma200')
            bot.config.USE_SMA200_FILTER = False
            self.assertIsNone(bot._get_entry_filter_rejection('BUY', data, 90.0))

    def test_ensemble_min_score_is_configurable(self):
        weights = {
            StrategyType.SWING: 0.30,
            StrategyType.MOMENTUM: 0.25,
            StrategyType.MEAN_REVERSION: 0.25,
            StrategyType.TREND_FOLLOWING: 0.20,
        }
        votes = {
            StrategyType.SWING: StrategySignal(
                StrategyType.SWING, 'BUY', 0.90, 100.0, 'test'
            )
        }

        strict = EnsembleSystem(weights=weights, min_score=0.30)
        demo = EnsembleSystem(weights=weights, min_score=0.25)

        self.assertIsNone(strict._aggregate_votes(votes)[0])
        signal, confidence, consensus = demo._aggregate_votes(votes)
        self.assertEqual(signal, 'BUY')
        self.assertAlmostEqual(confidence, 0.90)
        self.assertEqual(consensus, 1.0)

    def test_minimum_volume_does_not_persist(self):
        bot = self.make_bot()
        pair = TradingPair('TEST-EUR', 'TESTEUR', 2.0, 0.10)

        result = bot.open_position(pair, self.make_params(capital=100.0, price=100.0))

        self.assertEqual(result, 'minimum_volume')
        self.assertEqual(bot.trades_history, [])
        bot._save_trades_history.assert_not_called()

    def test_failed_live_order_does_not_persist(self):
        bot = self.make_bot(dry_run=False)
        bot.kraken.place_order.side_effect = RuntimeError('Kraken rejected order')
        pair = TradingPair('TEST-EUR', 'TESTEUR', 0.1, 0.10)

        result = bot.open_position(pair, self.make_params())

        self.assertEqual(result, 'api_error')
        self.assertEqual(bot.trades_history, [])
        bot._save_trades_history.assert_not_called()

    def test_empty_live_acknowledgement_does_not_persist(self):
        bot = self.make_bot(dry_run=False)
        bot.kraken.place_order.return_value = {}
        pair = TradingPair('TEST-EUR', 'TESTEUR', 0.1, 0.10)

        result = bot.open_position(pair, self.make_params())

        self.assertEqual(result, 'api_error')
        self.assertEqual(bot.trades_history, [])
        bot._save_trades_history.assert_not_called()

    def test_paper_order_persists_as_simulation(self):
        bot = self.make_bot(dry_run=True)
        pair = TradingPair('TEST-EUR', 'TESTEUR', 0.1, 0.10)

        result = bot.open_position(pair, self.make_params())

        self.assertEqual(result, 'opened')
        self.assertEqual(bot.trades_history[0]['mode'], 'SIMULATION')
        bot.kraken.place_order.assert_not_called()
        bot._save_trades_history.assert_called_once_with()

    def test_paper_allows_virtual_short_at_leverage_one(self):
        bot = self.make_bot(dry_run=True)
        pair = TradingPair('TEST-EUR', 'TESTEUR', 0.1, 0.10)

        result = bot.open_position(
            pair, self.make_params(signal='SELL', leverage=1)
        )

        self.assertEqual(result, 'opened')
        self.assertEqual(bot.trades_history[0]['type'], 'short')
        bot.kraken.place_order.assert_not_called()

    def test_live_rejects_spot_short_at_leverage_one(self):
        bot = self.make_bot(dry_run=False)
        pair = TradingPair('TEST-EUR', 'TESTEUR', 0.1, 0.10)

        result = bot.open_position(
            pair, self.make_params(signal='SELL', leverage=1)
        )

        self.assertEqual(result, 'spot_sell')
        self.assertEqual(bot.trades_history, [])
        bot.kraken.place_order.assert_not_called()

    def test_successful_live_order_persists_after_confirmation(self):
        bot = self.make_bot(dry_run=False)
        bot.kraken.place_order.return_value = {'txid': ['test-id']}
        pair = TradingPair('TEST-EUR', 'TESTEUR', 0.1, 0.10)

        result = bot.open_position(pair, self.make_params())

        self.assertEqual(result, 'opened')
        self.assertEqual(len(bot.trades_history), 1)
        self.assertEqual(bot.trades_history[0]['mode'], 'REAL')
        bot.kraken.place_order.assert_called_once_with(
            pair='TESTEUR',
            order_type='buy',
            volume=1.0,
            leverage=1,
            reduce_only=False,
            close_stop_price=96.0,
        )
        bot._save_trades_history.assert_called_once_with()

    def test_scheduled_workflow_is_forced_to_paper(self):
        workflow = Path('.github/workflows/trading-bot-v4.yml').read_text(encoding='utf-8')

        self.assertIn("DRY_RUN: ${{ github.event_name == 'schedule' && 'true' || inputs.dry_run }}", workflow)
        self.assertIn('SIMULATION_BALANCE: "1000"', workflow)
        live_secret_guard = (
            "github.event_name == 'workflow_dispatch' && "
            "inputs.dry_run == 'false' && secrets.KRAKEN_API_KEY"
        )
        self.assertIn(live_secret_guard, workflow)
        self.assertIn("USE_SENTIMENT_ANALYSIS: ${{ github.event_name == 'schedule' && 'false'", workflow)
        self.assertIn("USE_ONCHAIN_ANALYSIS: ${{ github.event_name == 'schedule' && 'false'", workflow)
        self.assertIn('group: trading-bot-v4-${{ github.ref }}', workflow)
        self.assertIn('cancel-in-progress: false', workflow)


if __name__ == '__main__':
    unittest.main()
