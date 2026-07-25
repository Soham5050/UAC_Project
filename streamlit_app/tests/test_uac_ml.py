import unittest
import warnings

import numpy as np
import pandas as pd

from uac_ml import build_feature_frame, build_supervised_frame, run_ml_pipeline


def synthetic_enriched(rows: int = 180) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "cbp_inflow": 100 + index,
            "cbp_stock": 300 + index,
            "cbp_to_hhs": 70 + index,
            "hhs_stock": 500 + index,
            "hhs_discharged": 50 + index,
            "system_net_flow": index.astype(float),
            "transfer_efficiency_ratio": np.full(rows, 0.5),
            "discharge_effectiveness": np.full(rows, 0.1),
            "cbp_bottleneck": (index % 31 == 0),
            "hhs_bottleneck": (index % 37 == 0),
        }
    )


class MLDataPreparationTests(unittest.TestCase):
    def test_lag_features_only_use_previous_reporting_sessions(self):
        source = synthetic_enriched()
        features = build_feature_frame(source)
        self.assertEqual(features.loc[14, "cbp_inflow_lag_1"], source.loc[13, "cbp_inflow"])
        self.assertEqual(features.loc[14, "cbp_inflow_mean_7"], source.loc[7:13, "cbp_inflow"].mean())

    def test_forecast_target_is_seven_reporting_sessions_ahead(self):
        source = synthetic_enriched()
        supervised = build_supervised_frame(source, horizon=7)
        origin = supervised.features.index[0]
        self.assertEqual(
            supervised.targets["hhs_discharged"].loc[origin, "horizon_7"],
            source.loc[origin + 7, "hhs_discharged"],
        )


class MLPipelineTests(unittest.TestCase):
    def test_pipeline_returns_seven_forecasts_and_bounded_risk_scores(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = run_ml_pipeline(synthetic_enriched(210), lstm_epochs=2)
        self.assertFalse(any("valid feature names" in str(warning.message) for warning in captured))
        self.assertEqual(len(result.forecasts), 2)
        for forecast in result.forecasts.values():
            self.assertEqual(len(forecast.values), 7)
            self.assertTrue(np.isfinite(forecast.values["prediction"]).all())
            self.assertTrue((forecast.values["lower_80"] <= forecast.values["upper_80"]).all())
        for risk in result.risks.values():
            if risk.available:
                self.assertGreaterEqual(risk.probability, 0.0)
                self.assertLessEqual(risk.probability, 1.0)

    def test_active_model_is_the_lowest_validation_mae_candidate(self):
        """Regression test for a real bug: the persistence baseline was computed and shown in the
        comparison table but never included as a candidate for the 'Active' model, so a much worse
        Tabular ML model was always selected by default. This asserts the winner shown in
        `forecasts` always matches whichever row actually has the lowest Validation MAE for that
        target in `comparison` — including the persistence baseline as an eligible winner."""
        result = run_ml_pipeline(synthetic_enriched(210), lstm_epochs=2)
        for target, forecast in result.forecasts.items():
            target_label = target.replace("_", " ").title()
            rows = result.comparison[result.comparison["Target"] == target_label]
            best_row = rows.loc[rows["Validation MAE"].idxmin()]
            self.assertEqual(
                forecast.model_name,
                best_row["Model"],
                f"Active model for {target} was '{forecast.model_name}' but "
                f"'{best_row['Model']}' had the lowest validation MAE ({best_row['Validation MAE']}).",
            )


if __name__ == "__main__":
    unittest.main()
