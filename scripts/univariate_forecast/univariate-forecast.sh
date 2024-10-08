# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_yearly.json" --save-path "yearly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "darts.LinearRegressionModel" "darts.RandomForest" "darts.KalmanForecaster" "darts.XGBModel" 
# echo "Finished running Group 1 yearly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_quarterly.json" --save-path "quarterly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "darts.LinearRegressionModel" "darts.RandomForest" "darts.KalmanForecaster" "darts.XGBModel" 
# echo "Finished running Group 1 quarterly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_monthly.json" --save-path "monthly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "darts.LinearRegressionModel" "darts.RandomForest" "darts.KalmanForecaster" "darts.XGBModel" 
# echo "Finished running Group 1 monthly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json" --save-path "weekly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "darts.LinearRegressionModel" "darts.RandomForest" "darts.KalmanForecaster" "darts.XGBModel" 
# echo "Finished running Group 1 weekly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_daily.json" --save-path "daily" --gpus 0 --num-workers 1 --timeout 60000 --model-name "darts.LinearRegressionModel" "darts.RandomForest" "darts.KalmanForecaster" "darts.XGBModel" 
# echo "Finished running Group 1 daily"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_hourly.json" --save-path "hourly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "darts.LinearRegressionModel" "darts.RandomForest" "darts.KalmanForecaster" "darts.XGBModel" 
# echo "Finished running Group 1 hourly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_other.json" --save-path "other" --gpus 0 --num-workers 1 --timeout 60000 --model-name "darts.LinearRegressionModel" "darts.RandomForest" "darts.KalmanForecaster" "darts.XGBModel" 
# echo "Finished running Group 1 other"


python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_yearly.json" --save-path "yearly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "darts.RNNModel" "darts.BlockRNNModel" "darts.TiDEModel" 
echo "Finished running Group 2 yearly"

# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_yearly.json" --save-path "yearly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "darts.NaiveMean" "darts.NaiveSeasonal" "darts.NaiveDrift" "darts.NaiveMovingAverage" "darts.RNNModel" "darts.BlockRNNModel" "darts.TiDEModel" 
# echo "Finished running Group 2 yearly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_quarterly.json" --save-path "quarterly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "darts.NaiveMean" "darts.NaiveSeasonal" "darts.NaiveDrift" "darts.NaiveMovingAverage" "darts.RNNModel" "darts.BlockRNNModel" "darts.TiDEModel" 
# echo "Finished running Group 2 quarterly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_monthly.json" --save-path "monthly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "darts.NaiveMean" "darts.NaiveSeasonal" "darts.NaiveDrift" "darts.NaiveMovingAverage" "darts.RNNModel" "darts.BlockRNNModel" "darts.TiDEModel" 
# echo "Finished running Group 2 monthly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json" --save-path "weekly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "darts.NaiveMean" "darts.NaiveSeasonal" "darts.NaiveDrift" "darts.NaiveMovingAverage" "darts.RNNModel" "darts.BlockRNNModel" "darts.TiDEModel" 
# echo "Finished running Group 2 weekly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_daily.json" --save-path "daily" --gpus 0 --num-workers 1 --timeout 60000 --model-name "darts.NaiveMean" "darts.NaiveSeasonal" "darts.NaiveDrift" "darts.NaiveMovingAverage" "darts.RNNModel" "darts.BlockRNNModel" "darts.TiDEModel" 
# echo "Finished running Group 2 daily"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_hourly.json" --save-path "hourly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "darts.NaiveMean" "darts.NaiveSeasonal" "darts.NaiveDrift" "darts.NaiveMovingAverage" "darts.RNNModel" "darts.BlockRNNModel" "darts.TiDEModel" 
# echo "Finished running Group 2 hourly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_other.json" --save-path "other" --gpus 0 --num-workers 1 --timeout 60000 --model-name "darts.NaiveMean" "darts.NaiveSeasonal" "darts.NaiveDrift" "darts.NaiveMovingAverage" "darts.RNNModel" "darts.BlockRNNModel" "darts.TiDEModel" 
# echo "Finished running Group 2 other"


# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_yearly.json" --save-path "yearly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "time_series_library.Triformer" "time_series_library.PatchTST" "time_series_library.Nonstationary_Transformer" "time_series_library.Informer" "time_series_library.TimesNet" "time_series_library.FEDformer" "time_series_library.NLinear" "time_series_library.Linear" "time_series_library.DLinear" "time_series_library.FiLM" "time_series_library.MICN" "time_series_library.Crossformer" "darts.TCNModel" "darts.NBEATSModel" "darts.NHiTSModel" "darts.BlockRNNModel" "darts.StatsForecastAutoETS" "darts.StatsForecastAutoCES" "darts.StatsForecastAutoTheta" "darts.AutoARIMA"  --model-hyper-params   "{\"d_model\":8,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":16,\"patch_len\":4,\"stride\":2}"   "{\"p_hidden_layers\":2,\"p_hidden_dims\":[256,256],\"d_model\":8,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":16,\"factor\":3}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}" --adapter "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" 
# echo "Finished running Group 3 yearly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_quarterly.json" --save-path "quarterly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "time_series_library.Triformer" "time_series_library.PatchTST" "time_series_library.Nonstationary_Transformer" "time_series_library.Informer" "time_series_library.TimesNet" "time_series_library.FEDformer" "time_series_library.NLinear" "time_series_library.Linear" "time_series_library.DLinear" "time_series_library.FiLM" "time_series_library.MICN" "time_series_library.Crossformer" "darts.TCNModel" "darts.NBEATSModel" "darts.NHiTSModel" "darts.BlockRNNModel" "darts.StatsForecastAutoETS" "darts.StatsForecastAutoCES" "darts.StatsForecastAutoTheta" "darts.AutoARIMA"  --model-hyper-params   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"patch_len\":4,\"stride\":2}"   "{\"p_hidden_layers\":2,\"p_hidden_dims\":[256,256],\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}" --adapter "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" 
# echo "Finished running Group 3 quarterly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_monthly.json" --save-path "monthly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "time_series_library.Triformer" "time_series_library.PatchTST" "time_series_library.Nonstationary_Transformer" "time_series_library.Informer" "time_series_library.TimesNet" "time_series_library.FEDformer" "time_series_library.NLinear" "time_series_library.Linear" "time_series_library.DLinear" "time_series_library.FiLM" "time_series_library.MICN" "time_series_library.Crossformer" "darts.TCNModel" "darts.NBEATSModel" "darts.NHiTSModel" "darts.BlockRNNModel" "darts.StatsForecastAutoETS" "darts.StatsForecastAutoCES" "darts.StatsForecastAutoTheta" "darts.AutoARIMA"  --model-hyper-params   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"patch_len\":8,\"stride\":4}"   "{\"p_hidden_layers\":2,\"p_hidden_dims\":[256,256],\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}" --adapter "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" 
# echo "Finished running Group 3 monthly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json" --save-path "weekly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "time_series_library.Triformer" "time_series_library.PatchTST" "time_series_library.Nonstationary_Transformer" "time_series_library.Informer" "time_series_library.TimesNet" "time_series_library.FEDformer" "time_series_library.NLinear" "time_series_library.Linear" "time_series_library.DLinear" "time_series_library.FiLM" "time_series_library.MICN" "time_series_library.Crossformer" "darts.TCNModel" "darts.NBEATSModel" "darts.NHiTSModel" "darts.BlockRNNModel" "darts.StatsForecastAutoETS" "darts.StatsForecastAutoCES" "darts.StatsForecastAutoTheta" "darts.AutoARIMA"  --model-hyper-params   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"patch_len\":8,\"stride\":4}"   "{\"p_hidden_layers\":2,\"p_hidden_dims\":[256,256],\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}" --adapter "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" 
# echo "Finished running Group 3 weekly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_daily.json" --save-path "daily" --gpus 0 --num-workers 1 --timeout 60000 --model-name "time_series_library.Triformer" "time_series_library.PatchTST" "time_series_library.Nonstationary_Transformer" "time_series_library.Informer" "time_series_library.TimesNet" "time_series_library.FEDformer" "time_series_library.NLinear" "time_series_library.Linear" "time_series_library.DLinear" "time_series_library.FiLM" "time_series_library.MICN" "time_series_library.Crossformer" "darts.TCNModel" "darts.NBEATSModel" "darts.NHiTSModel" "darts.BlockRNNModel" "darts.StatsForecastAutoETS" "darts.StatsForecastAutoCES" "darts.StatsForecastAutoTheta" "darts.AutoARIMA"  --model-hyper-params   "{\"d_model\":8,\"d_ff\":8,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":8,\"patch_len\":4,\"stride\":2}"   "{\"p_hidden_layers\":2,\"p_hidden_dims\":[256,256],\"d_model\":8,\"d_ff\":8,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":8,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":8,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":8,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":8,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":8,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":8,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":8,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":8,\"factor\":3}"   "{\"d_model\":8,\"d_ff\":8,\"factor\":3}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}" --adapter "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" 
# echo "Finished running Group 3 daily"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_hourly.json" --save-path "hourly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "time_series_library.Triformer" "time_series_library.PatchTST" "time_series_library.Nonstationary_Transformer" "time_series_library.Informer" "time_series_library.TimesNet" "time_series_library.FEDformer" "time_series_library.NLinear" "time_series_library.Linear" "time_series_library.DLinear" "time_series_library.FiLM" "time_series_library.MICN" "time_series_library.Crossformer" "darts.TCNModel" "darts.NBEATSModel" "darts.NHiTSModel" "darts.BlockRNNModel" "darts.StatsForecastAutoETS" "darts.StatsForecastAutoCES" "darts.StatsForecastAutoTheta" "darts.AutoARIMA"  --model-hyper-params   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"patch_len\":4,\"stride\":2}"   "{\"p_hidden_layers\":2,\"p_hidden_dims\":[256,256],\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}" --adapter "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" 
# echo "Finished running Group 3 hourly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_other.json" --save-path "other" --gpus 0 --num-workers 1 --timeout 60000 --model-name "time_series_library.Triformer" "time_series_library.PatchTST" "time_series_library.Nonstationary_Transformer" "time_series_library.Informer" "time_series_library.TimesNet" "time_series_library.FEDformer" "time_series_library.NLinear" "time_series_library.Linear" "time_series_library.DLinear" "time_series_library.FiLM" "time_series_library.MICN" "time_series_library.Crossformer" "darts.TCNModel" "darts.NBEATSModel" "darts.NHiTSModel" "darts.BlockRNNModel" "darts.StatsForecastAutoETS" "darts.StatsForecastAutoCES" "darts.StatsForecastAutoTheta" "darts.AutoARIMA"  --model-hyper-params   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"patch_len\":4,\"stride\":2}"   "{\"p_hidden_layers\":2,\"p_hidden_dims\":[256,256],\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-4}}" --adapter "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" 
# echo "Finished running Group 3 other"


# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_yearly.json" --save-path "yearly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "time_series_library.Triformer" "time_series_library.PatchTST" "time_series_library.Nonstationary_Transformer" "time_series_library.Informer" "time_series_library.TimesNet" "time_series_library.FEDformer" "time_series_library.NLinear" "time_series_library.Linear" "time_series_library.DLinear" "time_series_library.FiLM" "time_series_library.MICN" "time_series_library.Crossformer" "darts.TCNModel" "darts.NBEATSModel" "darts.NHiTSModel" "darts.BlockRNNModel"  --model-hyper-params   "{\"d_model\":16,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":32,\"patch_len\":4,\"stride\":2}"   "{\"p_hidden_layers\":2,\"p_hidden_dims\":[256,256],\"d_model\":16,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":32,\"factor\":3}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}" --adapter "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" 
# echo "Finished running Group 4 yearly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_quarterly.json" --save-path "quarterly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "time_series_library.Triformer" "time_series_library.PatchTST" "time_series_library.Nonstationary_Transformer" "time_series_library.Informer" "time_series_library.TimesNet" "time_series_library.FEDformer" "time_series_library.NLinear" "time_series_library.Linear" "time_series_library.DLinear" "time_series_library.FiLM" "time_series_library.MICN" "time_series_library.Crossformer" "darts.TCNModel" "darts.NBEATSModel" "darts.NHiTSModel" "darts.BlockRNNModel"  --model-hyper-params   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"patch_len\":4,\"stride\":2}"   "{\"p_hidden_layers\":2,\"p_hidden_dims\":[256,256],\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}" --adapter "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" 
# echo "Finished running Group 4 quarterly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_monthly.json" --save-path "monthly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "time_series_library.Triformer" "time_series_library.PatchTST" "time_series_library.Nonstationary_Transformer" "time_series_library.Informer" "time_series_library.TimesNet" "time_series_library.FEDformer" "time_series_library.NLinear" "time_series_library.Linear" "time_series_library.DLinear" "time_series_library.FiLM" "time_series_library.MICN" "time_series_library.Crossformer" "darts.TCNModel" "darts.NBEATSModel" "darts.NHiTSModel" "darts.BlockRNNModel"  --model-hyper-params   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"patch_len\":8,\"stride\":4}"   "{\"p_hidden_layers\":2,\"p_hidden_dims\":[256,256],\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}" --adapter "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" 
# echo "Finished running Group 4 monthly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json" --save-path "weekly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "time_series_library.Triformer" "time_series_library.PatchTST" "time_series_library.Nonstationary_Transformer" "time_series_library.Informer" "time_series_library.TimesNet" "time_series_library.FEDformer" "time_series_library.NLinear" "time_series_library.Linear" "time_series_library.DLinear" "time_series_library.FiLM" "time_series_library.MICN" "time_series_library.Crossformer" "darts.TCNModel" "darts.NBEATSModel" "darts.NHiTSModel" "darts.BlockRNNModel"  --model-hyper-params   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"patch_len\":8,\"stride\":4}"   "{\"p_hidden_layers\":2,\"p_hidden_dims\":[256,256],\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}" --adapter "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" 
# echo "Finished running Group 4 weekly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_daily.json" --save-path "daily" --gpus 0 --num-workers 1 --timeout 60000 --model-name "time_series_library.Triformer" "time_series_library.PatchTST" "time_series_library.Nonstationary_Transformer" "time_series_library.Informer" "time_series_library.TimesNet" "time_series_library.FEDformer" "time_series_library.NLinear" "time_series_library.Linear" "time_series_library.DLinear" "time_series_library.FiLM" "time_series_library.MICN" "time_series_library.Crossformer" "darts.TCNModel" "darts.NBEATSModel" "darts.NHiTSModel" "darts.BlockRNNModel"  --model-hyper-params   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"patch_len\":8,\"stride\":4}"   "{\"p_hidden_layers\":2,\"p_hidden_dims\":[256,256],\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"d_model\":16,\"d_ff\":16,\"factor\":3}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}" --adapter "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" 
# echo "Finished running Group 4 daily"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_hourly.json" --save-path "hourly" --gpus 0 --num-workers 1 --timeout 60000 --model-name "time_series_library.Triformer" "time_series_library.PatchTST" "time_series_library.Nonstationary_Transformer" "time_series_library.Informer" "time_series_library.TimesNet" "time_series_library.FEDformer" "time_series_library.NLinear" "time_series_library.Linear" "time_series_library.DLinear" "time_series_library.FiLM" "time_series_library.MICN" "time_series_library.Crossformer" "darts.TCNModel" "darts.NBEATSModel" "darts.NHiTSModel" "darts.BlockRNNModel"  --model-hyper-params   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"patch_len\":16,\"stride\":8}"   "{\"p_hidden_layers\":2,\"p_hidden_dims\":[256,256],\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"d_model\":32,\"d_ff\":32,\"factor\":3}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}" --adapter "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" 
# echo "Finished running Group 4 hourly"
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_other.json" --save-path "other" --gpus 0 --num-workers 1 --timeout 60000 --model-name "time_series_library.Triformer" "time_series_library.PatchTST" "time_series_library.Nonstationary_Transformer" "time_series_library.Informer" "time_series_library.TimesNet" "time_series_library.FEDformer" "time_series_library.NLinear" "time_series_library.Linear" "time_series_library.DLinear" "time_series_library.FiLM" "time_series_library.MICN" "time_series_library.Crossformer" "darts.TCNModel" "darts.NBEATSModel" "darts.NHiTSModel" "darts.BlockRNNModel"  --model-hyper-params   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"patch_len\":4,\"stride\":2}"   "{\"p_hidden_layers\":2,\"p_hidden_dims\":[256,256],\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"d_model\":64,\"d_ff\":64,\"factor\":3}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"   "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}" --adapter "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" "transformer_adapter" 
# echo "Finished running Group 4 other"


