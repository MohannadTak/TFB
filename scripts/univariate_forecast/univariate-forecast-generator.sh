# Define main variables for gpus, num-workers, and timeout
gpus=0
num_workers=1
timeout=60000
script_path="./scripts/run_benchmark.py"

# Define the output file
output_file="univariate-forecast.sh"


# Function to generate the long hyperparameter string for group 3 (Yearly)
generate_hyperparams_string() {
  local d_model=$1
  local d_ff=$2
  local factor=$3
  local patch_len=$4
  local stride=$5
  local p_hidden_layers=$6
  local p_hidden_dims=$7
  local n_epochs=$8
  local batch_size=$9
  local lr=${10}

  # Construct the full long string with substitutions
  hyperparams="\
  \"{\\\"d_model\\\":${d_model},\\\"d_ff\\\":${d_ff},\\\"factor\\\":${factor}}\" \
  \"{\\\"d_model\\\":${d_model},\\\"d_ff\\\":${d_ff},\\\"patch_len\\\":${patch_len},\\\"stride\\\":${stride}}\" \
  \"{\\\"p_hidden_layers\\\":${p_hidden_layers},\\\"p_hidden_dims\\\":${p_hidden_dims},\\\"d_model\\\":${d_model},\\\"d_ff\\\":${d_ff},\\\"factor\\\":${factor}}\" \
  \"{\\\"d_model\\\":${d_model},\\\"d_ff\\\":${d_ff},\\\"factor\\\":${factor}}\" \
  \"{\\\"d_model\\\":${d_model},\\\"d_ff\\\":${d_ff},\\\"factor\\\":${factor}}\" \
  \"{\\\"d_model\\\":${d_model},\\\"d_ff\\\":${d_ff},\\\"factor\\\":${factor}}\" \
  \"{\\\"d_model\\\":${d_model},\\\"d_ff\\\":${d_ff},\\\"factor\\\":${factor}}\" \
  \"{\\\"d_model\\\":${d_model},\\\"d_ff\\\":${d_ff},\\\"factor\\\":${factor}}\" \
  \"{\\\"d_model\\\":${d_model},\\\"d_ff\\\":${d_ff},\\\"factor\\\":${factor}}\" \
  \"{\\\"d_model\\\":${d_model},\\\"d_ff\\\":${d_ff},\\\"factor\\\":${factor}}\" \
  \"{\\\"d_model\\\":${d_model},\\\"d_ff\\\":${d_ff},\\\"factor\\\":${factor}}\" \
  \"{\\\"d_model\\\":${d_model},\\\"d_ff\\\":${d_ff},\\\"factor\\\":${factor}}\" \
  \"{\\\"n_epochs\\\":${n_epochs},\\\"batch_size\\\":${batch_size},\\\"optimizer_kwargs\\\":{\\\"lr\\\":${lr}}}\" \
  \"{\\\"n_epochs\\\":${n_epochs},\\\"batch_size\\\":${batch_size},\\\"optimizer_kwargs\\\":{\\\"lr\\\":${lr}}}\" \
  \"{\\\"n_epochs\\\":${n_epochs},\\\"batch_size\\\":${batch_size},\\\"optimizer_kwargs\\\":{\\\"lr\\\":${lr}}}\" \
  \"{\\\"n_epochs\\\":${n_epochs},\\\"batch_size\\\":${batch_size},\\\"optimizer_kwargs\\\":{\\\"lr\\\":${lr}}}\""
  
  # Return the final string
  echo "$hyperparams"
}

# Define arrays for config paths
config_paths=(
  "fixed_forecast_config_yearly.json" 
  "fixed_forecast_config_quarterly.json" 
  "fixed_forecast_config_monthly.json" 
  "fixed_forecast_config_weekly.json" 
  "fixed_forecast_config_daily.json" 
  "fixed_forecast_config_hourly.json" 
  "fixed_forecast_config_other.json"
)

# Define arrays for save paths
save_paths=(
  "yearly" 
  "quarterly" 
  "monthly" 
  "weekly" 
  "daily" 
  "hourly" 
  "other"
)

# Define arrays for model groups
model_group_1=(
  "darts.LinearRegressionModel"
  "darts.RandomForest"
  "darts.KalmanForecaster"
  "darts.XGBModel"
)

model_group_2=(
  "darts.NaiveMean"
  "darts.NaiveSeasonal"
  "darts.NaiveDrift"
  "darts.NaiveMovingAverage"
  "darts.RNNModel"
  "darts.BlockRNNModel"
  "darts.TiDEModel"
)

model_group_3=(
  "time_series_library.Triformer"
  "time_series_library.PatchTST"
  "time_series_library.Nonstationary_Transformer"
  "time_series_library.Informer"
  "time_series_library.TimesNet"
  "time_series_library.FEDformer"
  "time_series_library.NLinear"
  "time_series_library.Linear"
  "time_series_library.DLinear"
  "time_series_library.FiLM"
  "time_series_library.MICN"
  "time_series_library.Crossformer"
  "darts.TCNModel"
  "darts.NBEATSModel"
  "darts.NHiTSModel"
  "darts.BlockRNNModel"
  "darts.StatsForecastAutoETS"
  "darts.StatsForecastAutoCES"
  "darts.StatsForecastAutoTheta"
  "darts.AutoARIMA"
)

model_group_4=(
  "time_series_library.Triformer"
  "time_series_library.PatchTST"
  "time_series_library.Nonstationary_Transformer"
  "time_series_library.Informer"
  "time_series_library.TimesNet"
  "time_series_library.FEDformer"
  "time_series_library.NLinear"
  "time_series_library.Linear"
  "time_series_library.DLinear"
  "time_series_library.FiLM"
  "time_series_library.MICN"
  "time_series_library.Crossformer"
  "darts.TCNModel"
  "darts.NBEATSModel"
  "darts.NHiTSModel"
  "darts.BlockRNNModel"
)

# Define group 3 model hyperparameters and adapters
model_hyperparams_group_3=(
  # d_model d_ff factor patch_len stride p_hidden_layers p_hidden_dims n_epochs batch_size lr
  "$(generate_hyperparams_string  8 16  3  4  2  2 "[256,256]"  5 16 1e-4)"  # Yearly
  "$(generate_hyperparams_string 32 32  3  4  2  2 "[256,256]"  5 16 1e-4)"  # Quarterly
  "$(generate_hyperparams_string 16 16  3  8  4  2 "[256,256]"  5 16 1e-4)"  # Monthly
  "$(generate_hyperparams_string 16 16  3  8  4  2 "[256,256]"  5 16 1e-4)"  # Weekly
  "$(generate_hyperparams_string  8  8  3  4  2  2 "[256,256]"  5 16 1e-4)"  # Daily
  "$(generate_hyperparams_string 16 16  3  4  2  2 "[256,256]"  5 16 1e-4)"  # Hourly
  "$(generate_hyperparams_string 32 32  3  4  2  2 "[256,256]"  5 16 1e-4)"  # Other
)

adapter_group_3=(
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
)

# Define group 4 model hyperparameters and adapters
model_hyperparams_group_4=(
  # d_model d_ff factor patch_len stride p_hidden_layers p_hidden_dims n_epochs batch_size lr
  "$(generate_hyperparams_string 16 32  3  4  2  2 "[256,256]"  5 16 1e-3)"  # Yearly
  "$(generate_hyperparams_string 64 64  3  4  2  2 "[256,256]"  5 16 1e-3)"  # Quarterly
  "$(generate_hyperparams_string 32 32  3  8  4  2 "[256,256]"  5 16 1e-3)"  # Monthly
  "$(generate_hyperparams_string 32 32  3  8  4  2 "[256,256]"  5 16 1e-3)"  # Weekly
  "$(generate_hyperparams_string 16 16  3  8  4  2 "[256,256]"  5 16 1e-3)"  # Daily
  "$(generate_hyperparams_string 32 32  3 16  8  2 "[256,256]"  5 16 1e-3)"  # Hourly
  "$(generate_hyperparams_string 64 64  3  4  2  2 "[256,256]"  5 16 1e-3)"  # Other
)

adapter_group_4=(
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
  "transformer_adapter"
)


# Function to properly quote each model name in the array
quote_model_names() {
  local models=("$@")
  local quoted_models=""
  for model in "${models[@]}"; do
    quoted_models+="\"$model\" "
  done
  echo "$quoted_models"
}

# Loop through model groups first, then iterate through configs
for group in 1 2 3 4; do
  for i in "${!config_paths[@]}"; do
    case $group in
      1)
        # Group 1
        model_names_group1=$(quote_model_names "${model_group_1[@]}")
        echo "python $script_path --config-path \"${config_paths[$i]}\" --save-path \"${save_paths[$i]}\" --gpus $gpus --num-workers $num_workers --timeout $timeout --model-name ${model_names_group1}" >> "$output_file"
        ;;
      2)
        # Group 2
        model_names_group2=$(quote_model_names "${model_group_2[@]}")
        echo "python $script_path --config-path \"${config_paths[$i]}\" --save-path \"${save_paths[$i]}\" --gpus $gpus --num-workers $num_workers --timeout $timeout --model-name ${model_names_group2}" >> "$output_file"
        ;;
      3)
        # Group 3
        model_names_group3=$(quote_model_names "${model_group_3[@]}")
        adapter_group3=$(quote_model_names "${adapter_group_3[@]}")
        echo "python $script_path --config-path \"${config_paths[$i]}\" --save-path \"${save_paths[$i]}\" --gpus $gpus --num-workers $num_workers --timeout $timeout --model-name ${model_names_group3} --model-hyper-params ${model_hyperparams_group_3[$i]} --adapter ${adapter_group3}" >> "$output_file"
        ;;
      4)
        # Group 4
        model_names_group4=$(quote_model_names "${model_group_4[@]}")
        adapter_group4=$(quote_model_names "${adapter_group_4[@]}")
        echo "python $script_path --config-path \"${config_paths[$i]}\" --save-path \"${save_paths[$i]}\" --gpus $gpus --num-workers $num_workers --timeout $timeout --model-name ${model_names_group4} --model-hyper-params ${model_hyperparams_group_4[$i]} --adapter ${adapter_group4}" >> "$output_file"
        ;;
    esac
  done
  echo "\n" >> "$output_file"
done