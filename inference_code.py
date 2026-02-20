import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuantileTransformerRegressor(nn.Module):
	def __init__(
		self,
		input_dim,
		num_quantiles=999,
		d_model=128,
		nhead=8,
		num_layers=3,
		dim_feedforward=512,
		dropout=0.1,
	):
		super().__init__()
		self.num_quantiles = num_quantiles

		self.input_embedding = nn.Linear(input_dim, d_model)
		self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True,
		)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		self.fc_out = nn.Sequential(
			nn.Linear(d_model, dim_feedforward // 2),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(dim_feedforward // 4, num_quantiles),
		)

	def forward(self, x):
		x = self.input_embedding(x)
		x = x.unsqueeze(1) + self.pos_encoder
		x = self.transformer_encoder(x)
		return self.fc_out(x.squeeze(1))


def load_dataset_and_scaler(csv_path, target_col="사정율", exclude_cols=None):
	if exclude_cols is None:
		exclude_cols = {target_col, "낙찰가"}

	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

	df = pd.read_csv(csv_path)
	if target_col not in df.columns:
		raise ValueError(f"타겟 컬럼 '{target_col}' 이(가) 데이터에 없습니다.")

	selected_features = [col for col in df.columns if col not in exclude_cols]
	df = df[selected_features + [target_col]].fillna(0)

	X = df[selected_features].values

	scaler = StandardScaler().fit(X)
	return df, selected_features, scaler


def build_input_vector(input_dict, feature_names, fill_missing=False, fill_value=0.0):
	missing = [f for f in feature_names if f not in input_dict]
	extra = [k for k in input_dict if k not in feature_names]

	if missing and not fill_missing:
		raise ValueError(
			"입력 피처가 누락되었습니다: " + ", ".join(missing)
		)

	vector = [input_dict.get(f, fill_value) for f in feature_names]
	X = np.array([vector], dtype=np.float32)
	return X, missing, extra


def load_model(model_path, input_dim, num_quantiles):
	model = QuantileTransformerRegressor(
		input_dim=input_dim,
		num_quantiles=num_quantiles,
		d_model=512,
		nhead=8,
		num_layers=2,
		dim_feedforward=2048,
		dropout=0.1,
	).to(device)

	if not os.path.exists(model_path):
		raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

	checkpoint = torch.load(model_path, map_location=device)
	model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
	model.eval()
	return model


def predict_quantiles(model, X):
	with torch.no_grad():
		X_tensor = torch.FloatTensor(X).to(device)
		return model(X_tensor).cpu().numpy()[0]


def predict_quantiles_batch(model, X, batch_size=2048):
	model.eval()
	outputs = []
	with torch.no_grad():
		for i in range(0, len(X), batch_size):
			batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
			pred = model(batch).cpu().numpy()
			outputs.append(pred)
	return np.vstack(outputs)


def quantile_intervals(pred_values, quantile_levels, top_k=None):
	pred_values = np.asarray(pred_values, dtype=np.float64)
	if pred_values.ndim != 1:
		pred_values = pred_values.flatten()

	quantile_levels = np.asarray(quantile_levels, dtype=np.float64)
	if len(pred_values) != len(quantile_levels):
		raise ValueError("pred_values와 quantile_levels 길이가 일치해야 합니다.")

	order = np.argsort(quantile_levels)
	q = quantile_levels[order]
	x = pred_values[order]

	# 비단조 예측값 보정: 값 기준으로 정렬하고 분위수는 오름차순 유지
	if np.any(np.diff(x) < 0):
		value_order = np.argsort(x)
		x = x[value_order]
		q = np.sort(q)

	kde = gaussian_kde(pred_values)
	bin_info = []
	for i in range(len(x) - 1):
		lower = float(x[i])
		upper = float(x[i + 1])
		prob = float(kde.integrate_box_1d(lower, upper))
		if prob <= 0:
			continue
		center = (lower + upper) / 2
		bin_info.append(
			{
				"lower": lower,
				"upper": upper,
				"center": float(center),
				"probability": prob,
				"probability_percent": prob * 100,
			}
		)

	total_prob = sum(b["probability"] for b in bin_info)
	if total_prob > 0:
		for b in bin_info:
			b["probability"] /= total_prob
			b["probability_percent"] = b["probability"] * 100

	sorted_bins = sorted(bin_info, key=lambda x: x["probability"], reverse=True)
	if top_k is None:
		top_k = len(sorted_bins)
	return {
		"top_ranges": sorted_bins[:top_k],
		"all_ranges": sorted_bins,
		"total_bins": len(sorted_bins),
		"prediction_range": {
			"min": float(x.min()),
			"max": float(x.max()),
			"range": float(x.max() - x.min()),
		},
		"quantile_levels": q.tolist(),
	}


def top1_interval_from_quantiles(pred_values, quantile_levels):
	result = quantile_intervals(pred_values, quantile_levels=quantile_levels, top_k=1)
	if not result["top_ranges"]:
		center = float(np.mean(pred_values))
		return center, center, center, 0.0
	best = result["top_ranges"][0]
	return best["lower"], best["upper"], best["center"], best["probability"]


def infer_top_ranges(
	input_dict,
	csv_path="dataset_feature_selected.csv",
	model_path="best_model.pt",
	bin_width=None,
	top_k=3,
	fill_missing=False,
):
	"""
	추론용 함수: 입력 피처로 예측 분위수 경계 구간을 반환합니다.
	"""
	df, feature_names, scaler = load_dataset_and_scaler(csv_path)

	if input_dict is None:
		input_dict = df[feature_names].iloc[40001].to_dict()

	X_raw, missing, extra = build_input_vector(
		input_dict, feature_names, fill_missing=fill_missing
	)

	X_scaled = scaler.transform(X_raw)

	quantiles = np.linspace(0.05, 0.95, 10).tolist()
	if 0.5 not in quantiles:
		quantiles.append(0.5)
		quantiles = sorted(quantiles)

	model = load_model(
		model_path, input_dim=len(feature_names), num_quantiles=len(quantiles)
	)
	pred_quantiles = predict_quantiles(model, X_scaled)

	result = quantile_intervals(pred_quantiles, quantile_levels=quantiles, top_k=top_k)
	result["input_features"] = {
		f: float(input_dict.get(f, 0.0)) for f in feature_names
	}
	result["missing_features"] = missing
	result["extra_features"] = extra
	result["pred_quantiles"] = pred_quantiles.tolist()
	return result


def main():
	"""
	50,000개 샘플에 대해 Top-1 예측값 분포와 실제 분포 비교.
	"""
	csv_path = "dataset_feature_selected.csv"
	model_path = "./tft_22_bid_extracted/best_model.pt"
	max_samples = None

	df, feature_names, scaler = load_dataset_and_scaler(csv_path)
	if "사정율" not in df.columns:
		raise ValueError("데이터에 '사정율' 컬럼이 없습니다.")

	if max_samples is not None:
		df = df.head(max_samples)
	X = df[feature_names].values
	y_true = df["사정율"].values.astype(np.float64)

	X_scaled = scaler.transform(X)

	quantiles = np.linspace(0.05, 0.95, 10).tolist()
	if 0.5 not in quantiles:
		quantiles.append(0.5)
		quantiles = sorted(quantiles)

	model = load_model(
		model_path, input_dim=len(feature_names), num_quantiles=len(quantiles)
	)
	pred_quantiles = predict_quantiles_batch(model, X_scaled, batch_size=2048)

	top1_centers = np.zeros(len(pred_quantiles), dtype=np.float64)
	top1_lowers = np.zeros(len(pred_quantiles), dtype=np.float64)
	top1_uppers = np.zeros(len(pred_quantiles), dtype=np.float64)
	for i in tqdm(range(len(pred_quantiles)), desc="Top-1 interval", unit="sample"):
		lower, upper, center, _ = top1_interval_from_quantiles(pred_quantiles[i], quantiles)
		top1_lowers[i] = lower
		top1_uppers[i] = upper
		top1_centers[i] = center

	# Top-1 구간 커버리지
	in_interval = (y_true >= top1_lowers) & (y_true <= top1_uppers)
	coverage = float(np.mean(in_interval) * 100)
	print(f"\nTop-1 구간 내 실제값 포함 비율: {coverage:.2f}%")

	# 테스트 샘플 10개 오차 확인
	np.random.seed(42)
	sample_size = min(10, len(y_true))
	sample_idx = np.random.choice(len(y_true), size=sample_size, replace=False)

	print("\n===== 테스트 샘플 10개 Top-1 오차 =====")
	print("{:>6} | {:>12} | {:>12} | {:>12} | {:>27} | {:>11}".format(
		"idx", "actual", "top1", "error", "interval [lower, upper]", "in_range"
	))
	print("-" * 94)
	for idx in sample_idx:
		y = y_true[idx]
		pred_center = top1_centers[idx]
		lower = top1_lowers[idx]
		upper = top1_uppers[idx]
		err = pred_center - y
		in_range = lower <= y <= upper
		print("{:>6} | {:>12.6f} | {:>12.6f} | {:>12.6f} | [{:>10.6f}, {:>10.6f}] | {:>11}".format(
			idx, y, pred_center, err, lower, upper, str(in_range)
		))

	# 실제 분포 CDF + 50,000개 Top-1 오차막대
	y_sorted = np.sort(y_true)
	n = len(y_sorted)
	cdf_y = np.arange(1, n + 1) / n

	def cdf_at_array(x_arr):
		idx = np.searchsorted(y_sorted, x_arr, side="right")
		return idx / n

	cdf_centers = cdf_at_array(top1_centers)

	plt.figure(figsize=(10, 6))
	plt.plot(y_sorted, cdf_y, label="Actual CDF (사정율)", linewidth=2)
	plt.errorbar(
		top1_centers,
		cdf_centers,
		xerr=[top1_centers - top1_lowers, top1_uppers - top1_centers],
		fmt=".",
		ecolor="red",
		color="red",
		alpha=0.15,
		elinewidth=0.5,
		capsize=0,
		label="Top-1 intervals (50k)",
	)
	plt.title("Actual CDF with 50k Top-1 Error Bars")
	plt.xlabel("사정율")
	plt.ylabel("CDF")
	plt.legend()
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig("top1_vs_actual_cdf_50k.png", dpi=200)

	print("\n저장 완료: top1_vs_actual_cdf_50k.png")

	# 실제 분포 vs 테스트(Top-1) 분포 비교 (KDE)
	x_min = float(min(y_true.min(), top1_centers.min()))
	x_max = float(max(y_true.max(), top1_centers.max()))
	x_range = np.linspace(x_min, x_max, 300)

	gt_kde = gaussian_kde(y_true)
	test_kde = gaussian_kde(top1_centers)

	gt_pdf = gt_kde(x_range)
	test_pdf = test_kde(x_range)

	plt.figure(figsize=(10, 6))
	plt.plot(x_range, gt_pdf, label="Actual (사정율)", linewidth=2)
	plt.plot(x_range, test_pdf, label="Test (Top-1)", linewidth=2)
	plt.fill_between(x_range, gt_pdf, alpha=0.2)
	plt.fill_between(x_range, test_pdf, alpha=0.2)
	plt.title("Distribution: Actual vs Test (Top-1, 50k)")
	plt.xlabel("사정율")
	plt.ylabel("Density")
	plt.legend()
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig("actual_vs_test_dist_50k.png", dpi=200)

	print("저장 완료: actual_vs_test_dist_50k.png")

	# 실제 분포 vs 테스트(Top-1) 분포 CDF 비교
	gt_sorted = np.sort(y_true)
	test_sorted = np.sort(top1_centers)
	gt_cdf = np.arange(1, len(gt_sorted) + 1) / len(gt_sorted)
	test_cdf = np.arange(1, len(test_sorted) + 1) / len(test_sorted)

	plt.figure(figsize=(10, 6))
	plt.plot(gt_sorted, gt_cdf, label="Actual CDF (사정율)", linewidth=2)
	plt.plot(test_sorted, test_cdf, label="Test CDF (Top-1)", linewidth=2)
	plt.title("CDF: Actual vs Test (Top-1, 50k)")
	plt.xlabel("사정율")
	plt.ylabel("CDF")
	plt.legend()
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig("actual_vs_test_cdf_50k.png", dpi=200)

	print("저장 완료: actual_vs_test_cdf_50k.png")
	print(f"Actual mean={y_true.mean():.6f}, std={y_true.std():.6f}")
	print(f"Top1 mean={top1_centers.mean():.6f}, std={top1_centers.std():.6f}")


if __name__ == "__main__":
	main()

