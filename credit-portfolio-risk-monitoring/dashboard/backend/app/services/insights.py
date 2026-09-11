"""Simple insights helpers for the Overview API endpoints.

Functions:
- load_data(period=None, data_url=...): load dataframe from the repo's data_splits (remote via GitHub API or raw) and return a pd.DataFrame
- summary(df): compute total lent, repaid, default rate, interest earned
- aggregates(df): group by loan purpose and compute lent/repaid/interest
- loan_purposes(df): return sorted unique loan purpose list

This module is intentionally lightweight and defensive: it tolerates missing columns
and falls back to 0/empty lists when an operation is not possible.
"""
from __future__ import annotations

import os
import io
import json
import requests
import pandas as pd
import functools
from typing import Optional, List, Dict

# Default remote data_splits location used by the app frontend
DEFAULT_DATA_URL = (
	"https://github.com/josephine-amponsah/credit-scoring-nlp-dl"
	"/tree/main/credit-portfolio-risk-monitoring/notebooks/data_splits"
)


def _read_bytes_to_df(content: bytes, name: str) -> pd.DataFrame:
	try:
		if name.endswith('.parquet'):
			return pd.read_parquet(io.BytesIO(content))
		# CSV or gzipped CSV
		return pd.read_csv(io.BytesIO(content), parse_dates=['issue_d'], low_memory=False)
	except Exception:
		# Last resort: try CSV without parse_dates
		try:
			return pd.read_csv(io.BytesIO(content), low_memory=False)
		except Exception:
			raise


def _normalize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
	for c in df.select_dtypes(include=['object']).columns:
		df[c] = df[c].apply(lambda v: None if pd.isna(v) else (v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else str(v)))
	return df


def _list_remote_datafiles(data_url: str, timeout: int = 10) -> List[str]:
	if not (data_url.startswith('https://github.com/') and '/tree/' in data_url):
		return []
	owner_repo, rest = data_url.replace('https://github.com/', '').split('/tree/', 1)
	branch, _, path = rest.partition('/')
	api_url = f"https://api.github.com/repos/{owner_repo}/contents/{path}"
	resp = requests.get(api_url, params={'ref': branch}, timeout=timeout)
	if resp.status_code != 200:
		return []
	items = resp.json()
	return [it['name'] for it in items if it.get('name', '').startswith('data_')]


@functools.lru_cache(maxsize=32)
def load_data(period: Optional[str] = None, data_url: str = DEFAULT_DATA_URL, timeout: int = 10) -> pd.DataFrame:
	"""Load data for a given quarter period (e.g. '2018Q4').

	If `period` is None the latest available period is selected from the
	remote listing. The function returns an empty DataFrame on failure.
	"""
	try:
		# Support local development: prefer local copies if available
		local_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'data_splits')
		local_dir = os.path.normpath(local_dir)
		if os.path.isdir(local_dir):
			names = [n for n in os.listdir(local_dir) if n.startswith('data_')]
			if names:
				if not period:
					period = sorted([n.split('data_')[1].split('.')[0] for n in names])[-1]
				for ext in ('.parquet', '.csv.gz', '.csv'):
					p = os.path.join(local_dir, f"data_{period}{ext}")
					if os.path.exists(p):
						try:
							if p.endswith('.parquet'):
								df = pd.read_parquet(p)
							else:
								df = pd.read_csv(p, parse_dates=['issue_d'], low_memory=False)
							return _normalize_object_columns(df)
						except Exception:
							continue

		# Remote: query GitHub contents API and fetch raw file
		owner_repo, rest = data_url.replace('https://github.com/', '').split('/tree/', 1)
		branch, _, path = rest.partition('/')
		api_url = f"https://api.github.com/repos/{owner_repo}/contents/{path}"
		resp = requests.get(api_url, params={'ref': branch}, timeout=timeout)
		if resp.status_code != 200:
			return pd.DataFrame()
		items = resp.json()
		names = [it['name'] for it in items if it.get('name', '').startswith('data_')]
		if not names:
			return pd.DataFrame()
		if not period:
			period = sorted([n.split('data_')[1].split('.')[0] for n in names])[-1]

		raw_base = f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{path}".rstrip('/')
		for name in (f"data_{period}.parquet", f"data_{period}.csv.gz", f"data_{period}.csv"):
			r = requests.get(f"{raw_base}/{name}", timeout=timeout)
			if r.status_code != 200:
				continue
			try:
				df = _read_bytes_to_df(r.content, name)
				return _normalize_object_columns(df)
			except Exception:
				continue
		return pd.DataFrame()
	except Exception:
		return pd.DataFrame()


def _ensure_loan_purpose_column(df: pd.DataFrame) -> pd.DataFrame:
	if 'loan_purpose' in df.columns:
		return df
	# common source column is 'purpose'
	if 'purpose' in df.columns:
		df = df.copy()
		df['loan_purpose'] = df['purpose']
		return df
	return df


def summary(df: pd.DataFrame) -> Dict[str, float]:
	"""Compute top-level summary metrics from a DataFrame.

	Returns dict with keys: total_lent, repaid, default_rate, interest_earned
	"""
	if df is None or df.empty:
		return {'total_lent': 0.0, 'repaid': 0.0, 'default_rate': 0.0, 'interest_earned': 0.0}
	df = _ensure_loan_purpose_column(df)
	total_lent = float(df.get('funded_amnt', pd.Series(dtype='float')).sum()) if 'funded_amnt' in df.columns else 0.0
	# repaid principal
	repaid = float(df.get('total_rec_prncp', pd.Series(dtype='float')).sum()) if 'total_rec_prncp' in df.columns else 0.0
	# interest earned
	interest = float(df.get('total_rec_int', pd.Series(dtype='float')).sum()) if 'total_rec_int' in df.columns else 0.0
	# Default rate: use notebook mapping to targets and exclude 'current' loans (label 2)
	default_rate = 0.0
	if 'loan_status' in df.columns:
		s = df['loan_status'].astype(str).fillna('')

		def _map_target(x: str) -> int:
			# follow notebook mapping: Fully Paid -> 0, Charged Off/Default/Late(31-120) ->1, else ->2
			if x == 'Fully Paid':
				return 0
			if x in ('Charged Off', 'Default', 'Late (31-120 days)'):
				return 1
			return 2

		try:
			targets = s.map(_map_target)
			non_current = targets != 2
			denom = int(non_current.sum())
			if denom > 0:
				default_rate = float(((targets == 1) & non_current).sum()) / denom
			else:
				default_rate = 0.0
		except Exception:
			default_rate = 0.0
	return {'total_lent': total_lent, 'repaid': repaid, 'default_rate': default_rate, 'interest_earned': interest}


def aggregates(df: pd.DataFrame) -> List[Dict[str, float]]:
	"""Return aggregates grouped by loan purpose with fields lent/repaid/interest.

	Output: list of dicts: { 'loan_purpose': str, 'lent': float, 'repaid': float, 'interest': float }
	"""
	if df is None or df.empty:
		return []
	df = _ensure_loan_purpose_column(df)
	group_col = 'loan_purpose' if 'loan_purpose' in df.columns else None
	if not group_col:
		return []
	# numeric columns with fallbacks
	lent_col = 'funded_amnt' if 'funded_amnt' in df.columns else None
	repaid_col = 'total_rec_prncp' if 'total_rec_prncp' in df.columns else None
	interest_col = 'total_rec_int' if 'total_rec_int' in df.columns else None

	agg = df.groupby(group_col).agg({
		lent_col: 'sum' if lent_col else pd.NamedAgg(column=group_col, aggfunc=lambda x: 0),
		repaid_col: 'sum' if repaid_col else pd.NamedAgg(column=group_col, aggfunc=lambda x: 0),
		interest_col: 'sum' if interest_col else pd.NamedAgg(column=group_col, aggfunc=lambda x: 0),
	})
	# normalize column names
	agg = agg.rename(columns={
		lent_col: 'lent' if lent_col else 'lent',
		repaid_col: 'repaid' if repaid_col else 'repaid',
		interest_col: 'interest' if interest_col else 'interest',
	})
	agg = agg.reset_index()
	# ensure numeric values
	rows = []
	for _, r in agg.iterrows():
		rows.append({'loan_purpose': r[group_col], 'lent': float(r.get('lent', 0.0) or 0.0), 'repaid': float(r.get('repaid', 0.0) or 0.0), 'interest': float(r.get('interest', 0.0) or 0.0)})
	# sort by lent desc
	rows = sorted(rows, key=lambda x: x['lent'], reverse=True)
	return rows


def loan_purposes(df: pd.DataFrame) -> List[str]:
	if df is None or df.empty:
		return []
	df = _ensure_loan_purpose_column(df)
	if 'loan_purpose' not in df.columns:
		return []
	opts = sorted(pd.Series(df['loan_purpose'].dropna().unique()).astype(str).tolist())
	return opts


def list_periods(data_url: str = DEFAULT_DATA_URL, timeout: int = 10) -> List[str]:
	"""Return available quarter period identifiers (e.g. '2018Q4').

	Prefers local copies under `notebooks/data_splits` when present, otherwise
	lists files from the configured GitHub `data_url`.
	"""
	# Local first
	try:
		local_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'data_splits')
		local_dir = os.path.normpath(local_dir)
		if os.path.isdir(local_dir):
			names = [n for n in os.listdir(local_dir) if n.startswith('data_')]
			if names:
				periods = sorted({n.split('data_')[1].split('.')[0] for n in names})
				return periods
	except Exception:
		pass

	# Remote listing
	try:
		names = _list_remote_datafiles(data_url, timeout=timeout)
		if not names:
			return []
		periods = sorted({n.split('data_')[1].split('.')[0] for n in names})
		return periods
	except Exception:
		return []

