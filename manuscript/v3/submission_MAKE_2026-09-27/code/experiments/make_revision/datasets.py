"""Audited native SUSHI source; raw files and provider IDs remain local."""
import hashlib
import io
import os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from arrowflow.ranking import inverse_positions
from .evaluation import dataset_fingerprint

SUSHI_ID = 'sushi_childhood_east_west_v2'
ARCHIVE_SHA256 = '4f8bbf3acd6f796cb3d0add6c73c394664982c57d3b48e75e92014e5278558a8'
SOURCE_HASHES = {
    'sushi3-2016/sushi3a.5000.10.order': 'c56da325c0f58f9a38bafc650ed699b48dd2fb9f6ad64ba61ffc288d1ae4cc24',
    'sushi3-2016/sushi3.udata': '592d345613b6fbc1d7d126775eb5fb5628619e855c292166250f09651716eca2',
    'sushi3-2016/README-en.txt': 'deb639156f15fd4ba5659c62bdac752f99e232db3b93b4d3c9f04d86c3a11d56',
}


def parse_sushi_files(order_text, user_text, expected_rows=5000):
    """Provider row correspondence; column 7 is childhood East/West."""
    lines = order_text.splitlines()
    if not lines or lines[0].split() != ['10', '1']:
        raise ValueError('Unexpected SUSHI full-ranking header')
    try:
        ranking_rows = np.asarray([[int(x) for x in row.split()] for row in lines[1:]], dtype=np.int64)
        users = np.asarray([[int(x) for x in row.split()] for row in user_text.splitlines()], dtype=np.int64)
    except (ValueError, TypeError) as exc:
        raise ValueError('Malformed SUSHI row') from exc
    if ranking_rows.shape != (expected_rows, 12) or users.shape != (expected_rows, 11):
        raise ValueError('SUSHI row counts/widths do not align')
    if not np.all(ranking_rows[:, :2] == [0, 10]):
        raise ValueError('SUSHI rankings must be complete strict orders')
    orders = ranking_rows[:, 2:]
    inverse_positions(orders)
    user_ids, labels = users[:, 0], users[:, 6]
    if len(np.unique(user_ids)) != expected_rows:
        raise ValueError('Provider user IDs must be unique')
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError('Expected binary childhood East/West labels')
    return orders, labels, user_ids


def load_sushi(archive_path=None):
    """Read an explicit/env archive; verify exact audited bytes before parsing.

    Acquisition: https://www.kamishima.net/sushi/. Research use is permitted;
    redistribution requires provider permission. Never silently download/substitute.
    """
    location = archive_path or os.environ.get('ARROWFLOW_SUSHI_ARCHIVE')
    if not location or not Path(location).is_file():
        raise FileNotFoundError('Set ARROWFLOW_SUSHI_ARCHIVE or pass the audited SUSHI archive path')
    archive = Path(location).read_bytes()
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('SUSHI archive hash does not match audited source')
    with ZipFile(io.BytesIO(archive)) as source:
        contents = {}
        for name, expected in SOURCE_HASHES.items():
            if source.namelist().count(name) != 1:
                raise ValueError(f'Missing/duplicate audited SUSHI member: {name}')
            contents[name] = source.read(name)
            if hashlib.sha256(contents[name]).hexdigest() != expected:
                raise ValueError(f'SUSHI member hash mismatch: {name}')
    order_text = contents['sushi3-2016/sushi3a.5000.10.order'].decode('ascii')
    user_text = contents['sushi3-2016/sushi3.udata'].decode('ascii')
    X, y, user_ids = parse_sushi_files(order_text, user_text)
    if np.bincount(y).tolist() != [3258, 1742]:
        raise ValueError('SUSHI corrected target class-count mismatch')
    users = np.asarray([[int(x) for x in row.split()] for row in user_text.splitlines()])
    disagreements = int(np.sum((users[:, 4] >= 24) != y))
    if disagreements != 105:
        raise ValueError('SUSHI legacy-label audit mismatch')
    names = [f'item_at_position_{i}' for i in range(10)]
    label_map = ['childhood_East', 'childhood_West']
    manifest = {'dataset_id': SUSHI_ID, 'source': 'https://www.kamishima.net/sushi/',
                'archive_sha256': ARCHIVE_SHA256, 'source_file_sha256': SOURCE_HASHES,
                'shape': [5000, 10], 'class_counts': [3258, 1742], 'feature_names': names,
                'label_map': label_map, 'label_column_one_based': 7,
                'legacy_label_disagreements': disagreements,
                'sample_order': 'documented corresponding source rows; sample_id 0..4999',
                'provider_user_ids': user_ids.tolist(),
                'local_only': 'Do not redistribute raw SUSHI data or provider user IDs',
                'dataset_hash': dataset_fingerprint(X, y, names, label_map)}
    return X, y, manifest
