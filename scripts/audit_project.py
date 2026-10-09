"""Reproducible local inventory, dataset and model checks (no source FER input)."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from expression_deid.models import load_checkpoint
from expression_deid.normalize import load_stats, compute_raf_train_stats, iter_raf


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def classification_metrics(logits, labels, temperature=1.0):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    probabilities = (logits / temperature).softmax(1)
    confidence, predicted = probabilities.max(1)
    correct = predicted == labels
    ece = 0.0
    for low in np.linspace(0, .9, 10):
        selected = (confidence > low) & (confidence <= low + .1 + 1e-7)
        if selected.any():
            ece += float(selected.float().mean() * (correct[selected].float().mean() - confidence[selected].mean()).abs())
    matrix = np.zeros((7, 7), dtype=int)
    for truth, prediction in zip(labels.tolist(), predicted.tolist()):
        matrix[truth, prediction] += 1
    return dict(samples=len(labels), accuracy=float(correct.float().mean()),
                nll=float(torch.nn.functional.cross_entropy(logits / temperature, labels)),
                ece_10_bins=ece, mean_confidence=float(confidence.mean()),
                confusion=matrix.tolist(), by_class={str(i): dict(samples=int(matrix[i].sum()),
                accuracy=float(matrix[i, i] / matrix[i].sum())) for i in range(7) if matrix[i].sum()})


def source_inventory():
    result = []
    for folder in ('expression_deid', 'scripts', 'tests'):
        for path in sorted((ROOT / folder).glob('*.py')):
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            result.append(dict(path=str(path.relative_to(ROOT)), sha256=sha256(path), lines=len(path.read_text(encoding='utf-8-sig').splitlines()),
                               definitions=[n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))], syntax='pass'))
    return result


def datasets():
    reports = []
    for root in sorted((ROOT / 'datasets').rglob('*')):
        if not root.is_dir() or not (root / 'manifest.jsonl').is_file(): continue
        rows = [json.loads(line) for line in (root / 'manifest.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        issues, dimensions, statuses = [], Counter(), Counter()
        seen_outputs, partitions, people = {}, {}, {}
        checked = 0
        for row in rows:
            statuses[row.get('status', 'unlabelled_gallery')] += 1
            key = row.get('output', row.get('path'))
            if not key or row.get('status') == 'fail': continue
            path = Path(key)
            if path.is_absolute(): path = root / path.name  # relocated legacy gallery
            else: path = root / path
            if not path.is_relative_to(root): issues.append(f'output escapes root: {key}'); continue
            if key in seen_outputs: issues.append(f'duplicate output: {key}')
            seen_outputs[key] = True
            if not path.is_file(): issues.append(f'missing: {path}'); continue
            try:
                with Image.open(path) as image:
                    image.load(); dimensions[f'{image.size}/{image.mode}'] += 1
            except Exception as exc: issues.append(f'cannot decode {key}: {exc}')
            checked += 1
            if 'label' in row and not 0 <= int(row['label']) < 7: issues.append(f'invalid label: {key}')
            if 'split' in row:
                source = (row.get('domain', ''), row.get('source', ''))
                partitions.setdefault(source, set()).add(row['split'])
                if row.get('person'): people.setdefault(row['person'], set()).add(row['split'])
        for key, values in partitions.items():
            if len(values) > 1: issues.append(f'source split leakage: {key}')
        reports.append(dict(root=str(root.relative_to(ROOT)), records=len(rows), checked_images=checked,
                            statuses=dict(statuses), dimensions=dict(dimensions), issues=issues,
                            people_spanning_partitions=[key for key, values in people.items() if len(values) > 1]))
        print(f'dataset {root.name}: {checked} images, {len(issues)} issues', flush=True)
    raf = {}
    hashes = {}
    overlap = []
    for split in ('train', 'valid'):
        counts, invalid = Counter(), []
        for path, label in iter_raf(split):
            counts[label] += 1
            try:
                with Image.open(path) as image: image.load()
            except Exception as exc: invalid.append(f'{path}: {exc}')
            digest = sha256(path)
            if digest in hashes and hashes[digest][0] != split: overlap.append((str(hashes[digest][1].relative_to(ROOT)), str(path.relative_to(ROOT))))
            hashes.setdefault(digest, (split, path))
        raf[split] = dict(classes=dict(counts), decode_errors=invalid)
    stats, count = compute_raf_train_stats()
    raf.update(train_valid_exact_duplicates=overlap, recomputed_stats=stats.to_json(), images=count, stored_stats=load_stats().to_json())
    return dict(manifests=reports, raf=raf)


def models(cache, device):
    cache = cache.resolve()
    data = np.load(cache)
    indices = np.flatnonzero(np.isin(data['splits'], ['valid', 'test']))
    values = data['images'][indices]
    stats = load_stats()
    reports = []
    for path in sorted((ROOT / 'models').glob('*.pt')):
        started = time.perf_counter()
        row = dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha256(path))
        try:
            model, payload = load_checkpoint(path, device=device)
            row.update(kind=payload['kind'], epoch=payload.get('epoch'), model_config=payload.get('model_config'),
                       finite_weights=all(torch.isfinite(value).all().item() for value in model.state_dict().values()),
                       metadata=payload.get('metrics', {}))
            x = torch.from_numpy(values[:8].copy()).float()[:, None].to(device) / 255
            if payload['kind'] == 'identity':
                from expression_deid.inference import ExpressionDeidentifier
                crops = np.stack([ExpressionDeidentifier._face_only_view(image) for image in values[:8]])
                x = torch.from_numpy(crops).float()[:, None].to(device) / 255
            if payload['kind'] != 'anonymizer': x = (x - stats.mean) / stats.std
            with torch.inference_mode(): output = model(x)
            row.update(output_shape=list(output.shape), finite_output=bool(torch.isfinite(output).all()),
                       output_range=[float(output.min()), float(output.max())])
            if payload['kind'] == 'expression':
                logits = []
                with torch.inference_mode():
                    for batch in np.array_split(values, max(1, int(np.ceil(len(values) / 32)))):
                        batch = torch.from_numpy(batch.copy()).float()[:, None].to(device) / 255
                        logits.append(model((batch - stats.mean) / stats.std).cpu())
                logits = torch.cat(logits)
                temp = float(payload.get('metrics', {}).get('calibration_temperature', 1))
                row['fast_anonymous_domain'] = {}
                for split in ('valid', 'test'):
                    for domain in ('raf', 'kmu', 'ppb'):
                        mask = (data['splits'][indices] == split) & (data['domains'][indices] == domain)
                        if mask.any(): row['fast_anonymous_domain'][f'{domain}/{split}'] = classification_metrics(logits[mask], data['labels'][indices][mask], temp)
            if device == 'cuda':
                fp32 = output.float().cpu()
                model.half()
                with torch.inference_mode(): fp16 = model(x.half()).float().cpu()
                row['cuda_fp16'] = dict(finite=bool(torch.isfinite(fp16).all()), max_abs_diff=float((fp16-fp32).abs().max()),
                                       argmax_agreement=float((fp16.argmax(1) == fp32.argmax(1)).float().mean()) if fp16.ndim == 2 else None)
            row['status'] = 'pass' if row['finite_weights'] and row['finite_output'] and row.get('cuda_fp16', {}).get('finite', True) else 'fail'
        except Exception as exc: row.update(status='fail', error=f'{type(exc).__name__}: {exc}')
        row['seconds'] = time.perf_counter()-started
        reports.append(row)
        print(f"model {path.name}: {row['status']}", flush=True)
    ledger_path = ROOT / 'models/anonymization_assets.json'
    ledger = json.loads(ledger_path.read_text())
    assets = []
    for path in sorted((ROOT / 'models').iterdir()):
        if path.suffix not in ('.onnx', '.task', '.ckpt'): continue
        digest = sha256(path)
        assets.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=digest,
                           ledger_match=digest == ledger[path.name]['sha256'] if path.name in ledger else None))
    return dict(checkpoints=reports, external_assets=assets, cache=str(cache.relative_to(ROOT)) if cache.is_relative_to(ROOT) else str(cache),
                domain_note='All expression metrics here use the common fast anonymous cache. GAN classifiers need their own generated domain to judge quality.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'outputs/full_audit_20261008'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--cache', default=str(ROOT / 'outputs/realtime_training/anonymous_cache.npz'))
    parser.add_argument('--section', choices=('all', 'sources', 'datasets', 'models'), default='all')
    args = parser.parse_args(); torch.set_num_threads(1)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for section, callback in (('sources', source_inventory), ('datasets', datasets), ('models', lambda: models(Path(args.cache), args.device))):
        if args.section in ('all', section):
            result = callback()
            (out / (section + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            if section == 'models' and any(row['status'] != 'pass' for row in result['checkpoints']):
                raise SystemExit(1)


if __name__ == '__main__': main()
