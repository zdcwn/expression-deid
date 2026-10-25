"""Adapt the classifier to the actual fast anonymous video input domain.

Source images are used only by anonymization. Both supervised training and
consistency training receive anonymous pixels. PPB participants and KMU
sequence tails follow the existing checkpoint's validation boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from expression_deid.preprocessing.align import similarity_matrix, template_for, warp_gray
from expression_deid.ui.opencv_demo import build_pipeline
from expression_deid.config import ROOT, PPB_ALIGNED_DIR
from expression_deid.training.data import KMUIdentityDataset
from expression_deid.models import load_checkpoint, save_checkpoint
from expression_deid.preprocessing.normalize import iter_raf, load_gray, load_stats
from expression_deid.training.train import _identity_split_indices


def cache_signature(base, per_class, seed, validation_per_class, profile):
    from expression_deid.config import KMU_ALIGNED_DIR, STATS_PATH
    def digest(path):
        with Path(path).open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()
    _, metadata = load_checkpoint(base)
    people = metadata.get('metrics', {}).get('ppb_people')
    if not people or not all(people.get(split) for split in ('train', 'valid', 'test')):
        raise ValueError('基准模型缺少完整 PPB 参与者划分')
    if sum(map(len, people.values())) != len(set(person for values in people.values() for person in values)):
        raise ValueError('PPB 参与者在多个划分中重复')
    anonymizer = next(path for path in (ROOT/'models/anonymization/anonymizer_adversarial_v1.pt', ROOT/'models/training/anonymizer.pt') if path.is_file())
    roots = (KMU_ALIGNED_DIR, PPB_ALIGNED_DIR, ROOT/'datasets/raf-db')
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for root in roots for path in sorted(root.rglob('*'))
                     if path.is_file() and path.suffix in ('.jpg', '.png', '.jsonl')}
    return dict(schema=3, per_class=per_class, seed=seed, validation_per_class=validation_per_class,
                ppb_people=people, machine_privacy=profile, anonymizer_sha256=digest(anonymizer), stats_sha256=digest(STATS_PATH),
                code={name:digest(ROOT/'expression_deid'/name) for name in ('inference.py','align.py','normalize.py','feature_geometry.py','data.py')},
                data_sha256=hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest())


def validate_cache(data, signature):
    if 'signature' not in data or json.loads(str(data['signature'])) != signature:
        raise ValueError('匿名缓存与当前代码、数据或匿名化参数不一致；使用 --rebuild-cache 重建')
    if (str(data['privacy_source']) != 'anonymized_only' or data['images'].dtype != np.uint8
            or data['images'].shape[1:] != (100, 100) or not np.isin(data['labels'], range(7)).all()):
        raise ValueError('匿名缓存输入格式无效')


def prepare(cache: Path, device: str, base: Path, per_class: int, seed: int, signature=None):
    rng = np.random.default_rng(seed)
    pipeline = build_pipeline(device, use_gan=False)
    if signature is not None: pipeline.machine_privacy = signature['machine_privacy']
    try:
        _, metadata = load_checkpoint(base)
        items = []
        for split in ('train', 'valid'):
            raf = list(iter_raf(split))
            for label in range(7):
                selected = [item for item in raf if item[1] == label]
                if split == 'train':
                    order = rng.permutation(len(selected))[:per_class]
                    selected = [selected[i] for i in order]
                elif signature is None or signature['validation_per_class']:
                    selected = selected[:50 if signature is None else signature['validation_per_class']]
                items.extend((path, label, 'raf', split) for path, label in selected)
        kmu = KMUIdentityDataset()
        train, valid = _identity_split_indices(kmu)
        for split, indices in (('train', train), ('valid', valid)):
            items.extend((kmu.items[i][0], kmu.items[i][1], 'kmu', split) for i in indices)
        people = metadata['metrics']['ppb_people']
        for line in (PPB_ALIGNED_DIR / 'manifest.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('status') != 'ok':
                continue
            split = next((key for key in ('train', 'valid', 'test') if row['person'] in people[key]), None)
            if split:
                items.append((PPB_ALIGNED_DIR / row['output'], 6, 'ppb', split))
        images, labels, domains, partitions, sources = [], [], [], [], []
        for i, (path, label, domain, split) in enumerate(items):
            gray = load_gray(path)
            size = gray.shape[0]
            if gray.shape != (size, size):
                raise ValueError(f'Expected aligned square face: {path}')
            landmarks = template_for(size)
            matrix = similarity_matrix(landmarks, template_for(256))
            aligned = warp_gray(gray, landmarks)[0]
            pipeline._runtime_donor = pipeline._select_component_donor(pipeline._resize(aligned))
            edited = pipeline._machine_runtime_candidate(aligned, variant=3)
            # Match video inference, including the feature mask and the second
            # alignment. Training directly on the uncomposited candidate is a
            # different domain and can overstate runtime accuracy.
            anonymous = pipeline._composite_component_edit(gray, edited, (0, 0, size, size), matrix, mask_mode=pipeline.machine_privacy.get('mask_mode', 'features'))
            classifier_input = pipeline._resize(warp_gray(anonymous, landmarks)[0])
            images.append(classifier_input.copy()); labels.append(label)
            domains.append(domain); partitions.append(split); sources.append(str(path.relative_to(ROOT)))
            if (i + 1) % 200 == 0:
                print(f'anonymous cache {i + 1}/{len(items)}', flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, images=np.stack(images), labels=labels,
                            domains=domains, splits=partitions, sources=sources,
                            privacy_source='anonymized_only', ppb_people=json.dumps(people), signature=json.dumps(signature, sort_keys=True))
    finally:
        pipeline.close()


def augment(x):
    count = len(x)
    theta = torch.eye(2, 3, device=x.device).repeat(count, 1, 1)
    theta[:, :, 2] = torch.empty(count, 2, device=x.device).uniform_(-0.04, 0.04)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode='border', align_corners=False)
    gain = torch.empty(count, 1, 1, 1, device=x.device).uniform_(0.92, 1.08)
    offset = torch.empty_like(gain).uniform_(-0.04, 0.04)
    return (x * gain + offset).clamp(0, 1)


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device
    base, output, cache = Path(args.base).resolve(), Path(args.output).resolve(), Path(args.cache).resolve()
    pending = output.with_name(output.name + '.pending')
    pending.unlink(missing_ok=True)
    if base == output:
        raise ValueError('Keep the original checkpoint; use a new output path')
    if args.epochs < 1 or args.batch_size < 1 or args.per_class < 1 or args.lr <= 0:
        raise ValueError('训练轮数、批大小、样本数和学习率必须为正数')
    profile = json.loads(Path(args.machine_profile).read_text(encoding='utf-8')) if args.machine_profile else {}
    signature = cache_signature(base, args.per_class, args.seed, args.validation_per_class, profile)
    if not cache.is_file() or args.rebuild_cache:
        prepare(cache, device, base, args.per_class, args.seed, signature)
    data = np.load(cache)
    validate_cache(data, signature)
    if args.prepare_only:
        return
    if str(data['privacy_source']) != 'anonymized_only':
        raise ValueError('Training requires an anonymized-only cache')
    partitions = {}
    for source, split in zip(data['sources'], data['splits']):
        partitions.setdefault(str(source), set()).add(str(split))
    if any(len(splits) != 1 for splits in partitions.values()):
        raise ValueError('A source image occurs in more than one partition')
    images = torch.from_numpy(data['images'].copy()).float()[:, None] / 255
    targets = torch.from_numpy(data['labels'].copy()).long()
    training = np.flatnonzero(data['splits'] == 'train')
    validation = np.flatnonzero(data['splits'] == 'valid')
    for split, indices in (('train',training),('valid',validation)):
        if any(not (data['domains'][indices] == domain).any() for domain in ('raf','kmu','ppb')):
            raise ValueError(f'{split} 缺少 RAF/KMU/PPB 域样本')
    stats = load_stats()
    model, metadata = load_checkpoint(base, device=device)
    weights = np.zeros(len(training))
    for domain, fraction in (('raf', .45), ('kmu', .40), ('ppb', .15)):
        domain_mask = data['domains'][training] == domain
        classes = np.unique(data['labels'][training][domain_mask])
        for label in classes:
            mask = domain_mask & (data['labels'][training] == label)
            weights[mask] = fraction / len(classes) / mask.sum()
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(training), replacement=True)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images[training], targets[training]),
                                       batch_size=args.batch_size, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')

    def evaluate():
        model.eval(); logits = []
        with torch.inference_mode():
            for batch in images[validation].split(args.batch_size):
                x = (batch.to(device) - stats.mean) / stats.std
                logits.append(model(x).float().cpu())
        logits = torch.cat(logits)
        predicted = logits.argmax(1).numpy()
        result = {domain: float((predicted[data['domains'][validation] == domain] == data['labels'][validation][data['domains'][validation] == domain]).mean()) for domain in ('raf', 'kmu', 'ppb')}
        result['score'] = float(np.mean(list(result.values())))
        return result, logits

    baseline, _ = evaluate()
    best = baseline['score']
    selected = False
    history = []
    print(json.dumps({'baseline': baseline, 'train_samples': len(training), 'valid_samples': len(validation)}), flush=True)
    for epoch in range(args.epochs):
        model.train(); loss_sum = count = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == 'cuda'):
                views = torch.cat([augment(x), augment(x)])
                a, b = model((views - stats.mean) / stats.std).chunk(2)
                ce = (F.cross_entropy(a, y, label_smoothing=.06) + F.cross_entropy(b, y, label_smoothing=.06)) / 2
                pa, pb = a.float().softmax(1), b.float().softmax(1)
                mean = (pa + pb) / 2
                consistency = (F.kl_div(mean.clamp_min(1e-7).log(), pa, reduction='batchmean') + F.kl_div(mean.clamp_min(1e-7).log(), pb, reduction='batchmean')) / 2
                loss = ce + .3 * consistency
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.detach()) * len(y); count += len(y)
        metrics, _ = evaluate()
        row = {'epoch': epoch + 1, 'loss': loss_sum / count, **metrics}
        history.append(row); print(json.dumps(row), flush=True)
        # Do not adopt a neutral improvement at the expense of the other
        # domains. This gate compares the exact same anonymous validation set.
        if metrics['score'] > best and all(metrics[domain] >= baseline[domain] - .01 for domain in ('kmu','raf','ppb')):
            best = metrics['score']
            selected = True
            save_checkpoint(model, pending, kind='expression', epoch=epoch + 1,
                            metrics={**row, 'base_checkpoint': base.name, 'ppb_people': metadata['metrics']['ppb_people'],
                                     'expression_input_source': 'anonymized_only', 'temporal_consistency_weight': .3,
                                     'machine_privacy': profile, 'anonymous_cache_signature': signature,
                                     'calibration_temperature': 1.0})
    (output.parent / (output.stem + '.training.json')).write_text(json.dumps({'baseline':baseline,'history':history,
        'selected':selected,'cache':str(cache),'machine_privacy':profile},indent=2),encoding='utf-8')
    if not selected:
        pending.unlink(missing_ok=True)
        raise RuntimeError('No candidate passed the validation gate; the runtime checkpoint was not changed')
    model, payload = load_checkpoint(pending)
    model.to(device); _, logits = evaluate()
    labels = targets[validation]
    log_temperature = torch.zeros((), requires_grad=True)
    calibration = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=50)
    def closure():
        calibration.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp().clamp(.5, 5), labels)
        loss.backward(); return loss
    calibration.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(.5, 5))
    payload['metrics'].update(calibration_temperature=temperature,
                              calibration_source='anonymous_validation_frames', baseline=baseline)
    save_checkpoint(model, pending, kind='expression', epoch=payload['epoch'], metrics=payload['metrics'])
    pending.replace(output)
    (output.parent / (output.stem + '.training.json')).write_text(json.dumps({'baseline': baseline, 'history': history,
                     'temperature': temperature, 'cache': str(cache)}, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'temperature': temperature}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', default=str(ROOT / 'models/expression/expression_machine_selector_texture.pt'))
    p.add_argument('--output', default=str(ROOT / 'models/training/expression_realtime_candidate.pt'))
    p.add_argument('--cache', default=str(ROOT / 'outputs/realtime_training/anonymous_cache.npz'))
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--per-class', type=int, default=150)
    p.add_argument('--validation-per-class', type=int, default=50, help='0 使用完整 RAF 验证集')
    p.add_argument('--machine-profile', help='局部匿名化参数 JSON；默认保留旧输入域')
    p.add_argument('--rebuild-cache', action='store_true')
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--prepare-only', action='store_true')
    train(p.parse_args())


if __name__ == '__main__':
    main()
