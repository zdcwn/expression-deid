"""Training entry points for expression, identity and anonymizer branches."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from expression_deid.config import FERPLUS_DIR, KMU_ALIGNED_DIR, PPB_ALIGNED_DIR, ROOT
from expression_deid.training.data import (
    CachedAnonymizedExpressionDataset, FERPlusExpressionDataset, KMUExpressionDataset, KMUIdentityDataset,
    PPBNeutralExpressionDataset, RAFExpressionDataset, split_ppb_people,
)
from expression_deid.models import load_checkpoint, model_class, require_torch, save_checkpoint
from expression_deid.preprocessing.normalize import load_stats
from expression_deid.runtime.pipeline import ExpressionDeidentifier


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _, _ = require_torch()
    torch.manual_seed(seed)


def _device(name: str):
    torch, _, _ = require_torch()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _loader(dataset, *, batch_size: int, shuffle: bool, workers: int):
    torch, _, _ = require_torch()
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )


def _normalise(batch, stats):
    return (batch - float(stats.mean)) / float(stats.std)


def _identity_face_only_batch(batch):
    """Apply the same tight face crop used by runtime identity diagnostics."""
    torch, _, _ = require_torch()
    values = batch.detach().cpu().numpy()[:, 0]
    cropped = np.stack([
        ExpressionDeidentifier._face_only_view(
            np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)
        ) for value in values
    ]).astype(np.float32) / 255.0
    return torch.from_numpy(cropped[:, None]).to(batch.device, dtype=batch.dtype)


def _identity_face_only_differentiable(batch):
    """Same face oval as the verifier, with gradients reaching anonymization."""
    torch, _, F = require_torch()
    from PIL import Image, ImageDraw, ImageFilter
    x0, y0, x1, y1 = ExpressionDeidentifier.FACE_DIAGNOSTIC_BOX
    crop = batch[:, :, y0:y1, x0:x1]
    mask_image = Image.new('L', (x1-x0, y1-y0), 0)
    ImageDraw.Draw(mask_image).ellipse((2,0,x1-x0-2,y1-y0), fill=255)
    mask_array = np.asarray(mask_image.filter(ImageFilter.GaussianBlur(1)), dtype=np.float32)/255
    mask = torch.as_tensor(mask_array, device=batch.device, dtype=batch.dtype)[None,None]
    fill = crop[:, :, mask_array > .5].median(2).values[:, :, None, None]
    return F.interpolate(crop*mask+fill*(1-mask), (100,100), mode='bilinear', align_corners=False)


def _accuracy(logits, labels):
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def _expression_preservation_losses(original_logits, anonymized_logits, labels, temperature: float = 2.0):
    """Preserve the original expression prediction while using labels weakly.

    KMU and RAF have a domain gap, so forcing the anonymized image directly to
    the KMU filename label can improve accuracy while changing the classifier's
    original decision.  Distillation from the original frame is the primary
    objective; the ground-truth label is only a small auxiliary signal.
    """
    torch, _, F = require_torch()
    if original_logits is None:
        zero = anonymized_logits.sum()*0
        return zero, zero, F.cross_entropy(anonymized_logits, labels, label_smoothing=.03)
    with torch.no_grad():
        teacher_probabilities = F.softmax(original_logits / temperature, dim=1)
        teacher_labels = original_logits.argmax(dim=1)
    consistency = F.kl_div(
        F.log_softmax(anonymized_logits / temperature, dim=1),
        teacher_probabilities,
        reduction="batchmean",
    ) * (temperature ** 2)
    teacher_label_loss = F.cross_entropy(anonymized_logits, teacher_labels)
    label_loss = F.cross_entropy(anonymized_logits, labels)
    return consistency, teacher_label_loss, label_loss


def _total_variation(image):
    """Anisotropic TV penalty for reducing high-frequency stripe artifacts."""
    torch, _, _ = require_torch()
    horizontal = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).mean()
    vertical = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean()
    return 0.5 * (horizontal + vertical)


def _identity_split_indices(dataset, validation_fraction: float = 0.2):
    """Tail frames within each existing person/sequence; not unseen people."""
    from expression_deid.training.data import split_kmu_sequence_tails
    train,valid=split_kmu_sequence_tails(dataset.items,validation_fraction)
    if not train or not valid:
        raise RuntimeError("KMU 身份划分失败：每个表情序列至少需要两帧")
    return train,valid


def train_expression(
    *, output: str = "models/training/expression.pt", epochs: int = 20, batch_size: int = 128,
    lr: float = 1e-3, workers: int = 0, seed: int = 0, device: str = "auto",
) -> dict:
    torch, nn, _ = require_torch()
    ExpressionNet = model_class("ExpressionNet")
    _seed_everything(seed)
    target = _device(device)
    train_set = RAFExpressionDataset("train", augment=True)
    valid_set = RAFExpressionDataset("valid", augment=False, stats=train_set.stats)
    train_loader = _loader(train_set, batch_size=batch_size, shuffle=True, workers=workers)
    valid_loader = _loader(valid_set, batch_size=batch_size, shuffle=False, workers=workers)
    model = ExpressionNet().to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best = {"accuracy": -1.0, "epoch": 0}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_acc = seen = 0
        for values, labels, _paths in train_loader:
            x = values.to(target)
            y = labels.to(target, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            count = y.numel()
            total_loss += float(loss.item()) * count
            total_acc += _accuracy(logits.detach(), y) * count
            seen += count
        valid_loss, valid_acc = _evaluate_classifier(model, valid_loader, target, criterion)
        row = {"epoch": epoch, "train_loss": total_loss / seen, "train_accuracy": total_acc / seen,
               "valid_loss": valid_loss, "valid_accuracy": valid_acc}
        history.append(row)
        if valid_acc >= best["accuracy"]:
            best = {"accuracy": valid_acc, "epoch": epoch}
            save_checkpoint(model, output, kind="expression", epoch=epoch, metrics=row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return {"best": best, "history": history, "output": str(output), "device": str(target)}


def _evaluate_classifier(model, loader, device, criterion):
    torch, _, _ = require_torch()
    model.eval()
    total_loss = correct = seen = 0
    with torch.no_grad():
        for values, labels, _paths in loader:
            x = values.to(device)
            y = labels.to(device, dtype=torch.long)
            logits = model(x)
            count = y.numel()
            total_loss += float(criterion(logits, y).item()) * count
            correct += int((logits.argmax(1) == y).sum().item())
            seen += count
    return total_loss / seen, correct / seen


def train_identity(
    *, output: str = "models/training/identity.pt", epochs: int = 30, batch_size: int = 64,
    lr: float = 1e-3, workers: int = 0, seed: int = 0, device: str = "auto",
    root=KMU_ALIGNED_DIR,
) -> dict:
    torch, nn, _ = require_torch()
    IdentityNet = model_class("IdentityNet")
    _seed_everything(seed)
    target = _device(device)
    stats = load_stats()
    dataset = KMUIdentityDataset(Path(root), include_review=False)
    train_indices, valid_indices = _identity_split_indices(dataset)
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    valid_subset = torch.utils.data.Subset(dataset, valid_indices)
    train_loader = _loader(train_subset, batch_size=batch_size, shuffle=True, workers=workers)
    valid_loader = _loader(valid_subset, batch_size=batch_size, shuffle=False, workers=workers)
    model = IdentityNet(len(dataset.person_to_id)).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best = {"accuracy": -1.0, "epoch": 0}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = correct = seen = 0
        for values, labels, _expr, _paths in train_loader:
            values = _identity_face_only_batch(values.to(target))
            x = _normalise(values, stats)
            y = labels.to(target, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            count = y.numel()
            total_loss += float(loss.item()) * count
            correct += int((logits.argmax(1) == y).sum().item())
            seen += count
        valid_loss, valid_acc = _evaluate_identity(model, valid_loader, target, stats, criterion)
        row = {"epoch": epoch, "train_loss": total_loss / seen, "train_accuracy": correct / seen,
               "valid_loss": valid_loss, "valid_accuracy": valid_acc,
               "num_identities": len(dataset.person_to_id),
               "identity_region": "face_only_ellipse",
               "person_to_id": dict(dataset.person_to_id)}
        history.append(row)
        if valid_acc >= best["accuracy"]:
            best = {"accuracy": valid_acc, "epoch": epoch}
            save_checkpoint(model, output, kind="identity", epoch=epoch, metrics=row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return {"best": best, "history": history, "output": str(output), "device": str(target)}


def _evaluate_identity(model, loader, device, stats, criterion):
    torch, _, _ = require_torch()
    model.eval()
    total_loss = correct = seen = 0
    with torch.no_grad():
        for values, labels, _expr, _paths in loader:
            values = _identity_face_only_batch(values.to(device))
            x = _normalise(values, stats)
            y = labels.to(device, dtype=torch.long)
            logits = model(x)
            count = y.numel()
            total_loss += float(criterion(logits, y).item()) * count
            correct += int((logits.argmax(1) == y).sum().item())
            seen += count
    return total_loss / seen, correct / seen


def train_anonymizer(
    *, identity_checkpoint: str, expression_checkpoint: str,
    output: str = "models/training/anonymizer.pt", epochs: int = 30, batch_size: int = 64,
    lr: float = 2e-4, workers: int = 0, seed: int = 0, device: str = "auto",
    root=KMU_ALIGNED_DIR, pixel_weight: float = 5.0, expression_weight: float = 1.0,
    identity_weight: float = 1.0, identity_cosine_weight: float = 0.75,
    identity_margin_weight: float = 1.0, identity_target_cosine: float = 0.30,
    smoothness_weight: float = 0.1, consistency_weight: float = 0.0,
    teacher_label_weight: float = 0.0, label_weight: float = 1.0,
    temperature: float = 2.0, use_source_teacher: bool = False,
) -> dict:
    if not use_source_teacher and (consistency_weight != 0 or teacher_label_weight != 0):
        raise ValueError('源图表情教师已禁用；consistency_weight 和 teacher_label_weight 必须为 0')
    torch, _, F = require_torch()
    AnonymizerUNet = model_class("AnonymizerUNet")
    _seed_everything(seed)
    target = _device(device)
    stats = load_stats()
    dataset = KMUIdentityDataset(Path(root), include_review=False)
    train_indices, valid_indices = _identity_split_indices(dataset)
    loader = _loader(torch.utils.data.Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, workers=workers)
    valid_loader = _loader(torch.utils.data.Subset(dataset, valid_indices), batch_size=batch_size, shuffle=False, workers=workers)
    identity, identity_metadata = load_checkpoint(identity_checkpoint, device=target)
    expression, expression_metadata = load_checkpoint(expression_checkpoint, device=target)
    if identity_metadata.get('kind') != 'identity' or expression_metadata.get('kind') != 'expression':
        raise ValueError('匿名化训练需要 identity 和 expression 类型的检查点')
    identity.eval(); expression.eval()
    for parameter in identity.parameters(): parameter.requires_grad_(False)
    for parameter in expression.parameters(): parameter.requires_grad_(False)
    model = AnonymizerUNet().to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    history = []
    best_validation_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0; seen = 0; component_totals = {}
        for values, _identity, expression_labels, _paths in loader:
            original = values.to(target)
            y_expression = expression_labels.to(target, dtype=torch.long)
            anonymized = model(original)
            original_n = _normalise(original, stats)
            anonymized_n = _normalise(anonymized, stats)
            original_logits = None
            if use_source_teacher:
                with torch.no_grad(): original_logits = expression(original_n)
            expression_logits = expression(anonymized_n)
            consistency_loss, teacher_label_loss, label_loss = _expression_preservation_losses(
                original_logits, expression_logits, y_expression, temperature=temperature
            )
            expression_loss = (
                consistency_weight * consistency_loss
                + teacher_label_weight * teacher_label_loss
                + label_weight * label_loss
            )
            # Maximising identity entropy discourages a confident identity result.
            identity_anonymous = _normalise(_identity_face_only_differentiable(anonymized), stats)
            identity_probs = F.softmax(identity(identity_anonymous), dim=1)
            entropy = -(identity_probs * torch.log(identity_probs.clamp_min(1e-8))).sum(1).mean()
            with torch.no_grad():
                original_embedding = identity.embedding(_normalise(_identity_face_only_differentiable(original), stats))
            anonymized_embedding = identity.embedding(identity_anonymous)
            pair_cosines = F.cosine_similarity(
                original_embedding, anonymized_embedding, dim=1
            )
            identity_cosine = pair_cosines.mean()
            identity_margin = F.relu(pair_cosines - float(identity_target_cosine)).mean()
            identity_loss = (
                -entropy
                + identity_cosine_weight * identity_cosine
                + identity_margin_weight * identity_margin
            )
            pixel_loss = F.l1_loss(anonymized, original)
            smoothness_loss = _total_variation(anonymized)
            loss = (
                pixel_weight * pixel_loss
                + expression_weight * expression_loss
                + identity_weight * identity_loss
                + smoothness_weight * smoothness_loss
            )
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            count = original.shape[0]; total += float(loss.item()) * count; seen += count
            for name, value in {'pixel_loss': pixel_loss, 'expression_loss': expression_loss,
                                'expression_consistency_loss': consistency_loss,
                                'expression_teacher_label_loss': teacher_label_loss, 'expression_label_loss': label_loss,
                                'identity_entropy': entropy, 'identity_cosine': identity_cosine,
                                'identity_margin': identity_margin, 'smoothness_loss': smoothness_loss}.items():
                component_totals[name] = component_totals.get(name, 0.0) + float(value.detach()) * count
        row = {"epoch": epoch, "loss": total / seen, **{name: value / seen for name, value in component_totals.items()},
               "identity_region": "face_only_ellipse", "expression_input_source": "legacy_source_teacher" if use_source_teacher else "anonymized_only",
               "samples": seen, "train_samples": len(train_indices), "valid_samples": len(valid_indices),
               "validation_scope": "Aligned UNet output before runtime compositing; not end-to-end privacy acceptance"}
        model.eval()
        valid_loss = valid_cosine = 0.0
        valid_seen = valid_correct = 0
        with torch.inference_mode():
            for values, _identity, labels, _paths in valid_loader:
                source = values.to(target)
                anonymous = model(source)
                logits = expression(_normalise(anonymous, stats))
                probs = F.softmax(identity(_normalise(_identity_face_only_differentiable(anonymous), stats)), dim=1)
                entropy = -(probs * probs.clamp_min(1e-8).log()).sum(1).mean()
                a = identity.embedding(_normalise(_identity_face_only_differentiable(source), stats))
                b = identity.embedding(_normalise(_identity_face_only_differentiable(anonymous), stats))
                cosines = F.cosine_similarity(a, b, dim=1)
                objective = (pixel_weight * F.l1_loss(anonymous, source)
                    + expression_weight * label_weight * F.cross_entropy(logits, labels.to(target))
                    + identity_weight * (-entropy + identity_cosine_weight * cosines.mean()
                        + identity_margin_weight * F.relu(cosines-identity_target_cosine).mean())
                    + smoothness_weight * _total_variation(anonymous))
                count = len(source)
                valid_loss += float(objective) * count
                valid_cosine += float(cosines.sum())
                valid_correct += int((logits.argmax(1)==labels.to(target)).sum())
                valid_seen += count
        row.update(valid_loss=valid_loss/valid_seen, valid_identity_cosine=valid_cosine/valid_seen,
                   valid_expression_accuracy=valid_correct/valid_seen)
        history.append(row)
        if row['valid_loss'] < best_validation_loss:
            best_validation_loss = row['valid_loss']
            save_checkpoint(model, output, kind="anonymizer", epoch=epoch, metrics=row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return {"history": history, "output": str(output), "device": str(target)}


def train_expression_kmu(
    *, base_checkpoint: str = "models/training/expression.pt",
    output: str = "models/training/expression_kmu.pt", epochs: int = 20,
    batch_size: int = 64, lr: float = 2e-4, workers: int = 0,
    seed: int = 0, device: str = "auto", root=KMU_ALIGNED_DIR,
) -> dict:
    """Adapt the RAF expression model to the camera/domain used by KMU.

    The split is sequence-aware: the final frames of each person/expression
    sequence are validation data. Participants and neighboring frames still
    overlap; this is a tail-frame regression, not an unseen-sequence benchmark.
    """
    torch, nn, _ = require_torch()
    _seed_everything(seed)
    target = _device(device)
    stats = load_stats()
    dataset = KMUIdentityDataset(Path(root), include_review=False)
    train_indices, valid_indices = _identity_split_indices(dataset)
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    valid_subset = torch.utils.data.Subset(dataset, valid_indices)
    train_loader = _loader(train_subset, batch_size=batch_size, shuffle=True, workers=workers)
    valid_loader = _loader(valid_subset, batch_size=batch_size, shuffle=False, workers=workers)
    model, base_meta = load_checkpoint(base_checkpoint, device=target)
    if base_meta.get("kind") != "expression":
        raise ValueError("base_checkpoint 不是 expression 模型检查点")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # KMU has no neutral class. Keep that class available for RAF inputs while
    # adapting the six observed classes with a mild label smoothing penalty.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    best = {"accuracy": -1.0, "epoch": 0}
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); total_loss = total_acc = seen = 0
        for values, _identity, labels, _paths in train_loader:
            x = _normalise(values.to(target), stats)
            y = labels.to(target, dtype=torch.long)
            # Small photometric jitter improves transfer to changing cabin light.
            if target.type == "cuda":
                x = x + torch.empty((x.shape[0], 1, 1, 1), device=target).uniform_(-0.04, 0.04)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            count = y.numel(); total_loss += float(loss.item()) * count
            total_acc += _accuracy(logits.detach(), y) * count; seen += count
        valid_loss, valid_acc = _evaluate_identity_expression(model, valid_loader, target, stats, criterion)
        row = {"epoch": epoch, "train_loss": total_loss / seen,
               "train_accuracy": total_acc / seen, "valid_loss": valid_loss,
               "valid_accuracy": valid_acc, "base_checkpoint": str(base_checkpoint),
               "num_identities": len(dataset.person_to_id)}
        history.append(row)
        if valid_acc >= best["accuracy"]:
            best = {"accuracy": valid_acc, "epoch": epoch}
            save_checkpoint(model, output, kind="expression", epoch=epoch, metrics=row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return {"best": best, "history": history, "output": str(output), "device": str(target)}


def _evaluate_identity_expression(model, loader, device, stats, criterion):
    torch, _, _ = require_torch()
    model.eval(); total_loss = correct = seen = 0
    with torch.no_grad():
        for values, _identity, labels, _paths in loader:
            x = _normalise(values.to(device), stats)
            y = labels.to(device, dtype=torch.long)
            logits = model(x); count = y.numel()
            total_loss += float(criterion(logits, y).item()) * count
            correct += int((logits.argmax(1) == y).sum().item()); seen += count
    return total_loss / seen, correct / seen


def train_expression_joint(
    *, base_checkpoint: str = "models/training/expression_raf.pt",
    output: str = "models/training/expression_joint.pt", epochs: int = 12,
    batch_size: int = 128, lr: float = 1e-4, workers: int = 0,
    seed: int = 0, device: str = "auto", root=KMU_ALIGNED_DIR,
    kmu_fraction: float = 0.25, ferplus_root=FERPLUS_DIR,
    ferplus_fraction: float = 0.35, ferplus_min_votes: int = 5,
    ppb_root=PPB_ALIGNED_DIR, ppb_fraction: float = 0.08,
    ppb_include_review: bool = False, component_augment: bool = True,
    gan_root=ROOT / "datasets" / "kmu-gan-aligned", gan_fraction: float = 0.0,
) -> dict:
    """Train one expression model across RAF, FER+, and the KMU camera domain.

    The RAF validation set protects general expression recognition, while the
    KMU tail-frame validation checks the cabin domain with overlapping people. FER+
    contributes high-agreement labels from the official FER+ annotations.
    A weighted sampler prevents any one domain from drowning out the others.
    """
    torch, nn, _ = require_torch()
    if not 0.0 < kmu_fraction < 1.0 or not 0.0 <= ferplus_fraction < 1.0 or not 0.0 <= ppb_fraction < 1.0 or not 0.0 <= gan_fraction < 1.0:
        raise ValueError("kmu_fraction、ferplus_fraction、ppb_fraction 和 gan_fraction 必须在 [0, 1) 内")
    if kmu_fraction + ferplus_fraction + ppb_fraction + gan_fraction >= 1.0:
        raise ValueError("kmu_fraction + ferplus_fraction + ppb_fraction + gan_fraction 必须小于 1")
    _seed_everything(seed)
    target = _device(device)
    stats = load_stats()
    raf_train = RAFExpressionDataset("train", augment=True, component_augment=component_augment, stats=stats)
    raf_valid = RAFExpressionDataset("valid", augment=False, stats=stats)
    kmu_identity = KMUIdentityDataset(Path(root), include_review=False)
    kmu_train_indices, kmu_valid_indices = _identity_split_indices(kmu_identity)
    kmu_train = torch.utils.data.Subset(
        KMUExpressionDataset(Path(root), include_review=False, augment=True, component_augment=component_augment, stats=stats),
        kmu_train_indices,
    )
    kmu_valid = torch.utils.data.Subset(
        KMUExpressionDataset(Path(root), include_review=False, augment=False, stats=stats),
        kmu_valid_indices,
    )
    domain_sets = [raf_train]
    domain_fractions = [1.0 - kmu_fraction - ferplus_fraction - ppb_fraction - gan_fraction]
    domain_sets.append(kmu_train)
    domain_fractions.append(kmu_fraction)
    gan_path = Path(gan_root)
    if gan_fraction:
        if not (gan_path / "manifest.jsonl").is_file():
            raise FileNotFoundError(f"GAN 匿名化训练集不存在: {gan_path}; 先运行 scripts/data/prepare_gan_expression.py")
        gan_train = CachedAnonymizedExpressionDataset(gan_path, split="train", augment=True, stats=stats)
        held_sources = {kmu_identity.items[index][2]['source'] for index in kmu_valid_indices}
        # Every view of a validation source must stay out of training, even
        # when an older generated manifest used a different split convention.
        gan_train.records = [row for row in gan_train.records if row['source'] not in held_sources]
        if not gan_train.records: raise ValueError('排除验证源图后 GAN 训练集为空')
        domain_sets.append(gan_train)
        domain_fractions.append(gan_fraction)
    ferplus_train = ferplus_valid = None
    ferplus_path = Path(ferplus_root)
    image_csv = ferplus_path / "original" / "fer2013.csv"
    if not image_csv.is_file():
        image_csv = ferplus_path / "fer2013.csv"
    if (image_csv.is_file() and (ferplus_path / "fer2013new.csv").is_file()):
        ferplus_min_votes = max(1, min(10, int(ferplus_min_votes)))
        ferplus_train = FERPlusExpressionDataset(
            "train", root=ferplus_path, min_votes=ferplus_min_votes,
            augment=True, component_augment=component_augment, stats=stats,
        )
        ferplus_valid = FERPlusExpressionDataset(
            "valid", root=ferplus_path, min_votes=ferplus_min_votes,
            augment=False, stats=stats,
        )
        domain_sets.insert(1, ferplus_train)
        domain_fractions.insert(1, ferplus_fraction)
    elif ferplus_fraction:
        raise FileNotFoundError(
            f"FER+ 数据不存在: {ferplus_path}; 请准备原始 fer2013.csv 和官方 fer2013new.csv"
        )
    ppb_train = ppb_valid = ppb_test = None
    ppb_people = None
    ppb_path = Path(ppb_root)
    if ppb_fraction:
        ppb_all = PPBNeutralExpressionDataset(ppb_path, include_review=ppb_include_review,
                                              augment=False, stats=stats)
        ppb_train_indices, ppb_valid_indices, ppb_test_indices, ppb_people = split_ppb_people(
            ppb_all, validation_fraction=0.2, test_fraction=0.2, seed=seed,
        )
        ppb_aug = PPBNeutralExpressionDataset(ppb_path, include_review=ppb_include_review,
                                             augment=True, component_augment=component_augment, stats=stats)
        ppb_eval = PPBNeutralExpressionDataset(ppb_path, include_review=ppb_include_review,
                                               augment=False, stats=stats)
        ppb_train = torch.utils.data.Subset(ppb_aug, ppb_train_indices)
        ppb_valid = torch.utils.data.Subset(ppb_eval, ppb_valid_indices)
        ppb_test = torch.utils.data.Subset(ppb_eval, ppb_test_indices)
        domain_sets.append(ppb_train)
        domain_fractions.append(ppb_fraction)
    elif (ppb_path / "manifest.jsonl").is_file():
        # Keep the default training command backwards compatible, but make an
        # explicit fraction the only way PPB can affect the weights.
        ppb_people = {"enabled": False}
    combined = torch.utils.data.ConcatDataset(domain_sets)
    weights = torch.tensor(
        [fraction / len(dataset) for dataset, fraction in zip(domain_sets, domain_fractions)
         for _ in range(len(dataset))],
        dtype=torch.double,
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(combined), replacement=True,
    )
    loader = torch.utils.data.DataLoader(
        combined, batch_size=batch_size, sampler=sampler, num_workers=workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )
    raf_valid_loader = _loader(raf_valid, batch_size=batch_size, shuffle=False, workers=workers)
    kmu_valid_loader = _loader(kmu_valid, batch_size=batch_size, shuffle=False, workers=workers)
    ferplus_valid_loader = (
        _loader(ferplus_valid, batch_size=batch_size, shuffle=False, workers=workers)
        if ferplus_valid is not None else None
    )
    ppb_valid_loader = (
        _loader(ppb_valid, batch_size=batch_size, shuffle=False, workers=workers)
        if ppb_valid is not None else None
    )
    model, base_meta = load_checkpoint(base_checkpoint, device=target)
    if base_meta.get("kind") != "expression":
        raise ValueError("base_checkpoint 不是 expression 模型检查点")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    best = {"score": -1.0, "epoch": 0}
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); total_loss = total_acc = seen = 0
        for values, labels, _paths in loader:
            x = values.to(target)
            y = labels.to(target, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            count = y.numel(); total_loss += float(loss.item()) * count
            total_acc += _accuracy(logits.detach(), y) * count; seen += count
        raf_acc = _evaluate_expression_accuracy(model, raf_valid_loader, target)
        kmu_acc = _evaluate_expression_accuracy(model, kmu_valid_loader, target)
        ferplus_acc = (
            _evaluate_expression_accuracy(model, ferplus_valid_loader, target)
            if ferplus_valid_loader is not None else None
        )
        ppb_acc = (
            _evaluate_expression_accuracy(model, ppb_valid_loader, target)
            if ppb_valid_loader is not None else None
        )
        scores = [raf_acc, kmu_acc] + ([ferplus_acc] if ferplus_acc is not None else []) + ([ppb_acc] if ppb_acc is not None else [])
        score = float(np.mean(scores))
        row = {
            "epoch": epoch, "train_loss": total_loss / seen,
            "train_accuracy": total_acc / seen,
            "raf_valid_accuracy": raf_acc, "kmu_valid_accuracy": kmu_acc,
            "ferplus_valid_accuracy": ferplus_acc,
            "ppb_valid_accuracy": ppb_acc,
            "balanced_validation_score": score,
            "base_checkpoint": str(base_checkpoint),
            "kmu_fraction": kmu_fraction,
            "ferplus_fraction": ferplus_fraction if ferplus_valid is not None else 0.0,
            "ferplus_min_votes": ferplus_min_votes if ferplus_valid is not None else None,
            "ppb_fraction": ppb_fraction if ppb_valid is not None else 0.0,
            "ppb_people": ppb_people,
            "component_augment": bool(component_augment),
            "gan_fraction": gan_fraction,
            "gan_root": str(gan_path) if gan_fraction else None,
        }
        history.append(row)
        if score >= best["score"]:
            best = {"score": score, "epoch": epoch}
            save_checkpoint(model, output, kind="expression", epoch=epoch, metrics=row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    best_model, payload = load_checkpoint(output, device=target)
    calibration_loaders = [raf_valid_loader, kmu_valid_loader]
    if ferplus_valid_loader is not None:
        calibration_loaders.append(ferplus_valid_loader)
    if ppb_valid_loader is not None:
        calibration_loaders.append(ppb_valid_loader)
    calibration_temperature = _fit_expression_temperature(
        best_model, calibration_loaders, target
    )
    payload.setdefault("metrics", {})["calibration_temperature"] = calibration_temperature
    torch.save(payload, output)
    print(json.dumps({"calibration_temperature": calibration_temperature}, ensure_ascii=False), flush=True)
    ppb_test_accuracy = None
    if ppb_test is not None:
        ppb_test_accuracy = _evaluate_expression_accuracy(
            best_model, _loader(ppb_test, batch_size=batch_size, shuffle=False, workers=workers), target
        )
    return {"best": best, "history": history, "output": str(output),
            "ppb_test_accuracy": ppb_test_accuracy,
            "device": str(target), "calibration_temperature": calibration_temperature}


def _evaluate_expression_accuracy(model, loader, device):
    torch, _, _ = require_torch()
    model.eval(); correct = seen = 0
    with torch.no_grad():
        for values, labels, _paths in loader:
            y = labels.to(device, dtype=torch.long)
            logits = model(values.to(device))
            correct += int((logits.argmax(1) == y).sum().item())
            seen += int(y.numel())
    return correct / seen


def _fit_expression_temperature(model, loaders, device):
    """Fit one scalar temperature on held-out logits for honest confidence."""
    torch, nn, _ = require_torch()
    model.eval(); logits_all = []; labels_all = []
    with torch.no_grad():
        for loader in loaders:
            for values, labels, _paths in loader:
                logits_all.append(model(values.to(device)).detach())
                labels_all.append(labels.to(device, dtype=torch.long))
    logits = torch.cat(logits_all); labels = torch.cat(labels_all)
    log_temperature = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.05, 20.0).detach().cpu().item())


def build_parser(subparsers):
    p = subparsers.add_parser("train-expression", help="训练 RAF 表情分类器")
    p.add_argument("--output", default="models/training/expression.pt"); p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="auto")
    p.set_defaults(func=lambda a: train_expression(output=a.output, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, workers=a.workers, seed=a.seed, device=a.device))
    p = subparsers.add_parser("train-identity", help="训练 KMU 身份验收模型")
    p.add_argument("--output", default="models/training/identity.pt"); p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="auto"); p.add_argument("--root", default=str(KMU_ALIGNED_DIR))
    p.set_defaults(func=lambda a: train_identity(output=a.output, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, workers=a.workers, seed=a.seed, device=a.device, root=a.root))
    p = subparsers.add_parser("train-expression-kmu", help="用 KMU 座舱图像微调表情模型")
    p.add_argument("--base-checkpoint", default="models/training/expression.pt")
    p.add_argument("--output", default="models/training/expression_kmu.pt"); p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=0); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="auto"); p.add_argument("--root", default=str(KMU_ALIGNED_DIR))
    p.set_defaults(func=lambda a: train_expression_kmu(base_checkpoint=a.base_checkpoint, output=a.output, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, workers=a.workers, seed=a.seed, device=a.device, root=a.root))
    p = subparsers.add_parser("train-expression-joint", help="混合 RAF、FER+ 与 KMU 训练表情模型，避免域适配遗忘")
    p.add_argument("--base-checkpoint", default="models/training/expression_raf.pt"); p.add_argument("--output", default="models/training/expression_joint.pt"); p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="auto"); p.add_argument("--root", default=str(KMU_ALIGNED_DIR)); p.add_argument("--kmu-fraction", type=float, default=0.25)
    p.add_argument("--ferplus-root", default=str(FERPLUS_DIR)); p.add_argument("--ferplus-fraction", type=float, default=0.35); p.add_argument("--ferplus-min-votes", type=int, default=5)
    p.add_argument("--ppb-root", default=str(PPB_ALIGNED_DIR)); p.add_argument("--ppb-fraction", type=float, default=0.08); p.add_argument("--ppb-include-review", action="store_true")
    p.add_argument("--gan-root", default=str(ROOT / "datasets" / "kmu-gan-aligned")); p.add_argument("--gan-fraction", type=float, default=0.0, help="加入离线 GAN 匿名图做域适配；需要先运行 prepare_gan_expression")
    p.add_argument("--component-augment", dest="component_augment", action="store_true", default=True, help="训练时模拟眼睛/鼻子/嘴部局部匿名化扰动（默认开启）")
    p.add_argument("--no-component-augment", dest="component_augment", action="store_false", help="关闭局部匿名化扰动增强")
    p.set_defaults(func=lambda a: train_expression_joint(base_checkpoint=a.base_checkpoint, output=a.output, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, workers=a.workers, seed=a.seed, device=a.device, root=a.root, kmu_fraction=a.kmu_fraction, ferplus_root=a.ferplus_root, ferplus_fraction=a.ferplus_fraction, ferplus_min_votes=a.ferplus_min_votes, ppb_root=a.ppb_root, ppb_fraction=a.ppb_fraction, ppb_include_review=a.ppb_include_review, component_augment=a.component_augment, gan_root=a.gan_root, gan_fraction=a.gan_fraction))
    p = subparsers.add_parser("train-anonymizer", help="训练人脸匿名化网络")
    p.add_argument("--identity-checkpoint", required=True); p.add_argument("--expression-checkpoint", required=True); p.add_argument("--output", default="models/training/anonymizer.pt"); p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--workers", type=int, default=0); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="auto"); p.add_argument("--root", default=str(KMU_ALIGNED_DIR)); p.add_argument("--pixel-weight", type=float, default=2.5); p.add_argument("--expression-weight", type=float, default=1.0); p.add_argument("--identity-weight", type=float, default=1.0); p.add_argument("--identity-cosine-weight", type=float, default=0.75); p.add_argument("--identity-margin-weight", type=float, default=1.0); p.add_argument("--identity-target-cosine", type=float, default=0.30); p.add_argument("--smoothness-weight", type=float, default=0.1); p.add_argument("--consistency-weight", type=float, default=0.0); p.add_argument("--teacher-label-weight", type=float, default=0.0); p.add_argument("--label-weight", type=float, default=1.0); p.add_argument("--temperature", type=float, default=2.0)
    p.set_defaults(func=lambda a: train_anonymizer(identity_checkpoint=a.identity_checkpoint, expression_checkpoint=a.expression_checkpoint, output=a.output, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, workers=a.workers, seed=a.seed, device=a.device, root=a.root, pixel_weight=a.pixel_weight, expression_weight=a.expression_weight, identity_weight=a.identity_weight, identity_cosine_weight=a.identity_cosine_weight, identity_margin_weight=a.identity_margin_weight, identity_target_cosine=a.identity_target_cosine, smoothness_weight=a.smoothness_weight, consistency_weight=a.consistency_weight, teacher_label_weight=a.teacher_label_weight, label_weight=a.label_weight, temperature=a.temperature))
