"""The required CelebA-HQ / FER-2013 / Pins data contract and preparation."""
from __future__ import annotations

from collections import Counter, defaultdict
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image

from expression_deid.config import ROOT
from expression_deid.preprocessing.normalize import to_gray

PRIMARY = {'anonymization': 'celebahq', 'expression': 'fer2013', 'identity': 'pins'}
AUXILIARY_FOLDERS = {'kmu': 'datasets/kmu-auxiliary/prepared', 'ppb-emo': 'datasets/ppb-emo/prepared',
                     'raf':'datasets/raf-auxiliary/prepared', 'affectnet':'datasets/affectnet-auxiliary/prepared'}
FER_TO_PROJECT = {0: 5, 1: 2, 2: 1, 3: 3, 4: 4, 5: 0, 6: 6}
USAGE_SPLITS = {'Training': 'train', 'PublicTest': 'valid', 'PrivateTest': 'test'}
_local = threading.local()


def load_policy(path=None):
    policy = json.loads(Path(path or ROOT/'configs/required_datasets.json').read_text(encoding='utf-8'))
    for module, required in PRIMARY.items():
        spec = policy[module]
        if spec['primary'] != required:
            raise ValueError(f'{module} must use the required primary dataset {required}')
        auxiliary = spec.get('auxiliary', [])
        if len(auxiliary) > 1 or len(set(auxiliary)) != len(auxiliary) or required in auxiliary:
            raise ValueError('Each module may use at most one different auxiliary dataset')
        if auxiliary and (module!='expression' or auxiliary[0] not in AUXILIARY_FOLDERS):
            raise ValueError('The configured auxiliary has no preparation and training implementation')
    if not .75 <= policy['minimum_primary_sampling_fraction'] <= 1:
        raise ValueError('Primary data must account for at least 75% of training draws')
    if policy['expression_input_source'] != 'anonymized_only':
        raise ValueError('Expression classification requires anonymized pixels')
    fraction=float(policy['expression'].get('primary_sampling_fraction',1))
    if not policy['minimum_primary_sampling_fraction']<=fraction<=1:
        raise ValueError('The primary sampling fraction violates the dataset policy')
    return policy


def _pixel_hash(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _dhash(image):
    small = np.asarray(Image.fromarray(image).resize((9, 8), Image.Resampling.BILINEAR))
    return int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), 'big')


def grouped_split(records, *, seed=42, per_identity=False):
    """Keep exact and near-duplicate image groups in one partition."""
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for index, row in enumerate(records):
        by_class[row['label'] if per_identity else 0].append(index)
    for indices in by_class.values():
        parent = {i: i for i in indices}
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        buckets = defaultdict(list)
        for index in indices:
            code = records[index]['dhash']
            candidates = set()
            for shift in range(4):
                candidates.update(buckets[(shift, (code >> (16*shift)) & 65535)])
            for previous in candidates:
                if (code ^ records[previous]['dhash']).bit_count() <= 3:
                    parent[find(index)] = find(previous)
            for shift in range(4):
                buckets[(shift, (code >> (16*shift)) & 65535)].append(index)
        groups = defaultdict(list)
        for index in indices: groups[find(index)].append(index)
        values = list(groups.values()); rng.shuffle(values)
        if len(values) < 3:
            raise ValueError('At least three independent image groups are needed per identity')
        valid_target = max(1, round(len(indices)*.1)); test_target = valid_target
        valid_count = test_count = 0
        for number, group in enumerate(values):
            if number < len(values)-2 and valid_count < valid_target:
                split = 'valid'; valid_count += len(group)
            elif number < len(values)-1 and test_count < test_target:
                split = 'test'; test_count += len(group)
            else: split = 'train'
            group_id = min(records[i]['pixel_sha256'] for i in group)
            for index in group:
                records[index].update(split=split, group=group_id)
    return records


def _save_prepared(folder, images, records, metadata):
    if not records: raise ValueError('No usable data was prepared')
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder/'images.npy', np.stack(images).astype(np.uint8))
    np.save(folder/'labels.npy', np.asarray([r['label'] for r in records], dtype=np.int64))
    np.save(folder/'splits.npy', np.asarray([r['split'] for r in records], dtype='<U5'))
    for index, row in enumerate(records): row['index'] = index
    (folder/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    metadata.update(samples=len(records), splits=dict(Counter(r['split'] for r in records)),
                    shape=[100,100], channels=1, class_counts=dict(Counter(r['label'] for r in records)))
    (folder/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    verify_prepared(folder)
    return metadata


def prepare_fer2013(csv_path, output):
    """Read original FER labels and preserve PublicTest/PrivateTest boundaries."""
    images, records = [], []
    with Path(csv_path).open(encoding='utf-8-sig',newline='') as handle:
        for index,row in enumerate(csv.DictReader(handle)):
            if row.get('Usage') not in USAGE_SPLITS: raise ValueError('FER-2013 requires its original Usage column')
            label = int(row['emotion'])
            if label not in FER_TO_PROJECT: raise ValueError('Invalid FER-2013 label')
            pixels = np.fromstring(row['pixels'],sep=' ',dtype=np.int16)
            if pixels.size!=2304 or (pixels<0).any() or (pixels>255).any():
                raise ValueError(f'Invalid 48x48 pixels in FER row {index}')
            original = pixels.astype(np.uint8).reshape(48,48)
            image = np.asarray(Image.fromarray(original).resize((100,100),Image.Resampling.BILINEAR))
            images.append(image)
            records.append({'source':f'fer2013/row-{index:05d}','label':FER_TO_PROJECT[label],
                            'original_label':label,'split':USAGE_SPLITS[row['Usage']],
                            'pixel_sha256':_pixel_hash(original),'group':_pixel_hash(original)})
    # Held-out pixels have priority; duplicated held-out images never enter training.
    priority={'train':0,'valid':1,'test':2}; owners={}
    for row in records:
        key=row['pixel_sha256']; owners[key]=max(owners.get(key,-1),priority[row['split']])
    keep=[i for i,r in enumerate(records) if priority[r['split']]==owners[r['pixel_sha256']]]
    counts=dict(Counter(r['split'] for r in records))
    with Path(csv_path).open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return _save_prepared(Path(output),[images[i] for i in keep],[records[i] for i in keep],
        {'dataset':'fer2013','labels':'original FER-2013 emotion column','csv_sha256':digest,
         'official_split_counts':counts,'cross_split_duplicates_removed':len(records)-len(keep),
         'split_scope':'official Usage; validation and test pixel overlap removed from lower-priority splits'})


def _aligned_item(item):
    source,label=item
    try:
        from expression_deid.preprocessing.align import select_face,landmarks_of,warp_gray
        from expression_deid.preprocessing.detect import YuNet
        if not hasattr(_local,'detector'): _local.detector=YuNet()
        with Image.open(source) as photo: rgb=np.asarray(photo.convert('RGB'))
        gray=to_gray(Image.fromarray(rgb))
        face,reason=select_face(_local.detector.detect(rgb),gray.shape)
        if face is None: return {'source':str(source),'error':reason},None
        aligned,fit=warp_gray(gray,landmarks_of(face),size=100)
        if fit>7 or face[14]<.65: return {'source':str(source),'error':'alignment_quality'},None
        return {'source':str(source),'label':label,'pixel_sha256':_pixel_hash(aligned),
                'dhash':_dhash(aligned),'alignment_error':fit},aligned
    except (OSError,ValueError) as exc:
        return {'source':str(source),'error':str(exc)},None


def prepare_faces(name, raw, output, *, workers=6, seed=42, limit=0):
    root=Path(raw); output=Path(output)
    paths=sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'} and not p.name.startswith('.'))
    if limit: paths=paths[:limit]
    if not paths: raise FileNotFoundError(f'No downloaded {name} images in {root}')
    classes=sorted({p.parent.name for p in paths}) if name=='pins' else []
    mapping={name:i for i,name in enumerate(classes)}
    inputs=[(p,mapping[p.parent.name] if classes else -1) for p in paths]
    images=[]; records=[]; rejected=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for number,(record,image) in enumerate(pool.map(_aligned_item,inputs),1):
            if image is None: rejected.append(record)
            else:
                record['source']=str(Path(record['source']).relative_to(root))
                records.append(record); images.append(image)
            if number%500==0: print(json.dumps({'dataset':name,'prepared':number,'usable':len(records)}),flush=True)
    # An image with conflicting identity labels cannot be a valid training target.
    labels=defaultdict(set)
    for r in records: labels[r['pixel_sha256']].add(r['label'])
    before=len(records)
    keep=[i for i,r in enumerate(records) if len(labels[r['pixel_sha256']])==1]
    images=[images[i] for i in keep]; records=[records[i] for i in keep]
    grouped_split(records,seed=seed,per_identity=name=='pins')
    output.mkdir(parents=True,exist_ok=True)
    (output/'rejected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rejected),encoding='utf-8')
    metadata={'dataset':name,'classes':mapping,'raw_images':len(paths),'rejected':len(rejected),'limited_preparation':bool(limit),
              'split_scope':'known-identity grouped images' if classes else 'image duplicate groups; identity labels not supplied',
              'seed':seed,'conflicting_duplicates_removed':before-len(keep)}
    return _save_prepared(output,images,records,metadata)


def verify_prepared(folder):
    folder=Path(folder)
    images=np.load(folder/'images.npy',mmap_mode='r',allow_pickle=False)
    labels=np.load(folder/'labels.npy',allow_pickle=False)
    splits=np.load(folder/'splits.npy',allow_pickle=False)
    records=[json.loads(line) for line in (folder/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(images)!=len(labels) or len(images)!=len(splits) or len(images)!=len(records): raise ValueError('Prepared arrays and manifest lengths differ')
    if images.dtype!=np.uint8 or images.shape[1:]!=(100,100): raise ValueError('Prepared images must be uint8 100x100 grayscale')
    groups=defaultdict(set)
    for index,row in enumerate(records):
        if row['label']!=int(labels[index]) or row['split']!=str(splits[index]): raise ValueError('Prepared label or split mismatch')
        groups['group:'+row.get('group',row['pixel_sha256'])].add(row['split'])
        groups['pixels:'+row['pixel_sha256']].add(row['split'])
    if any(len(value)>1 for value in groups.values()): raise ValueError('An image group appears in multiple partitions')
    if set(splits)!={'train','valid','test'}: raise ValueError('All three data partitions are required')
    return {'samples':len(images),'splits':dict(Counter(splits.tolist()))}


def prepare_required(name, *, data_root=None, workers=6, seed=42, limit=0):
    root=Path(data_root or ROOT/'datasets')
    if name=='fer2013':
        source=root/'fer2013/fer2013.csv'
        if not source.is_file(): raise FileNotFoundError('Place original fer2013.csv in datasets/fer2013/')
        return prepare_fer2013(source,root/'fer2013/prepared')
    if name=='kmu':
        return prepare_kmu_auxiliary(root/'kmu-fed-aligned',root/'kmu-auxiliary/prepared',seed=seed)
    if name=='ppb-emo':
        return prepare_ppb_auxiliary(root/'ppb-emo/neutral-aligned',root/'ppb-emo/prepared',seed=seed)
    if name=='raf':
        return prepare_raf_auxiliary(root/'raf-db',root/'raf-auxiliary/prepared',root/'fer2013/prepared',seed=seed)
    if name=='affectnet':
        return prepare_affectnet_auxiliary(root/'affectnet-kaggle/raw',root/'affectnet-auxiliary/prepared',
                                           root/'fer2013/prepared',workers=workers,seed=seed,limit=limit)
    if name not in {'celebahq','pins'}: raise ValueError('Unknown required dataset')
    return prepare_faces(name,root/name/'raw',root/name/'prepared',workers=workers,seed=seed,limit=limit)


def prepare_kmu_auxiliary(source, output, *, seed=42):
    """The sole optional expression auxiliary, with disjoint participants."""
    source=Path(source)
    entries=[json.loads(line) for line in (source/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    entries=[r for r in entries if r.get('status')=='ok' and r.get('person') and 0<=int(r.get('label',-1))<7]
    people=sorted({r['person'] for r in entries})
    if len(people)<5: raise ValueError('KMU auxiliary needs at least five participants')
    np.random.default_rng(seed).shuffle(people)
    n=max(1,round(len(people)*.2)); owners={p:'valid' if i<n else 'test' if i<2*n else 'train' for i,p in enumerate(people)}
    records=[]; images=[]
    for row in entries:
        relative=Path(row['output'])
        if relative.is_absolute() or relative.drive or '..' in relative.parts: raise ValueError('Invalid KMU image path')
        with Image.open(source/relative) as photo:
            image=np.asarray(photo.convert('L').resize((100,100),Image.Resampling.BILINEAR))
        images.append(image); records.append({'source':row['source'],'label':int(row['label']),
            'split':owners[row['person']],'group':'person-'+row['person'],'person':row['person'],'pixel_sha256':_pixel_hash(image)})
    return _save_prepared(Path(output),images,records,{'dataset':'kmu','role':'auxiliary only',
        'split_scope':'participants disjoint across train, valid and test','people':owners,'seed':seed})


def prepare_ppb_auxiliary(source, output, *, seed=42):
    source=Path(source)
    rows=[json.loads(s) for s in (source/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    rows=[r for r in rows if r['status']=='ok' and int(r['label'])==6]
    people=sorted({r['person'] for r in rows}); np.random.default_rng(seed).shuffle(people)
    if len(people)<10: raise ValueError('Too few independent PPB participants')
    n=max(1,round(len(people)*.2)); owners={p:'valid' if i<n else 'test' if i<2*n else 'train' for i,p in enumerate(people)}
    images=[]; records=[]
    for row in rows:
        relative=Path(row['output'])
        if relative.is_absolute() or relative.drive or '..' in relative.parts: raise ValueError('Invalid PPB image path')
        with Image.open(source/relative) as photo:
            image=np.asarray(photo.convert('L').resize((100,100),Image.Resampling.BILINEAR))
        images.append(image); records.append({'source':row['source']+':'+str(row['frame']),
            'label':6,'split':owners[row['person']],'group':'person-'+row['person'],
            'person':row['person'],'pixel_sha256':_pixel_hash(image)})
    return _save_prepared(Path(output),images,records,{'dataset':'ppb-emo','role':'auxiliary only',
        'label_scope':'central IR neutral ND clips only','split_scope':'participants disjoint across train, valid and test',
        'people':owners,'seed':seed})


def _perceptual_hash(image):
    import cv2
    small=cv2.resize(image,(32,32)).astype(np.float32)
    flat=cv2.dct(small)[:8,:8].ravel()[1:]
    return int.from_bytes(np.packbits(flat>np.median(flat)).tobytes(),'big')


def prepare_raf_auxiliary(source,output,primary,*,seed=42):
    """Keep RAF official train and exclude overlap with every prepared FER image."""
    import cv2
    cv2.setNumThreads(1)
    source=Path(source); verify_prepared(primary)
    primary_images=np.load(Path(primary)/'images.npy',mmap_mode='r',allow_pickle=False)
    buckets=defaultdict(list)
    for image in primary_images:
        code=_perceptual_hash(image)
        for shift in range(8): buckets[(shift,(code>>(8*shift))&255)].append(code)
    images=[]; rows=[]; excluded=0
    for official in ['train','valid']:
        paths=sorted(p for p in (source/official).rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'})
        if not paths: raise FileNotFoundError(f'Missing RAF {official} images')
        for path in paths:
            label=int(path.parent.name)
            if label not in range(7): raise ValueError('RAF folders must use project labels 0..6')
            with Image.open(path) as photo:
                image=np.asarray(photo.convert('L').resize((100,100),Image.Resampling.BILINEAR))
            code=_perceptual_hash(image); candidates=set()
            for shift in range(8): candidates.update(buckets[(shift,(code>>(8*shift))&255)])
            if any((code^previous).bit_count()<=7 for previous in candidates):
                excluded+=1; continue
            images.append(image); rows.append({'source':str(path.relative_to(source)),'label':label,
                'official_split':official,'pixel_sha256':_pixel_hash(image),'dhash':_dhash(image),'phash':code})
    grouped_split(rows,seed=seed)
    heldout={r['group'] for r in rows if r['official_split']=='valid'}
    groups=sorted(heldout); np.random.default_rng(seed).shuffle(groups)
    owners={group:'valid' if i%2==0 else 'test' for i,group in enumerate(groups)}
    kept=[]; kept_images=[]; duplicate_removed=0
    for image,row in zip(images,rows):
        if row['official_split']=='train' and row['group'] in heldout:
            duplicate_removed+=1; continue
        row['split']='train' if row['official_split']=='train' else owners[row['group']]
        kept.append(row); kept_images.append(image)
    return _save_prepared(Path(output),kept_images,kept,{'dataset':'raf','role':'auxiliary only',
        'seed':seed,'cross_primary_near_duplicates_removed':excluded,'cross_official_duplicates_removed':duplicate_removed,
        'split_scope':'official RAF train; official test image groups divided into project valid and test',
        'primary_overlap_policy':'exclude RAF images within 7 perceptual-hash bits of any prepared FER-2013 image'})


def _affectnet_val_split(records, seed=42):
    """Split supplied AffectNet validation images into project valid/test."""
    groups=defaultdict(list)
    for index,row in enumerate(records): groups[row['phash']].append(index)
    assignments={}
    for group in sorted(groups):
        digest=hashlib.sha256(f'{seed}:{group}'.encode()).digest()
        assignments[group]='valid' if digest[0] % 2 == 0 else 'test'
    for label in range(7):
        for wanted in ('valid','test'):
            if any(r['label']==label and assignments[r['phash']]==wanted for r in records): continue
            candidates=[r for r in records if r['label']==label]
            if candidates: assignments[candidates[0]['phash']]=wanted
    for row in records: row['split']=assignments[row['phash']]


def prepare_affectnet_auxiliary(source,output,primary,*,workers=6,seed=42,limit=0):
    """Prepare the supplied AffectNet-Kaggle package as one expression auxiliary.

    Classes 1..7 map to the project's seven classes. Class 8 (contempt) is
    excluded because the required FER classifier has seven outputs. All images
    use the same YuNet/five-point alignment path as the primary face datasets.
    """
    source=Path(source); verify_prepared(primary)
    mapping={1:6,2:3,3:4,4:0,5:1,6:2,7:5}
    entries=[]; excluded_label=0
    for official,folder_name in [('train','train_class'),('val','val_class')]:
        for class_id in range(1,9):
            paths=sorted((source/folder_name/f'class{class_id:03d}').glob('*.jpg'))
            if class_id==8:
                excluded_label+=len(paths); continue
            entries.extend((path,mapping[class_id],official) for path in paths)
    if limit: entries=entries[:limit]
    if not entries: raise FileNotFoundError('No AffectNet JPG images found under affectnet-kaggle/raw')
    prepared=[]; images=[]; rejected=[]
    def aligned(entry):
        path,label,official=entry
        row,image=_aligned_item((path,label)); row['official_split']=official
        return row,image
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for number,(row,image) in enumerate(pool.map(aligned,entries),1):
            if image is None: rejected.append(row)
            else:
                row['source']=str(Path(row['source']).relative_to(source))
                row['phash']=_perceptual_hash(image)
                prepared.append(row); images.append(image)
            if number%500==0: print(json.dumps({'dataset':'affectnet','prepared':number,'usable':len(prepared)}),flush=True)
    primary_images=np.load(Path(primary)/'images.npy',mmap_mode='r',allow_pickle=False)
    buckets=defaultdict(list)
    for image in primary_images:
        code=_perceptual_hash(image)
        for shift in range(8): buckets[(shift,(code>>(8*shift))&255)].append(code)
    kept=[]; kept_images=[]; overlap_removed=0
    for image,row in zip(images,prepared):
        code=row['phash']; candidates=set()
        for shift in range(8): candidates.update(buckets[(shift,(code>>(8*shift))&255)])
        if any((code-previous).bit_count()<=7 for previous in candidates):
            overlap_removed+=1; continue
        kept.append(row); kept_images.append(image)
    prepared,images=kept,kept_images
    # The package's train/val split is not guaranteed to be duplicate-free.
    # Give the supplied training partition priority and remove matching val
    # groups before making the project's independent valid/test partitions.
    train_phashes={row['phash'] for row in prepared if row['official_split']=='train'}
    filtered=[]; filtered_images=[]; internal_removed=0
    for image,row in zip(images,prepared):
        if row['official_split']=='val' and row['phash'] in train_phashes:
            internal_removed+=1; continue
        filtered.append(row); filtered_images.append(image)
    prepared,images=filtered,filtered_images
    validation=[row for row in prepared if row['official_split']=='val']
    _affectnet_val_split(validation,seed=seed)
    validation_by_source={row['source']:row['split'] for row in validation}
    for row in prepared:
        row['split']='train' if row['official_split']=='train' else validation_by_source[row['source']]
        row['group']=str(row['phash'])
    metadata={'dataset':'affectnet','role':'auxiliary only','label_scope':'AffectNet classes 1..7; class 8 contempt excluded',
        'class_mapping':mapping,'excluded_contempt':excluded_label,'rejected':len(rejected),
        'cross_primary_near_duplicates_removed':overlap_removed,
        'cross_internal_train_val_duplicates_removed':internal_removed,
        'split_scope':'supplied AffectNet train; supplied val deterministically divided into project valid/test',
        'seed':seed,'source':'AffectNet-Kaggle package supplied by user'}
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    (output/'rejected.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rejected),encoding='utf-8')
    return _save_prepared(output,images,prepared,metadata)
