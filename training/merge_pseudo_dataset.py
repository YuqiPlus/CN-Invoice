"""Merge qualified teacher labels without mixing invoice identities across splits."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path


def invoice_group(number):return hashlib.sha256(('vat:'+str(number)).encode()).hexdigest()
def pseudo_split(group):return 'val' if int(group[:8],16)%5==0 else 'train'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--labels',type=Path,nargs='+',required=True)
    parser.add_argument('--base-identities',type=Path,required=True)
    parser.add_argument('--heldout',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if '.private' not in args.output.resolve().parts:parser.error('Private dataset output required')
    args.output.mkdir(parents=True,exist_ok=True)
    base=json.loads(args.base.read_text());records=[]
    identities=json.loads(args.base_identities.read_text())
    reference=json.loads(args.heldout.read_text())['samples']
    forbidden={s['sha256'] for s in reference};forbidden_pixels={s['pixel_sha256'] for s in reference}
    seen_hashes=set();seen_pixels=set();existing={}
    for row in base['records']:
        row=dict(row);path=Path(row['file']);row['file']=str((path if path.is_absolute() else args.base.parent/path).resolve())
        if row['id'] in identities:
            row['group']=invoice_group(identities[row['id']]);existing[row['group']]=row['split']
        assert row['sha256'] not in forbidden and row.get('pixel_sha256') not in forbidden_pixels
        records.append(row);seen_hashes.add(row['sha256']);seen_pixels.add(row.get('pixel_sha256'))
    candidates=defaultdict(list);rejected=Counter()
    for label_file in args.labels:
        labelled=json.loads(label_file.read_text())
        if labelled['status']!='complete':raise ValueError('Labelling batch is incomplete')
        for case in labelled['cases']:
            if not case.get('accepted'):continue
            row=case['source'];number=case['values'].get('invoice_number')
            if not number:rejected['missing_business_identity']+=1;continue
            if row['sha256'] in forbidden or row.get('pixel_sha256') in forbidden_pixels:
                rejected['heldout_image']+=1;continue
            if row['sha256'] in seen_hashes or row.get('pixel_sha256') in seen_pixels:
                rejected['duplicate_image']+=1;continue
            group=invoice_group(number)
            split=existing.get(group,pseudo_split(group))
            image_path=Path(row['file'])
            if not image_path.is_absolute():raise ValueError('Teacher source image paths must be explicit')
            if hashlib.sha256(image_path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Teacher source image changed')
            item={'id':'PSEUDO-'+case['id'],'file':str(image_path),'sha256':row['sha256'],
                  'pixel_sha256':row['pixel_sha256'],'width':row['width'],'height':row['height'],
                  'category':'vat','group':group,'split':split,'origin':'lite_grounded_pseudo',
                  'label_quality':'auto_verified_pseudo_not_human_ground_truth','annotations':case['annotations'],
                  'teacher_model':case['teacher_payload'].get('_backend_model'),
                  'teacher_result_file':str(label_file.resolve()),'source_url':row.get('source_url')}
            candidates[group].append(item);seen_hashes.add(row['sha256']);seen_pixels.add(row.get('pixel_sha256'))
    added=[]
    for group,items in candidates.items():
        items.sort(key=lambda r:sum(a.get('ocr_score',0) for a in r['annotations'])/len(r['annotations']),reverse=True)
        added.extend(items[:3]);rejected['excess_views_same_invoice']+=max(0,len(items)-3)
    records.extend(added)
    train={r['group'] for r in records if r['split']=='train'};val={r['group'] for r in records if r['split']=='val'}
    assert not train&val
    manifest={**base,'records':records,'counts':dict(Counter(r['split']+':'+r['category'] for r in records)),
        'pseudo_summary':{'images':len(added),'distinct_invoice_groups':len(candidates),
                          'counts':dict(Counter(r['split'] for r in added)),'excluded':dict(rejected)},
        'notes':base['notes']+['Added Lite-prelabelled samples are noisy pseudo labels, not human ground truth.',
                              'Money must match literal OCR boxes and exact arithmetic; available native PDF values must agree.',
                              'Pseudo splits are by invoice identity, with at most three views per invoice.']}
    (args.output/'dataset.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'counts':manifest['counts'],'pseudo':manifest['pseudo_summary']},ensure_ascii=False))


if __name__=='__main__':main()
