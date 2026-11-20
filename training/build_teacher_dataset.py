"""Build train/validation data from Lite labels, excluding independent test identities."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from label_corpus import FIELDS, convert


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(value):
    value=re.sub(r'[-\s]', '', str(value or '')).upper()
    return value if re.fullmatch(r'(?:[A-Z]\d{5,25}|\d{6,30})',value) else None


def load_labels(path, legacy=False):
    report=json.loads(path.read_text())
    if report['status']!='complete':
        raise ValueError('Teacher batch is incomplete')
    for case in report['cases']:
        if legacy and case.get('teacher_payload'):
            case=dict(case);source=case['source'];payload=case['teacher_payload']
            lines=json.loads((path.parent/'ocr'/(case['id']+'.json')).read_text())
            fields={}
            for name,entry in (payload.get('fields') or {}).items():
                if isinstance(entry,dict):fields['vat.'+name]=[entry.get('value'),entry.get('ocr_id')]
            case.update(convert({**payload,'fields':fields},lines,source['width'],source['height']))
        yield {**case,'label_file':str(path.resolve())}


def build(cases, references, prior_numbers=(), max_views=3):
    forbidden_hashes={s['sha256'] for s in references}
    forbidden_pixels={s['pixel_sha256'] for s in references}
    forbidden_groups={s['source_group'] for s in references if s.get('source_group')}
    forbidden_numbers={identity(n) for n in prior_numbers if identity(n)}
    for source in references:
        expected=source.get('expected',{})
        for key in ('invoice_number','rail.ticket_number','air.ticket_number'):
            number=identity(expected.get(key))
            if number:forbidden_numbers.add(number)
        number=identity(source.get('publisher_invoice_number_for_isolation_only'))
        if number:forbidden_numbers.add(number)
        if source.get('ocr_file'):
            for line in json.loads(Path(source['ocr_file']).read_text()):
                text=re.sub(r'\s','',line['text']).upper()
                if re.fullmatch(r'[A-Z]\d{6,20}|\d{3}-?\d{10}',text):
                    forbidden_numbers.add(identity(text))
    records=[];excluded=Counter();seen_pixels=set();seen_hashes=set()
    for case in sorted(cases,key=lambda c:(-len(c.get('annotations',[])),c['id'])):
        if not case.get('accepted') or case.get('error'):
            excluded['no_usable_teacher_boxes']+=1;continue
        source=case['source']
        if source.get('split')=='test' or source['sha256'] in forbidden_hashes or source['pixel_sha256'] in forbidden_pixels or source.get('source_group') in forbidden_groups:
            excluded['heldout_image_or_publisher_identity']+=1;continue
        values=case.get('values',{})
        numbers={identity(value) for field,value in values.items() if field.endswith(('.invoice_number','.ticket_number'))}
        numbers.discard(None)
        publisher_number=identity(source.get('publisher_invoice_number_for_isolation_only'))
        if publisher_number:numbers.add(publisher_number)
        numbers.update(identity(value) for value in source.get('identity_tokens_for_isolation_only',[]) if identity(value))
        if numbers&forbidden_numbers:
            excluded['known_heldout_number']+=1;continue
        path=Path(source['file'])
        if digest(path.read_bytes())!=source['sha256']:
            raise ValueError('Teacher image changed after labelling')
        with Image.open(path) as image:
            width,height=image.size
            pixel=digest(str(image.size).encode()+image.convert('RGB').tobytes())
        if pixel!=source['pixel_sha256'] or (width,height)!=(source['width'],source['height']):
            raise ValueError('Teacher image dimensions/pixels changed')
        if pixel in seen_pixels or source['sha256'] in seen_hashes:
            excluded['duplicate_pixels_or_bytes']+=1;continue
        if source.get('source_dataset')=='github_public_media':
            ocr=Path(case['label_file']).parent/'ocr'/(case['id']+'.json')
            if ocr.exists():
                text=''.join(line['text'] for line in json.loads(ocr.read_text()))
                if len(re.findall(r'[\u4e00-\u9fff]',text))<3:
                    excluded['non_chinese_public_media']+=1;continue
        annotations=[]
        for annotation in case.get('annotations',[]):
            field=annotation['field'];x1,y1,x2,y2=annotation['box']
            if field not in FIELDS or not (0<=x1<x2<=width and 0<=y1<y2<=height):
                raise ValueError('Invalid teacher training target')
            if annotation['text']!=values[field]:
                raise ValueError('Annotation differs from preserved teacher value')
            if (x2-x1)*320/width>1 and (y2-y1)*320/height>1:
                annotations.append(annotation)
        if not annotations:
            excluded['boxes_too_small_at_model_resolution']+=1;continue
        tokens={'number:'+n for n in numbers}
        if source.get('source_group'):tokens.add('publisher:'+source['source_group'])
        url=source.get('source_url','')
        if '.pdf' in url.lower():tokens.add('document:'+url.split('#')[0])
        if not tokens:tokens.add('pixel:'+pixel)
        records.append({**source,'origin':'lite_teacher_supervised','category':case['kind'],
            'label_quality':'lite_teacher_supervision_not_human_ground_truth',
            'annotations':annotations,'teacher_values':values,'teacher_flags':case.get('flags',[]),
            'teacher_result_file':case['label_file'],'identity_tokens':sorted(tokens)})
        seen_pixels.add(pixel);seen_hashes.add(source['sha256'])
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    tokens={}
    for i,row in enumerate(records):
        for token in row['identity_tokens']:
            if token in tokens:parent[find(i)]=find(tokens[token])
            else:tokens[token]=i
    grouped=defaultdict(list)
    for i,row in enumerate(records):grouped[find(i)].append(row)
    selected=[]
    for items in grouped.values():
        group=digest('\n'.join(sorted({token for row in items for token in row['identity_tokens']})).encode())
        split='val' if int(group[:8],16)%5==0 else 'train'
        items.sort(key=lambda row:(-len(row['annotations']),row['id']))
        for row in items[:max_views]:selected.append({**row,'group':group,'split':split})
        excluded['excess_views_same_identity']+=max(0,len(items)-max_views)
    assert not ({r['group'] for r in selected if r['split']=='train'}&{r['group'] for r in selected if r['split']=='val'})
    return {'version':2,'fields':FIELDS,'records':sorted(selected,key=lambda row:row['id']),
        'counts':dict(Counter(r['split']+':'+r['category'] for r in selected)),
        'teacher_summary':{'labelled_records':len(selected),'known_identity_groups':len(grouped),
            'origin_counts':dict(Counter(r['origin'] for r in selected)),
            'location_bases':dict(Counter(a.get('location_basis','legacy_teacher_selected_box') for r in selected for a in r['annotations'])),
            'exclusions':dict(excluded)},
        'notes':['All training field targets originate from Lite, not publisher KV labels or synthetic examples.',
                 'Publisher text boxes support alignment; publisher KV values are used only for holdout evaluation and identity isolation.',
                 'OCR transcription agreement is not required: Lite may correct OCR; missing fields remain partial labels.',
                 'Same known invoice numbers, publisher groups and PDF documents stay in one split; unknown duplicates may remain.',
                 'Teacher labels and teacher-estimated boxes are noisy supervision, not human ground truth.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,nargs='+',required=True)
    parser.add_argument('--legacy-labels',type=Path,nargs='*',default=[])
    parser.add_argument('--references',type=Path,nargs='+',required=True)
    parser.add_argument('--holdout-collections',type=Path,nargs='*',default=[])
    parser.add_argument('--prior-results',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if '.private' not in args.output.resolve().parts:parser.error('Private dataset output required')
    references=[];prior=[];source_hashes={}
    for path in args.references:
        data=json.loads(path.read_text());references.extend(data['samples']);prior.extend(data.get('known_forbidden_identity_tokens',[]));source_hashes[str(path)]=digest(path.read_bytes())
    for path in args.holdout_collections:
        references.extend(s for s in json.loads(path.read_text())['samples'] if s.get('split')=='test');source_hashes[str(path)]=digest(path.read_bytes())
    if args.prior_results:
        for path in args.prior_results.glob('*.json'):
            invoice=json.loads(path.read_text()).get('invoice',{})
            prior.extend([invoice.get('invoice_number'),(invoice.get('rail') or {}).get('ticket_number'),(invoice.get('air') or {}).get('ticket_number')])
    cases=[]
    for legacy,paths in [(False,args.labels),(True,args.legacy_labels)]:
        for path in paths:cases.extend(load_labels(path,legacy));source_hashes[str(path)]=digest(path.read_bytes())
    result=build(cases,references,prior);result['source_sha256']=source_hashes
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'dataset.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'counts':result['counts'],'teacher_summary':result['teacher_summary']}),flush=True)


if __name__=='__main__':main()
