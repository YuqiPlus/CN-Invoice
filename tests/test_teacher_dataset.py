import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def builder(monkeypatch):
    folder=Path(__file__).parents[1]/'training'
    monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location('build_teacher_dataset',folder/'build_teacher_dataset.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def case(tmp_path,name,color,number='12345678'):
    path=tmp_path/(name+'.png');image=Image.new('RGB',(500,300),color);image.save(path)
    source={'id':name,'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'pixel_sha256':hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest(),
        'width':500,'height':300}
    return {'id':name,'source':source,'accepted':True,'kind':'other',
        'values':{'other.invoice_number':number},'label_file':str(tmp_path/'results.json'),
        'annotations':[{'field':'other.invoice_number','text':number,'box':[10,20,110,45],
                        'location_basis':'teacher_selected_ocr_box'}]}


def test_teacher_views_of_one_number_stay_in_one_split(builder,tmp_path):
    result=builder.build([case(tmp_path,'one','red'),case(tmp_path,'two','blue')],[])
    rows=result['records']
    assert len(rows)==2 and len({r['group'] for r in rows})==1
    assert len({r['split'] for r in rows})==1
    assert {r['origin'] for r in rows}=={'lite_teacher_supervised'}


def test_known_test_number_excludes_different_training_photo(builder,tmp_path):
    candidate=case(tmp_path,'teacher','red')
    gold=case(tmp_path,'gold','blue')['source']
    gold['expected']={'invoice_number':'12345678'}
    result=builder.build([candidate],[gold])
    assert not result['records']
    assert result['teacher_summary']['exclusions']['known_heldout_number']==1


def test_training_target_cannot_silently_differ_from_teacher(builder,tmp_path):
    candidate=case(tmp_path,'teacher','red')
    candidate['annotations'][0]['text']='87654321'
    with pytest.raises(ValueError,match='differs from preserved teacher'):
        builder.build([candidate],[])


def test_training_source_bytes_are_verified(builder,tmp_path):
    candidate=case(tmp_path,'teacher','red')
    Image.new('RGB',(500,300),'green').save(candidate['source']['file'])
    with pytest.raises(ValueError,match='image changed'):
        builder.build([candidate],[])


def test_test_ticket_identity_from_publisher_text_excludes_another_view(builder,tmp_path):
    candidate=case(tmp_path,'teacher','red',number='7811234567890')
    gold=case(tmp_path,'gold','blue')['source']
    ocr=tmp_path/'publisher-text.json'
    ocr.write_text(json.dumps([{'text':'781-1234567890'}]))
    gold['ocr_file']=str(ocr)
    assert not builder.build([candidate],[gold])['records']


def test_source_ticket_identity_keeps_views_together_even_when_teacher_misses_it(builder,tmp_path):
    first=case(tmp_path,'first','red',number='11111111')
    second=case(tmp_path,'second','blue',number='22222222')
    first['source']['identity_tokens_for_isolation_only']=['A1234567']
    second['source']['identity_tokens_for_isolation_only']=['A1234567']
    rows=builder.build([first,second],[])['records']
    assert len(rows)==2 and len({r['group'] for r in rows})==1
