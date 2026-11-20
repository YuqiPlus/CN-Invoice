import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from cninvoice import DocumentProfile, InvoiceReader, ReaderConfig, RecognitionError, SpecialistVisionBackend
from cninvoice.models import Source
from cninvoice.ocr import OCRLine
from cninvoice.parsers import classify
from cninvoice.profiles import route_profile
from cninvoice.vision import FieldDetection, extract_detected_fields


@pytest.mark.parametrize('texts,expected',[
    (['电子发票（普通发票）'], 'vat_invoice'),
    (['增值税专用发票'], 'vat_invoice'),
    (['电子发票', '（铁路电子客票）'], 'rail_einvoice'),
    (['铁路电子客票'], 'rail_legacy'),
    (['火车票'], 'rail_legacy'),
    (['北京市出租汽车专用发票'], 'taxi_receipt'),
    (['滴滴出行行程报销单'], 'ride_itinerary'),
    (['高德打车', '行程单'], 'ride_itinerary'),
    (['航空运输电子客票行程单'], 'air_itinerary'),
    (['航空运输电子客票行程单','温馨提示：滴滴出行'], 'air_itinerary'),
    (['滴滴出行', '金额100.00'], 'unknown'),
    (['普通发票', '税额', '纳税人识别号', '价税合计'], 'vat_invoice'),
    (['价税合计'], 'unknown'),
    (['通发票','纳税人识别号','货物或应税劳务、服务名称','税率','税'], 'vat_invoice'),
    (['航空运输','电子客票号码','合计 TOTAL'], 'air_itinerary'),
    (['电子发票（普通发票）', '备注：铁路客运服务'], 'vat_invoice'),
    (['电子发票（普通发票）', '滴滴出行行程报销单'], 'unknown'),
])
def test_document_profiles_separate_target_and_legacy_families(texts,expected):
    assert route_profile(texts).profile.value == expected


def test_ride_native_classification_is_not_air():
    assert classify('滴滴出行行程报销单') == 'ride_itinerary'


def models(tmp_path,profile='vat_invoice',fields=None):
    model=tmp_path/'model';model.mkdir()
    (model/'model.json').write_text(json.dumps({'document_profile':profile,'fields':fields or ['vat.total'],'input_size':320}))
    (model/'detector.onnx').write_bytes(b'test-never-loaded')
    ocr=tmp_path/'ocr';ocr.mkdir()
    from cninvoice.ocr import RapidOCRBackend
    for name in RapidOCRBackend.MODEL_FILES:(ocr/name).write_bytes(b'test-never-loaded')
    return model,ocr


def set_ocr(backend,texts):
    class OCR:
        calls=0
        def recognize(self,images):
            self.calls+=1
            return {'kind':'vat_invoice','total':'999.00','_ocr_lines':[
                {'text':t,'bbox':[0,i*30,300,i*30+20],'score':.98,'page':1} for i,t in enumerate(texts)]}
    backend.ocr=OCR()


def test_only_one_expert_and_one_shared_ocr_pass(monkeypatch,tmp_path):
    directory,ocr=models(tmp_path)
    backend=SpecialistVisionBackend({'vat_invoice':directory},ocr)
    set_ocr(backend,['电子发票（普通发票）'])
    class Expert:
        calls=0
        def __init__(self,*args,**kwargs):assert kwargs['ocr_backend'] is backend.ocr
        def recognize_with_ocr(self,images,baseline,**kwargs):
            self.calls+=1
            assert baseline['_ocr_lines'] and kwargs['known_kind']=='vat_invoice'
            return {'kind':'vat_invoice','total':'113.00'}
    monkeypatch.setattr('cninvoice.specialists.FieldVisionBackend',Expert)
    with Image.new('RGB',(100,100)) as image:
        result=backend.recognize([image])
    assert result['total']=='113.00' and backend.ocr.calls==1
    assert list(backend._experts)==[DocumentProfile.VAT]
    assert result['_specialist_route']['status']=='expert_used'


@pytest.mark.parametrize('texts,code',[
    (['电子发票（铁路电子客票）'],'specialist_unavailable'),
    (['滴滴出行行程报销单'],'specialist_unavailable'),
    (['火车票'],'specialist_unsupported_document'),
    (['出租汽车专用发票'],'specialist_unsupported_document'),
    (['电子发票（普通发票）','滴滴出行行程报销单'],'specialist_route_conflict'),
])
def test_missing_or_legacy_profile_never_uses_merged_ocr_money(tmp_path,texts,code):
    directory,ocr=models(tmp_path)
    backend=SpecialistVisionBackend({'vat_invoice':directory},ocr)
    set_ocr(backend,texts)
    with Image.new('RGB',(100,100)) as image:
        data=backend.recognize([image])
    assert data['total'] is None and data['_ocr_evidence']=={}
    assert data['_ocr_issues'][0]['code']==code and not backend._experts


def test_explicit_profile_cannot_override_conflicting_title(tmp_path):
    directory,ocr=models(tmp_path)
    backend=SpecialistVisionBackend({'vat_invoice':directory},ocr,document_profile='vat_invoice')
    set_ocr(backend,['航空运输电子客票行程单'])
    with Image.new('RGB',(100,100)) as image:data=backend.recognize([image])
    assert data['_ocr_issues'][0]['code']=='specialist_route_conflict' and not backend._experts


def test_merged_model_cannot_masquerade_as_vat_expert(tmp_path):
    directory,ocr=models(tmp_path,fields=['vat.total','other.total'])
    with pytest.raises(RecognitionError,match='metadata'):
        SpecialistVisionBackend({'vat_invoice':directory},ocr)


def test_config_accepts_missing_experts_without_cloud_and_native_xml_still_first(tmp_path,xml_invoice):
    directory,ocr=models(tmp_path)
    reader=ReaderConfig(recognizer='specialists',specialist_model_dirs={'vat_invoice':directory},ocr_model_dir=ocr).create_reader()
    assert reader.reviewer is None
    assert reader.read(xml_invoice).total == 113
    assert reader.backend.ocr._engine is None


def test_new_railway_invoice_number_is_not_ticket_number_or_vat_kind():
    fields=['rail.invoice_number','rail.total']
    data=extract_detected_fields([
        FieldDetection('rail.invoice_number',(0,0,300,20),.9),
        FieldDetection('rail.total',(0,40,300,60),.9)],
        [OCRLine('发票号码：12345678901234567890',(0,0,300,20),.99),
         OCRLine('￥113.00',(0,40,300,60),.99)],known_kind='rail_ticket',trained_fields=fields)
    assert data['kind']=='rail_ticket' and data['invoice_number']=='12345678901234567890'
    assert data['rail']['ticket_number'] is None and data['total']=='113.00'


def test_ride_total_keeps_distinct_kind_and_literal_amount():
    data=extract_detected_fields([FieldDetection('ride.total',(0,0,100,20),.9)],
        [OCRLine('￥113.00',(0,0,100,20),.99)],known_kind='ride_itinerary',trained_fields=['ride.total'])
    assert data['kind']=='ride_itinerary' and data['total']=='113.00' and data['tax'] is None


spec=importlib.util.spec_from_file_location('build_specialists',Path(__file__).parents[1]/'training/build_specialists.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def row(identifier,category,split,group,field):
    return {'id':identifier,'category':category,'split':split,'group':group,'file':identifier+'.png',
            'annotations':[{'field':field,'text':'113.00','box':[0,0,100,20]}]}


def test_split_builder_isolates_legacy_and_preserves_groups_and_field_targets(tmp_path):
    rows=[row('vat','vat_invoice','train','groupA','vat.total'),
          row('rail','rail_ticket','val','groupB','rail.total'),
          row('taxi','taxi_invoice','train','groupC','taxi.total')]
    titles={'vat':['电子发票（普通发票）'],'rail':['火车票'],'taxi':['出租汽车专用发票']}
    selected,audit=builder.partition({'records':rows},tmp_path,lambda r:titles[r['id']])
    assert len(selected[DocumentProfile.VAT])==1
    assert not selected[DocumentProfile.RAIL_EINVOICE] and not selected[DocumentProfile.RIDE_ITINERARY]
    saved=selected[DocumentProfile.VAT][0]
    assert saved['split']=='train' and saved['group']=='groupA' and saved['annotations']==rows[0]['annotations']


def test_split_builder_never_relabels_teacher_category_conflicts(tmp_path):
    r=row('bad','other','train','group','other.total')
    selected,audit=builder.partition({'records':[r]},tmp_path,lambda r:['电子发票（普通发票）'])
    assert not any(selected.values())
    assert audit[0]['reason']=='teacher_category_conflict_requires_reannotation'


def test_split_builder_rejects_train_validation_identity_overlap(tmp_path):
    rows=[row('a','vat_invoice','train','same','vat.total'),row('b','vat_invoice','val','same','vat.total')]
    with pytest.raises(ValueError,match='overlap'):
        builder.partition({'records':rows},tmp_path,lambda r:['电子发票（普通发票）'])


def test_empty_family_is_not_trainable():
    assert builder.readiness([])['status']=='insufficient_data'
