import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('label_corpus',Path(__file__).parents[1]/'training/label_corpus.py')
teacher=importlib.util.module_from_spec(spec);spec.loader.exec_module(teacher)


@pytest.mark.parametrize('raw,expected',[
    ('70.00元','70.00'),('伍拾元整','50.00'),('拾元整','10.00'),
    ('人民币￥15元整','15.00'),('￥29·00元','29.00'),('负壹佰元零伍分','-100.05')])
def test_explicit_teacher_money_formats_keep_the_same_value(raw,expected):
    result=teacher.convert({'document_count':1,'kind':'other',
        'fields':{'other.total':[raw,0]}},[{'text':raw,'bbox':[10,20,110,40]}],500,300)
    assert result['accepted'] and result['values']['other.total']==expected
    assert result['annotations'][0]['text']==expected
    assert result['value_normalizations']['other.total']['original']==raw


@pytest.mark.parametrize('raw',['10+3','$12.00','12.345元','','未读到','70元+1元'])
def test_teacher_money_is_not_computed_rounded_or_assigned_a_foreign_currency(raw):
    result=teacher.convert({'document_count':1,'kind':'other',
        'fields':{'other.total':[raw,0]}},[{'text':raw,'bbox':[10,20,110,40]}],500,300)
    assert not result['accepted'] and 'other.total:invalid_money' in result['flags']


def test_teacher_transcription_is_not_replaced_by_erroneous_ocr():
    lines=[{'text':'￥100.80','bbox':[10,20,110,45],'score':.6}]
    result=teacher.convert({'document_count':1,'kind':'vat_invoice',
        'fields':{'vat.total':['100.00',0]}},lines,500,300)
    assert result['accepted'] and result['values']['vat.total']=='100.00'
    assert result['annotations'][0]['text']=='100.00'
    assert result['annotations'][0]['box']==[10,20,110,45]
    assert 'teacher_amount_incomplete' in result['flags']
    assert result['label_quality']=='lite_teacher_supervision_not_human_ground_truth'


def test_missing_ocr_still_keeps_teacher_value_and_valid_visual_box():
    result=teacher.convert({'document_count':1,'kind':'other',
        'fields':{'other.total':['20.00',None,[100,200,400,250]]}},[],1000,500)
    assert result['accepted']
    assert result['annotations'][0]['location_basis']=='teacher_estimated_box'
    assert result['annotations'][0]['box']==[100,100,400,125]


def test_value_without_location_is_preserved_but_not_a_detector_target():
    result=teacher.convert({'document_count':1,'kind':'other',
        'fields':{'other.total':['20.00',None,None]}},[],1000,500)
    assert not result['accepted'] and result['values']['other.total']=='20.00'
    assert 'other.total:value_without_location' in result['flags']


def test_invalid_bbox_and_multiple_invoices_do_not_become_supervised_boxes():
    payload={'document_count':1,'kind':'other',
        'fields':{'other.total':['20.00',None,[-5,200,400,250]]}}
    assert not teacher.convert(payload,[],1000,500)['accepted']
    payload['document_count']=2
    assert teacher.convert(payload,[],1000,500)['flags']==['not_one_supported_invoice']


def test_arithmetic_mismatch_keeps_raw_labels_without_training_bad_money_or_computing():
    lines=[{'text':'a','bbox':[i*100,10,i*100+80,35]} for i in range(4)]
    result=teacher.convert({'document_count':1,'kind':'vat_invoice','fields':{
        'vat.amount':['100',0],'vat.tax':['13',1],'vat.total':['114',2],
        'vat.invoice_number':['12345678',3]}},lines,500,300)
    assert result['accepted'] and result['values']['vat.total']=='114.00'
    assert 'teacher_amount_mismatch' in result['flags']
    assert [a['field'] for a in result['annotations']]==['vat.invoice_number']


def test_new_railway_invoice_preserves_separate_invoice_number_and_tax_fields():
    values={'rail.invoice_number':'12345678901234567890','rail.total':'113.00','rail.amount':'100.00'}
    lines=[{'text':v,'bbox':[i*100,10,i*100+80,35]} for i,v in enumerate(values.values())]
    payload={'document_count':1,'kind':'rail_einvoice','fields':{k:[v,i] for i,(k,v) in enumerate(values.items())}}
    result=teacher.convert(payload,lines,500,300)
    assert result['accepted'] and result['values']==values
    assert 'rail.ticket_number' not in result['values'] and 'rail.tax' not in result['values']
    assert 'teacher_amount_incomplete' in result['flags']


def test_ride_sheet_teacher_does_not_relabel_as_taxi_receipt():
    payload={'document_count':1,'kind':'ride_itinerary','fields':{'ride.total':['113.00',0]}}
    result=teacher.convert(payload,[{'text':'合计113.00','bbox':[0,0,100,20]}],500,300)
    assert result['accepted'] and result['kind']=='ride_itinerary'
    assert result['values']=={'ride.total':'113.00'}
