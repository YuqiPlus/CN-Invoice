import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('pseudo_label',Path(__file__).parents[1]/'training/pseudo_label.py')
pseudo=importlib.util.module_from_spec(spec);spec.loader.exec_module(pseudo)


def fixture():
    lines=[{'text':'￥100.00','bbox':[500,600,650,630],'score':.99},
           {'text':'￥13.00','bbox':[800,600,950,630],'score':.99},
           {'text':'￥113.00','bbox':[750,700,950,730],'score':.99},
           {'text':'No 12345678','bbox':[700,100,950,130],'score':.99}]
    payload={'document_count':1,'kind':'vat_invoice','fields':{
        name:{'value':value,'ocr_id':i,'confidence':.99} for i,(name,value) in enumerate([
            ('amount','100.00'),('tax','13.00'),('total','113.00'),('invoice_number','12345678')])}}
    return payload,lines


def test_teacher_values_must_have_literal_printed_boxes():
    payload,lines=fixture();result=pseudo.qualify(payload,lines,1000,1000)
    assert result['accepted'] and len(result['annotations'])==4
    assert result['label_quality']=='auto_verified_pseudo_not_human_ground_truth'
    payload['fields']['total']['value']='114.00'
    assert not pseudo.qualify(payload,lines,1000,1000)['accepted']


def test_decimal_string_box_id_is_normalized_without_changing_teacher_value():
    payload,lines=fixture()
    for field in payload['fields'].values():field['ocr_id']=str(field['ocr_id'])
    result=pseudo.qualify(payload,lines,1000,1000)
    assert result['accepted'] and result['annotations'][0]['ocr_id']==0


def test_arithmetic_consistency_without_matching_printed_total_is_rejected():
    payload,lines=fixture();payload['fields']['amount']['value']='113.00';payload['fields']['amount']['ocr_id']=2
    payload['fields']['total']['value']='126.00'
    result=pseudo.qualify(payload,lines,1000,1000)
    assert not result['accepted'] and 'total:not_unique_literal_money' in result['reasons']


def test_heldout_invoice_identity_never_becomes_training_label():
    payload,lines=fixture();result=pseudo.qualify(payload,lines,1000,1000,{'12345678'})
    assert not result['accepted'] and 'heldout_invoice_identity' in result['reasons']


def test_detail_row_not_silently_substituted_for_aggregate():
    payload,lines=fixture();lines.append({'text':'￥100.00','bbox':[500,500,650,530],'score':.99})
    payload['fields']['amount']['ocr_id']=4
    result=pseudo.qualify(payload,lines,1000,1000)
    assert not result['accepted'] and 'amount:possible_detail_instead_of_aggregate' in result['reasons']


def test_two_amounts_in_one_text_box_are_not_accepted_as_separate_fields():
    payload,lines=fixture();lines[0]['text']='￥100.00 ￥13.00'
    assert not pseudo.qualify(payload,lines,1000,1000)['accepted']
