import importlib.util
from pathlib import Path


spec=importlib.util.spec_from_file_location('prepare_data',Path(__file__).parents[1]/'training/prepare_data.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)


def test_publisher_ticket_labels_map_to_canonical_fields_without_invoice_number():
    rows='0,1,2,31,2,31,12,1,12,A123456,ticket_num\n1,50,20,100,20,100,40,50,40,￥113.5元,ticket_rates\n'
    result=prepare.parse_pick(rows)
    assert [r['field'] for r in result]==['rail.ticket_number','rail.total']
    assert result[0]['box']==[1,2,31,12] and result[1]['text']=='￥113.5元'


def test_multiple_photographs_of_same_ticket_share_a_split():
    first=[{'field':'rail.ticket_number','text':'A123456'}]
    second=[{'field':'rail.ticket_number','text':' A123456 '},{'field':'rail.total','text':'113.00'}]
    a=prepare.group_key(first,'photo1');b=prepare.group_key(second,'photo2')
    assert a==b and prepare.split_group(a)==prepare.split_group(b)


def test_unrelated_ocr_text_is_not_promoted_to_field_label():
    assert prepare.parse_pick('0,1,2,31,2,31,12,1,12,999.00,other\n')==[]
