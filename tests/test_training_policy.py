import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('training_policy',Path(__file__).parents[1]/'training/training_policy.py')
policy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def test_sqrt_balance_reduces_without_reversing_category_imbalance():
    weights=policy.category_weights(['common']*100+['rare'])
    assert sum(weights[:100])==pytest.approx(10)
    assert weights[-1]==1


def test_macro_selection_protects_rare_category_and_counts_false_detections():
    result={'by_class':{1:[90,100],2:[0,1]},'unmatched_by_class':{1:10,2:1},
        'recall_iou50':90/101,'fields':101,'unmatched_detections':11}
    assert policy.validation_score(result,['other.total','vat.total'],'macro-category')==pytest.approx((.89-.1)/2)
    assert policy.validation_score(result,['other.total','vat.total'],'micro')==pytest.approx((90-1.1)/101)


def test_specialist_trainer_refuses_empty_target_family():
    with pytest.raises(ValueError,match='Insufficient'):
        policy.validate_specialist_manifest({'document_profile':'rail_einvoice','records':[]})


def test_specialist_trainer_refuses_legacy_rows_even_with_many_images():
    with pytest.raises(ValueError,match='different document family'):
        policy.validate_specialist_manifest({'document_profile':'rail_einvoice',
            'records':[{'document_profile':'rail_legacy'}]*100})
