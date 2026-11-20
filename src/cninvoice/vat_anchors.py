"""Recover literal VAT values from explicit labels, independently of learned boxes.

This is a conservative fallback: no arithmetic, fuzzy digit substitution or
unlabelled biggest-number selection. Ambiguous anchors are returned separately.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from .ocr import OCRLine


@dataclass(frozen=True)
class AnchoredValue:
    value: str
    line: OCRLine
    basis: str


@dataclass(frozen=True)
class _Text:
    compact: str


@dataclass(frozen=True)
class _JoinedLabel:
    bbox: tuple[float, float, float, float]
    page: int
    compact: str = '合计'

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def cy(self):
        return (self.bbox[1] + self.bbox[3]) / 2


_NUMBER = r'-?\d[\d,]*(?:[.·]\d{1,2})?'
_MONEY = re.compile(rf'(?:CNY|RMB|[¥￥])?{_NUMBER}(?:[元圆](?:整|正)?)?', re.I)
_WRITTEN = re.compile(r'负?[零壹贰叁肆伍陆柒捌玖拾佰仟万亿]+[元圆]'
                      r'(?:[零壹贰叁肆伍陆柒捌玖]角)?(?:零?[壹贰叁肆伍陆柒捌玖]分)?[整正]?')


def anchored_vat_fields(lines: Sequence[OCRLine], decode: Callable) -> tuple[dict[str, AnchoredValue], set[str]]:
    """Return unambiguous labelled fields and conflicting field names.

    ``decode`` is the same literal money decoder used by the visual reader.
    Column headings, when unique, provide the slope of a photographed table row.
    """
    found: dict[str, list[AnchoredValue]] = {k: [] for k in ('amount', 'tax', 'total', 'invoice_number')}
    conflicts: set[str] = set()

    def literal(text, field):
        text = text.replace('−', '-').replace('－', '-')
        text = text.strip('()（）:：⊗× ')
        text = re.sub(r'^[.·](?=[¥￥])', '', text)
        if not (_MONEY.fullmatch(text) or _WRITTEN.fullmatch(text)):
            return None
        values = set(decode('vat.' + field, _Text(text)))
        return next(iter(values)) if len(values) == 1 else None

    for page in sorted({line.page for line in lines}):
        rows = [line for line in lines if line.page == page]
        headers = {name: [line for line in rows if line.compact == text]
                   for name, text in [('amount', '金额'), ('tax', '税额')]}
        slope = 0.
        if all(len(headers[k]) == 1 for k in headers):
            a, b = headers['amount'][0], headers['tax'][0]
            dx = (b.bbox[0] + b.bbox[2] - a.bbox[0] - a.bbox[2]) / 2
            if dx > max(a.height, b.height):
                candidate = (b.cy - a.cy) / dx
                if abs(candidate) <= .15:
                    slope = candidate

        def same_row(a, b):
            dx = (b.bbox[0] + b.bbox[2] - a.bbox[0] - a.bbox[2]) / 2
            return abs(b.cy - a.cy - slope * dx) <= .75 * max(a.height, b.height, 1)

        # Printed VAT tables often space these two characters across a cell.
        # Join only a unique, aligned pair; a lone 计 never becomes a label.
        if not any(line.compact.rstrip(':：') == '合计' for line in rows):
            left = [line for line in rows if line.compact == '合']
            right = [line for line in rows if line.compact == '计']
            if len(left) == len(right) == 1:
                a, b = left[0], right[0]
                gap = b.bbox[0] - a.bbox[2]
                if 0 <= gap <= 12 * max(a.height, b.height) and same_row(a, b):
                    rows.append(_JoinedLabel((a.bbox[0], min(a.bbox[1], b.bbox[1]),
                                              b.bbox[2], max(a.bbox[3], b.bbox[3])), page))

        def right_values(label, field, written=False):
            choices = []
            for line in rows:
                if line is label or line.bbox[0] < label.bbox[2] - .5 * label.height:
                    continue
                if not same_row(label, line):
                    continue
                text = line.compact.strip('⊗× ')
                if written and not _WRITTEN.fullmatch(text):
                    continue
                value = literal(text, field)
                if value is not None:
                    choices.append(AnchoredValue(value, line, 'label_same_row'))
            return choices

        for label in rows:
            text = label.compact
            number = re.fullmatch(r'(?:发票号码|No\.?)[:：]?(\d{8}|\d{20})', text, re.I)
            if number:
                found['invoice_number'].append(AnchoredValue(number.group(1), label, 'explicit_invoice_number'))
            elif re.fullmatch(r'(?:发票号码|No\.?)[：:]?', text, re.I):
                found['invoice_number'].extend(right_values(label, 'invoice_number'))

        lower = [line for line in rows if '小写' in line.compact]
        lower_values, upper_values = [], []
        # Repeated total labels on one page may represent multiple invoices.
        if len(lower) > 1:
            conflicts.add('total')
        for label in lower:
            suffix = label.compact.split('小写', 1)[1]
            value = literal(suffix, 'total')
            if value is not None:
                lower_values.append(AnchoredValue(value, label, 'explicit_lowercase_total'))
            else:
                lower_values.extend(right_values(label, 'total'))
        for label in rows:
            text = label.compact
            if '价税合计' in text and '大写' in text:
                suffix = text.split('大写', 1)[1].split('小写', 1)[0]
                value = literal(suffix, 'total')
                if value is not None:
                    upper_values.append(AnchoredValue(value, label, 'explicit_written_total'))
                else:
                    upper_values.extend(right_values(label, 'total', written=True))
            elif text.startswith('价税合计') and '小写' not in text:
                value = literal(text[len('价税合计'):], 'total')
                if value is not None:
                    lower_values.append(AnchoredValue(value, label, 'explicit_total'))
                elif text.rstrip(':：') == '价税合计':
                    lower_values.extend(right_values(label, 'total'))
        found['total'].extend(lower_values + upper_values)

        for label in rows:
            text = label.compact
            for field, names in [('amount', r'不含税金额|金额\(不含税\)|合计金额|金额合计'),
                                 ('tax', r'合计税额|税额合计')]:
                match = re.fullmatch(r'(?:' + names + r')[:：]?(.*)', text)
                if match:
                    value = literal(match.group(1), field)
                    if value is not None:
                        found[field].append(AnchoredValue(value, label, 'explicit_' + field))
                    else:
                        found[field].extend(right_values(label, field))
            combined = re.fullmatch(r'合计[:：]?([¥￥]' + _NUMBER + r')([¥￥]' + _NUMBER + r')', text)
            if combined:
                for field, part in zip(('amount', 'tax'), combined.groups()):
                    value = literal(part, field)
                    if value is not None:
                        found[field].append(AnchoredValue(value, label, 'explicit_aggregate_row'))
            if text.rstrip(':：') != '合计':
                continue
            # Require currency-marked values. Bare quantities and tax rates must
            # not become an amount/tax pair when a tax cell says exempt or ***.
            choices = [c for c in right_values(label, 'amount')
                       if re.match(r'^(?:CNY|RMB|[¥￥])', c.line.compact, re.I)]
            if len(choices) == 2:
                choices.sort(key=lambda c: c.line.bbox[0])
                left, right = choices[0].line, choices[1].line
                if left.bbox[2] > right.bbox[0] or not same_row(left, right):
                    continue
                for field, candidate in zip(('amount', 'tax'), choices):
                    basis = 'split_aggregate_row' if isinstance(label, _JoinedLabel) else 'explicit_aggregate_row'
                    found[field].append(AnchoredValue(candidate.value, candidate.line, basis))
            elif len(choices) == 1 and all(len(headers[k]) == 1 for k in ('amount', 'tax')):
                # An exempt/star tax cell is not numeric. Read the printed amount
                # independently only when both column headings establish its role.
                a, b = headers['amount'][0], headers['tax'][0]
                ax = (a.bbox[0] + a.bbox[2]) / 2
                bx = (b.bbox[0] + b.bbox[2]) / 2
                candidate = choices[0]
                cx = (candidate.line.bbox[0] + candidate.line.bbox[2]) / 2
                if bx > ax and ax - .5 * (bx - ax) <= cx < (ax + bx) / 2:
                    found['amount'].append(AnchoredValue(candidate.value, candidate.line,
                                                        'explicit_aggregate_amount_column'))

    selected = {}
    for field, candidates in found.items():
        values = {c.value for c in candidates}
        if len(values) > 1:
            conflicts.add(field)
        if len(values) == 1 and field not in conflicts:
            selected[field] = candidates[0]
    return selected, conflicts
