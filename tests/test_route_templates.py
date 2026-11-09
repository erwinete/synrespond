"""Юнит-тесты чистых модулей форка: шаблоны маршрутизации (Д1, Д2).

Модуль загружается напрямую по пути файла — без зависимостей open_webui.
Запуск: .venv-encoder/Scripts/python -m pytest tests/ -q  (из корня монорепозитория)
"""

import importlib.util
import sys
from pathlib import Path

FORK_UTILS = Path(__file__).resolve().parents[1] / 'open-webui-fork' / 'backend' / 'open_webui' / 'utils'


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, FORK_UTILS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


route_templates = _load('route_templates')


class TestRenderRouteTemplate:
    def test_user_message_and_intent(self):
        result = route_templates.render_route_template(
            'Вы обратились по теме {{intent}}. Ваш запрос: {{user_message}}',
            user_message='Где заказ?',
            intent='order_status',
        )
        assert result == 'Вы обратились по теме order_status. Ваш запрос: Где заказ?'

    def test_slots_substitution(self):
        result = route_templates.render_route_template(
            'Заказ {{slots.order_id}} вернем, причина: {{slots.reason}}',
            user_message='x',
            intent='t',
            slots={'order_id': '94821', 'reason': 'брак'},
        )
        assert result == 'Заказ 94821 вернем, причина: брак'

    def test_missing_slots_are_cleaned(self):
        result = route_templates.render_route_template(
            'Заказ {{slots.order_id}}, сумма {{slots.amount}}',
            user_message='x',
            intent='t',
            slots={'order_id': '123'},
        )
        assert result == 'Заказ 123, сумма '
        assert '{{' not in result

    def test_extra_variables(self):
        result = route_templates.render_route_template(
            'Гнев: {{voice.anger_score}}', user_message='x', intent='t', extra={'voice.anger_score': '0.82'}
        )
        assert result == 'Гнев: 0.82'

    def test_unknown_token_preserved(self):
        result = route_templates.render_route_template('Keep {{unknown_token}}', user_message='x', intent='t')
        assert result == 'Keep {{unknown_token}}'

    def test_webhook_body_rendering_via_json(self):
        # Вебхук: шаблон тела проходит через json.dumps -> render -> json.loads
        import json

        body = {'order_id': '{{slots.order_id}}', 'message': '{{user_message}}'}
        rendered = route_templates.render_route_template(
            json.dumps(body, ensure_ascii=False),
            user_message='вернуть товар',
            intent='return',
            slots={'order_id': '777'},
        )
        parsed = json.loads(rendered)
        assert parsed == {'order_id': '777', 'message': 'вернуть товар'}


class TestDisambiguation:
    def test_gray_zone_close_candidates_need_disambiguation(self):
        candidates = [
            {'intent_name': 'billing_refund', 'confidence': 0.62},
            {'intent_name': 'order_status', 'confidence': 0.55},
        ]
        needed, top_1, top_2, delta = route_templates.should_disambiguate(candidates, 0.15)
        assert needed and top_1 == 'billing_refund' and top_2 == 'order_status' and abs(delta - 0.07) < 1e-9

    def test_clear_leader_skips_disambiguation(self):
        candidates = [
            {'intent_name': 'greeting', 'confidence': 0.75},
            {'intent_name': 'smalltalk', 'confidence': 0.40},
        ]
        needed, _, _, delta = route_templates.should_disambiguate(candidates, 0.15)
        assert not needed and delta > 0.15

    def test_single_candidate_never_disambiguates(self):
        needed, _, _, delta = route_templates.should_disambiguate(
            [{'intent_name': 'a', 'confidence': 0.5}], 0.15
        )
        assert not needed and delta == 1.0

    def test_prompt_contains_both_candidates(self):
        candidates = [
            {'intent_name': 'billing_refund', 'confidence': 0.60},
            {'intent_name': 'order_status', 'confidence': 0.58},
        ]
        prompt = route_templates.build_disambiguation_prompt(candidates, 0.15)
        assert prompt and 'billing_refund' in prompt and 'order_status' in prompt


class TestSlotsFromSpans:
    def test_spans_to_dict(self):
        spans = [
            {'label': 'order_id', 'text': '94821', 'start': 0, 'end': 5, 'confidence': 0.99},
            {'label': 'PERSON', 'text': 'Иванов Иван', 'start': 6, 'end': 17, 'confidence': 0.95},
        ]
        assert route_templates.slots_from_spans(spans) == {'order_id': '94821', 'PERSON': 'Иванов Иван'}

    def test_empty_and_broken_spans(self):
        assert route_templates.slots_from_spans([]) == {}
        assert route_templates.slots_from_spans([{'label': '', 'text': 'x'}, {'label': 'a', 'text': ''}]) == {}
