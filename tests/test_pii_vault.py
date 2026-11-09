"""Юнит-тесты Reversible PII Vault (киллер-фича №5)."""

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


pii_vault = _load('pii_vault')


class TestMasking:
    def test_card_masked_and_reversed(self):
        vault = pii_vault.PiiVault()
        text = 'Карта 4276 1111 2222 3333, спишите 500 руб'
        masked = vault.mask(text)
        assert '4276' not in masked and '[CARD_1]' in masked
        assert vault.unmask(masked) == text

    def test_phone_masked_and_reversed(self):
        vault = pii_vault.PiiVault()
        text = 'Звоните на +7 913 123-45-67 после 18:00'
        masked = vault.mask(text)
        assert '913' not in masked and '[PHONE_1]' in masked
        assert vault.unmask(masked) == text

    def test_email_masked_and_reversed(self):
        vault = pii_vault.PiiVault()
        text = 'Дублируйте на ivan.petrov@example.com'
        masked = vault.mask(text)
        assert 'ivan.petrov' not in masked and '[EMAIL_1]' in masked
        assert vault.unmask(masked) == text

    def test_multiple_entities_increment_tokens(self):
        vault = pii_vault.PiiVault()
        text = '+7 913 123-45-67 и +7 914 555-66-77'
        masked = vault.mask(text)
        assert '[PHONE_1]' in masked and '[PHONE_2]' in masked

    def test_small_numbers_not_masked_as_inn(self):
        vault = pii_vault.PiiVault()
        text = 'Заказ номер 12345 стоит 5000'
        masked = vault.mask(text)
        # 5-значный номер заказа не должен превратиться в ИНН-токен
        assert masked == text

    def test_unmasked_without_tokens_returns_text(self):
        vault = pii_vault.PiiVault()
        assert vault.unmask('нет токенов') == 'нет токенов'


class TestRegistry:
    def test_registry_returns_same_vault_per_key(self):
        registry = pii_vault.PiiVaultRegistry()
        assert registry.get('chat-1') is registry.get('chat-1')
        assert registry.get('chat-1') is not registry.get('chat-2')

    def test_registry_eviction_on_overflow(self):
        registry = pii_vault.PiiVaultRegistry(max_sessions=3)
        for index in range(5):
            registry.get(f'chat-{index}')
        assert len(registry._vaults) <= 3
