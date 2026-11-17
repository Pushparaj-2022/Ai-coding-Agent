from __future__ import annotations

from typing import TYPE_CHECKING

from lca.providers.base import Provider

if TYPE_CHECKING:
    from lca.config import ProviderSettings


def create_provider(settings: ProviderSettings) -> Provider:
    if settings.kind == "ollama":
        from lca.providers.ollama import OllamaProvider

        return OllamaProvider(settings)
    from lca.providers.openai_compatible import OpenAICompatibleProvider

    return OpenAICompatibleProvider(settings)
