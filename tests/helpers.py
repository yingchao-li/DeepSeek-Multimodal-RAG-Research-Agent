from types import SimpleNamespace


class FakeClient:
    def __init__(self, replies=None):
        self.calls = []
        self.replies = iter(replies or ["mock answer"] * 30)
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=next(self.replies)))], usage=SimpleNamespace(prompt_tokens=12, completion_tokens=6))
