def _api_generate(self, messages: list, **kwargs) -> str:
    response = self._api_client.chat.completions.create(
        model=self._api_model,
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content.strip()
