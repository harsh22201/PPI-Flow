import time
from typing import List, Dict
from groq import Groq
from requests.exceptions import RequestException


class LM:
    """
    A class for making language model calls to Groq API with retries
    and API key rotation.
    """

    def __init__(
        self,
        model: str,
        api_keys: List[str]= [
                                "groq_api_key_1",
                                "groq_api_key_2",
                                ],
        retries: int = 1,
        temperature: float = 0.1,
        calls_per_key: int = 3,
    ):
        """
        Args:
            model (str): Model name to use
            api_keys (List[str]): List of Groq API keys
            retries (int): Number of retry attempts
            temperature (float): Sampling temperature
            calls_per_key (int): Number of calls before rotating API key
        """
        if not api_keys:
            raise ValueError("api_keys list cannot be empty")

        self.model = model
        self.api_keys = api_keys
        self.calls_per_key = calls_per_key
        self.retries = retries
        self.temperature = temperature

        self._key_index = 0
        self._call_count = 0
        self.client = Groq(api_key=self.api_keys[self._key_index])

    def _rotate_key_if_needed(self):
        if self._call_count >= self.calls_per_key:
            self._call_count = 0
            self._key_index = (self._key_index + 1) % len(self.api_keys)
            self.client = Groq(api_key=self.api_keys[self._key_index])

    def get_model(self) -> str:
        return self.model

    def __call__(self, messages) -> str:
        """
        Make a chat completion request to Groq API.

        Args:
            messages (List[Dict[str, str]]): Chat messages

        Returns:
            str: Model response
        """
        for attempt in range(1, self.retries + 1):
            try:
                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )

                response = chat_completion.choices[0].message.content or ""
                self._call_count += 1
                self._rotate_key_if_needed()
                return response.strip()

            except RequestException as e:
                print(f"(attempt {attempt}/{self.retries}) Request error: {e}")
            except Exception as e:
                print(f"(attempt {attempt}/{self.retries}) Unexpected error: {e}")

            if attempt < self.retries:
                time.sleep(1)

        raise Exception(
            f"Failed to get response from Groq API after {self.retries} retries."
        )


if __name__ == "__main__":
    # Example usage
    lm = LM(model="openai/gpt-oss-120b")
    messages = [
        {"role": "user", "content": "Carbocation having more stability is : A . \( 1^{0} \) carbocation B. \( 2^{0} \) carbocation \( \mathrm{c} \cdot 3^{0} \) carbocation D. none which chapter in ncert it belongs to"}, # type: ignore
    ]
    response = lm(messages)
    print("Model response:", response)