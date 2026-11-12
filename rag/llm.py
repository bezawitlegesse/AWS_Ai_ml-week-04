"""Gemini client with retries, plus friendly explanations for common API errors."""
import os

from .retry import call_with_retries


class MissingKeyError(RuntimeError):
    pass


def get_api_key():
    """Read the key from the environment (or Colab Secrets). Never hard-code it."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        try:
            from google.colab import userdata
            key = userdata.get("GEMINI_API_KEY")
        except Exception:
            key = None
    if not key:
        raise MissingKeyError(
            "No API key found. Set the GEMINI_API_KEY environment variable "
            "(see .env.example) or add it to Colab Secrets.")
    return key


def explain_error(exc):
    """Turn an API exception into a short message a student can act on."""
    code = getattr(exc, "code", None)
    text = str(exc)
    lowered = text.lower()
    if isinstance(exc, MissingKeyError):
        return text
    if code in (401, 403) or "api key" in lowered:
        return "The API key was rejected. Check GEMINI_API_KEY."
    if code == 404 or "is not found" in lowered:
        return "Model not found. Run `python app.py --list-models` and set GEMINI_MODEL to one of them."
    if code == 429:
        return "Rate limit or quota reached. Wait a minute and try again."
    if code in (500, 502, 503, 504):
        return "The AI service is busy right now. Try again in a moment."
    return f"{type(exc).__name__}: {text[:200]}"


class GeminiClient:
    def __init__(self, api_key, model, temperature=0.2, max_retries=4, on_retry=None):
        from google import genai
        from google.genai import types
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.on_retry = on_retry

    def generate(self, prompt, json_mode=False):
        """Send one prompt and return the reply text. Temporary errors are retried."""
        config = self._types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json" if json_mode else None)

        def call():
            response = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
            return response.text

        text = call_with_retries(call, max_attempts=self.max_retries, on_retry=self.on_retry)
        if not text:
            raise RuntimeError("The model returned an empty reply.")
        return text

    def list_models(self):
        return [m.name for m in self._client.models.list()]
