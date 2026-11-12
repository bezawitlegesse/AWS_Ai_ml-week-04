"""Small fake components so the pipeline can be tested offline (no API key, no model downloads)."""
import json
import re
import zlib

import numpy as np


STOPWORDS = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "what", "who", "how", "do", "does", "can",
             "i", "it", "you", "for", "or", "that", "this", "with", "be", "by", "on", "as", "at", "if", "when"}


class HashEmbedder:
    """Bag-of-words hashing embedder: texts that share (non-stop) words get similar vectors."""
    DIM = 8192

    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = np.zeros(self.DIM)
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if word not in STOPWORDS:
                    vector[zlib.crc32(word.encode()) % self.DIM] += 1.0
            vectors.append(vector.tolist())
        return vectors


class ScriptedLLM:
    """Returns the scripted replies in order (an Exception instance in the list is raised)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, json_mode=False):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class ExtractiveLLM:
    """Answers with the first sentence of the first context block, so whole-pipeline tests can run."""

    def generate(self, prompt, json_mode=False):
        if "Standalone question:" in prompt:
            return prompt.rsplit("Last question:", 1)[1].split("\n")[0].strip()
        context = re.findall(r"<context>\n(.*?)\n</context>", prompt, re.S)[-1]
        match = re.match(r"\[Page (\d+)\] (.*?)(?:\. |\n|$)", context, re.S)
        page, sentence = int(match.group(1)), match.group(2).strip()
        return json.dumps({"answer": sentence + ".", "found": True, "citations": [page], "confidence": "medium"})


class ApiError(Exception):
    """Looks like google.genai errors: has a numeric .code."""

    def __init__(self, code, message="api error"):
        super().__init__(message)
        self.code = code
