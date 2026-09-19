"""Transcript cleaning and speaker/sentence segmentation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Utterance:
    speaker: str
    text: str
    raw: str


class TranscriptCleaner:
    """
    Cleans raw meeting transcripts and segments them by speaker and sentence.
    Handles common formats:
      - "Name: text"
      - "[Name] text"
      - "Name - text"
    """

    SPEAKER_PATTERNS = [
        re.compile(r"^\[?\s*([A-Z][a-zA-Z\s\.\-']+?)\s*\]?\s*[:\-–—]\s*(.+)$"),
        re.compile(r"^([A-Z][a-zA-Z\s\.\-']+?)\s*:\s*(.+)$"),
    ]

    def __init__(self):
        self.sentence_splitter = re.compile(
            r"(?<=[.!?])\s+(?=[A-Z\"'])|(?<=[.!?])$"
        )

    def clean(self, raw: str) -> str:
        """Basic cleaning: normalize whitespace, remove timestamps, etc."""
        if not raw:
            return ""

        # Remove common timestamp patterns
        text = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", "", raw)
        text = re.sub(r"\(\d{1,2}:\d{2}\)", "", text)

        # Normalize line endings and whitespace
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove purely empty lines
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        return "\n".join(lines)

    def segment(self, raw: str) -> List[Utterance]:
        """
        Segment transcript into speaker utterances.
        Falls back to a single 'Unknown' speaker if no markers found.
        """
        cleaned = self.clean(raw)
        utterances: List[Utterance] = []
        current_speaker = "Unknown"
        buffer: List[str] = []

        for line in cleaned.split("\n"):
            speaker, text = self._parse_speaker_line(line)
            if speaker:
                # Flush previous buffer
                if buffer:
                    utterances.append(
                        Utterance(
                            speaker=current_speaker,
                            text=" ".join(buffer).strip(),
                            raw=" ".join(buffer),
                        )
                    )
                    buffer = []
                current_speaker = speaker
                buffer.append(text)
            else:
                buffer.append(line)

        if buffer:
            utterances.append(
                Utterance(
                    speaker=current_speaker,
                    text=" ".join(buffer).strip(),
                    raw=" ".join(buffer),
                )
            )

        # Further split long utterances into sentences
        final: List[Utterance] = []
        for utt in utterances:
            sentences = self._split_sentences(utt.text)
            for sent in sentences:
                if sent.strip():
                    final.append(
                        Utterance(speaker=utt.speaker, text=sent.strip(), raw=sent)
                    )
        return final

    def _parse_speaker_line(self, line: str) -> tuple[Optional[str], str]:
        for pat in self.SPEAKER_PATTERNS:
            m = pat.match(line.strip())
            if m:
                return m.group(1).strip(), m.group(2).strip()
        return None, line

    def _split_sentences(self, text: str) -> List[str]:
        # Simple but effective for meeting language
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]
