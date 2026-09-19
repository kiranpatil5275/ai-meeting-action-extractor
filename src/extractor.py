"""
Hybrid action-item extraction pipeline.

Strategy (fully offline after first model download):
1. Segment transcript (cleaner)
2. spaCy NER → people & dates
3. Rule-based candidate detection (imperative verbs, assignment patterns)
4. Zero-shot classification (BART-MNLI) to confirm action items
5. Heuristic assembly of task / owner / deadline / confidence
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import spacy
from transformers import pipeline

from .cleaner import TranscriptCleaner, Utterance
from .utils import parse_date, normalize_name, clean_task_text, extract_person_candidates

logger = logging.getLogger(__name__)


@dataclass
class ActionItem:
    task: str
    owner: Optional[str]
    deadline: Optional[str]
    status: str = "pending"
    confidence: float = 0.0
    source_sentence: str = ""
    issues: List[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class ActionItemExtractor:
    """
    Main extractor. Loads models lazily.
    """

    # Strong signals that a sentence is an action item
    ACTION_PATTERNS = [
        re.compile(
            r"\b(can you|could you|please|I need you to|you should|you will|"
            r"I'll|I will|we'll|we will|let's|make sure|need to|have to|"
            r"responsible for|own this|take this|handle this)\b",
            re.I,
        ),
        re.compile(
            r"\b(by|before|due|deadline|until)\s+(next|this|the|monday|tuesday|"
            r"wednesday|thursday|friday|saturday|sunday|end of|"
            r"\d{1,2}[\/\-]\d{1,2}|\w+\s+\d{1,2})",
            re.I,
        ),
    ]

    IMPERATIVE_VERBS = {
        "prepare", "schedule", "update", "send", "review", "finalize",
        "order", "book", "create", "assign", "fix", "write", "deploy",
        "notify", "launch", "coordinate", "set", "complete", "submit",
        "organize", "migrate", "train", "validate", "design", "conduct",
        "implement", "close", "reconcile", "forecast", "ship", "run",
        "negotiate", "get", "survey", "draft", "investigate", "propose",
        "share", "document", "select", "collect", "add", "refactor",
        "produce", "identify", "right-size", "prioritize", "publish",
        "sponsor", "approve", "implement", "create", "finalize",
        "schedule", "draft", "review", "prepare", "launch",
    }

    def __init__(
        self,
        zero_shot_model: str = "facebook/bart-large-mnli",
        spacy_model: str = "en_core_web_sm",
        confidence_threshold: float = 0.40,
        owner_boost: float = 0.12,
        deadline_boost: float = 0.12,
        rule_boost: float = 0.08,
        device: int = -1,  # -1 = CPU
        use_zero_shot: bool = True,
    ):
        """
        Hyperparameters (tunable from UI or eval):
          confidence_threshold – minimum zero-shot score to keep a candidate
          owner_boost          – extra confidence when an owner is found
          deadline_boost       – extra confidence when a deadline is found
          rule_boost           – extra confidence when strong rule patterns match
          use_zero_shot        – set False for pure rule-based (faster, no model)
        """
        self.confidence_threshold = confidence_threshold
        self.owner_boost = owner_boost
        self.deadline_boost = deadline_boost
        self.rule_boost = rule_boost
        self.use_zero_shot = use_zero_shot
        self.device = device
        self.cleaner = TranscriptCleaner()

        # Lazy-loaded models
        self._nlp = None
        self._zs = None
        self._spacy_model = spacy_model
        self._zero_shot_model = zero_shot_model

    @property
    def nlp(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self._spacy_model)
            except OSError:
                logger.warning(
                    "spaCy model not found. Run: python -m spacy download en_core_web_sm"
                )
                # Minimal fallback
                self._nlp = spacy.blank("en")
        return self._nlp

    @property
    def zero_shot(self):
        if self._zs is None:
            logger.info("Loading zero-shot model (first run downloads ~1.6 GB)...")
            self._zs = pipeline(
                "zero-shot-classification",
                model=self._zero_shot_model,
                device=self.device,
            )
        return self._zs

    def extract(self, transcript: str) -> List[ActionItem]:
        """Main entry point."""
        utterances = self.cleaner.segment(transcript)
        candidates: List[ActionItem] = []

        for utt in utterances:
            items = self._extract_from_utterance(utt)
            candidates.extend(items)

        # Deduplicate roughly
        unique = self._deduplicate(candidates)
        return unique

    def _extract_from_utterance(self, utt: Utterance) -> List[ActionItem]:
        text = utt.text
        if len(text) < 12:
            return []

        # Quick filter
        rule_match = self._looks_like_action(text)
        if not rule_match and not self.use_zero_shot:
            return []

        # Zero-shot confirmation (optional)
        if self.use_zero_shot:
            zs_score = self._zero_shot_score(text)
            if zs_score < self.confidence_threshold and not rule_match:
                return []
        else:
            zs_score = 0.55 if rule_match else 0.2
            if not rule_match:
                return []

        # Extract entities
        doc = self.nlp(text)
        people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

        # Also use heuristic name extraction
        people.extend(extract_person_candidates(text))
        people = list(dict.fromkeys([normalize_name(p) for p in people if p]))

        # Owner resolution (improved)
        owner = self._resolve_owner(text, utt.speaker, people)

        # Deadline
        deadline = None
        for d in dates:
            parsed = parse_date(d)
            if parsed:
                deadline = parsed
                break
        if deadline is None:
            deadline = parse_date(text)

        # Task formulation
        task = self._formulate_task(text, owner)

        # Tunable confidence combination
        conf = zs_score * 0.65
        if owner:
            conf += self.owner_boost
        if deadline:
            conf += self.deadline_boost
        if rule_match:
            conf += self.rule_boost
        conf = min(0.97, conf)

        item = ActionItem(
            task=task,
            owner=owner,
            deadline=deadline,
            status="pending",
            confidence=round(conf, 3),
            source_sentence=text,
        )
        return [item]

    def _looks_like_action(self, text: str) -> bool:
        lower = text.lower()
        for pat in self.ACTION_PATTERNS:
            if pat.search(lower):
                return True
        # Check for imperative verbs at start or after "I'll/can you"
        tokens = lower.split()
        for t in tokens[:6]:
            if t.rstrip(".,") in self.IMPERATIVE_VERBS:
                return True
        return False

    def _zero_shot_score(self, text: str) -> float:
        try:
            result = self.zero_shot(
                text,
                candidate_labels=["action item / task assignment", "discussion / statement"],
                multi_label=False,
            )
            # Probability of the action-item label
            labels = result["labels"]
            scores = result["scores"]
            for lab, sc in zip(labels, scores):
                if "action" in lab.lower():
                    return float(sc)
            return 0.0
        except Exception as e:
            logger.warning("Zero-shot failed: %s", e)
            # Fallback heuristic score
            return 0.55 if self._looks_like_action(text) else 0.2

    def _resolve_owner(
        self, text: str, speaker: str, people: List[str]
    ) -> Optional[str]:
        lower = text.lower()

        # 1. Explicit address patterns: "Name, can you..." / "Name, please..."
        for p in people:
            pl = p.lower()
            if re.search(rf"\b{re.escape(pl)}\b\s*[,:]?\s*(can you|could you|please|will you)", lower):
                return p
            if re.search(rf"(can you|could you|please)\s+{re.escape(pl)}\b", lower):
                return p

        # 2. "Name will / Name to / Name needs to"
        for p in people:
            pl = p.lower()
            if re.search(rf"\b{re.escape(pl)}\b\s+(will|to|needs? to|should|must)", lower):
                return p

        # 3. Name appears early in the sentence (addressed party)
        for p in people:
            idx = lower.find(p.lower())
            if 0 <= idx < max(25, len(text) * 0.35):
                return p

        # 4. Self-assignment
        if re.search(r"\b(I'll|I will|I'm going to|I can|I need to)\b", text, re.I):
            if speaker and speaker != "Unknown":
                return normalize_name(speaker)

        # 5. Fallback: first person mentioned
        if people:
            return people[0]

        return None

    def _formulate_task(self, text: str, owner: Optional[str]) -> str:
        # Remove the owner name and common assignment phrases
        task = text
        if owner:
            task = re.sub(re.escape(owner), "", task, flags=re.I)
        task = re.sub(
            r"\b(can you|could you|please|I need you to|you should|"
            r"I'll|I will|we'll|we will|let's)\b",
            "",
            task,
            flags=re.I,
        )
        task = re.sub(
            r"\b(by|before|due|deadline|until)\s+.{0,30}$",
            "",
            task,
            flags=re.I,
        )
        task = clean_task_text(task)
        # Limit length
        if len(task) > 120:
            task = task[:117] + "..."
        return task or text[:80]

    def _deduplicate(self, items: List[ActionItem]) -> List[ActionItem]:
        """Remove near-duplicate tasks (same owner + similar task)."""
        from .utils import similarity

        unique: List[ActionItem] = []
        for item in items:
            is_dup = False
            for existing in unique:
                if (
                    existing.owner
                    and item.owner
                    and existing.owner.lower() == item.owner.lower()
                    and similarity(existing.task, item.task) > 85
                ):
                    is_dup = True
                    # Keep higher confidence
                    if item.confidence > existing.confidence:
                        unique.remove(existing)
                        unique.append(item)
                    break
            if not is_dup:
                unique.append(item)
        return unique
