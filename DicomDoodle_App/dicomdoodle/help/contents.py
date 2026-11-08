"""The User Guide's table of contents: parts and their chapters, in reading order."""
from typing import List, Optional, Tuple

from . import part_annotate, part_reference, part_review, part_save, part_start, part_tools
from .chapter import Chapter

GROUPS: List[Tuple[str, List[Chapter]]] = [
    ("Getting started", part_start.CHAPTERS),
    ("Annotating", part_annotate.CHAPTERS),
    ("Saving & batch work", part_save.CHAPTERS),
    ("Reviewing & measuring", part_review.CHAPTERS),
    ("Advanced tools", part_tools.CHAPTERS),
    ("Reference", part_reference.CHAPTERS),
]

HOME = "welcome"


def all_chapters() -> List[Chapter]:
    return [chapter for _group, chapters in GROUPS for chapter in chapters]


def find(chapter_id: str) -> Optional[Chapter]:
    return next((c for c in all_chapters() if c.id == chapter_id), None)


def group_of(chapter_id: str) -> str:
    return next((g for g, chapters in GROUPS for c in chapters if c.id == chapter_id), "")


def neighbours(chapter_id: str) -> Tuple[Optional[Chapter], Optional[Chapter]]:
    chapters = all_chapters()
    ids = [c.id for c in chapters]
    if chapter_id not in ids:
        return None, None
    i = ids.index(chapter_id)
    return (chapters[i - 1] if i > 0 else None), (chapters[i + 1] if i + 1 < len(chapters) else None)
