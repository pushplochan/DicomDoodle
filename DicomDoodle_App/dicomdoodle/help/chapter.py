"""One chapter of the User Guide."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter:
    id: str            # used in links: [text](chapter:id)
    title: str
    icon: str          # icon name from ui/icons.py
    summary: str       # one line under the title
    body: str          # guide markup (see ui/guide/markup.py)
    keywords: str = "" # extra words for the search box
    short: str = ""    # title in the guide's sidebar, when the full title is long

    @property
    def nav_title(self) -> str:
        return self.short or self.title
