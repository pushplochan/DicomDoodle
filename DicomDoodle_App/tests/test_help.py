"""
Tests for the help content: the User Guide chapters, their markup and the
references.

    python -m pytest tests          (or simply: python tests/test_help.py)
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dicomdoodle import resources  # noqa: E402
from dicomdoodle.help import contents  # noqa: E402
from dicomdoodle.help.references import all_references, app_reference, bibtex_file  # noqa: E402
from dicomdoodle.ui.guide.markup import Renderer, plain_text  # noqa: E402
from dicomdoodle.ui.theme import DARK  # noqa: E402

# links of the form action:<name> that the main window knows (ui/main/tools.py, guide_action)
ACTIONS = {"open", "overlay", "viewer3d", "mpr", "converter", "preferences",
           "shortcuts", "references", "about", "add_label", "theme"}


def test_chapters_and_links():
    chapters = contents.all_chapters()
    ids = [c.id for c in chapters]
    assert len(ids) == len(set(ids)), "chapter ids must be unique"
    assert contents.HOME in ids
    for chapter in chapters:
        for kind, target in re.findall(r"\]\((chapter|action|shot):([\w-]+)\)", chapter.body):
            if kind == "chapter":
                assert target in ids, f"{chapter.id}: link to unknown chapter {target}"
            elif kind == "action":
                assert target in ACTIONS, f"{chapter.id}: unknown action {target}"
            else:
                path = os.path.join(resources.GUIDE_DIR, f"{target}.png")
                assert os.path.isfile(path), f"{chapter.id}: screenshot {target}.png is missing"
    first, last = chapters[0], chapters[-1]
    assert contents.neighbours(first.id)[0] is None and contents.neighbours(last.id)[1] is None
    for chapter in chapters:                  # the synthetic demo series is gone; the guide shows real scans
        assert "demo" not in (chapter.summary + chapter.body).lower(), chapter.id


def test_markup_renders_every_chapter():
    renderer = Renderer(DARK, 760, lambda source: (760, 400))
    for chapter in contents.all_chapters():
        html = renderer.chapter("Part", chapter.title, chapter.summary, chapter.body)
        text = re.sub(r"<[^>]+>", "", html)
        for leftover in ("**", "[[", "]]", "{icon:", "{n:", "{color:", "](", "\\|"):
            assert leftover not in text, f"{chapter.id}: unrendered markup {leftover!r}"
        assert plain_text(chapter.body).strip()


def test_markup_details():
    renderer = Renderer(DARK, 600, lambda source: (1200, 600))
    rows = "".join(renderer.blocks("| Key | Label |\n|---|---|\n| [[1]] | Tumor \\| Brain |"))
    assert "Tumor | Brain" in rows and rows.count("<tr>") == 2
    image = "".join(renderer.blocks("![A caption](shot:overview)"))
    assert 'src="shot:overview@600~dark"' in image and 'height="300"' in image and "zoom:overview" in image
    assert "<b>bold</b>" in renderer.inline("**bold**") and "<i>it</i>" in renderer.inline("*it*")


def test_references():
    refs = all_references()
    assert len({r.key for r in refs}) == len(refs), "BibTeX keys must be unique"
    for ref in refs:
        assert ref.link.startswith("https://"), ref.key
        assert ref.text().startswith(ref.author_text())
    bib = bibtex_file()
    assert bib.count("{") == bib.count("}")
    assert bib.startswith("@") and bib.count("\n@") == len(refs)
    assert "DicomDoodle" in app_reference().text()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"ok  {name}")
