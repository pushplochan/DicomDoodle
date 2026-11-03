"""
Turns a guide chapter written in a small markup into HTML for QTextBrowser,
colored with the current theme.

Blocks (one per line):
    ## Heading   /   ### Subheading
    plain lines              paragraph (a blank line ends it)
    1. text                  numbered step
    - text                   bullet
    > TIP: text              callout box: TIP, NOTE, IMPORTANT or TRY IT
                             (following "> " lines continue the same box)
    ![Caption](shot:name)    screenshot (dicomdoodle/assets/guide/name.png);
                             clicking it opens it full size
    ![Caption](diagram:x)    diagram drawn by ui/guide/diagrams.py
    | a | b |                table row (the first row is the header)

Inline:
    **bold**  *italic*  `file name`  [[Ctrl+S]] key  {icon:save} toolbar icon
    {n:3} numbered marker (as drawn on the screenshots)   {color:FF0000} swatch
    [text](chapter:id)  [text](action:name)  [text](https://...)
    \\| \\[ \\] \\* ...         a literal character (e.g. a "|" inside a table cell)

Images are referenced by URL (e.g. ``shot:overview@760~dark``); the guide
window supplies them in ``loadResource`` at exactly the requested width.
The ``~theme`` suffix makes the document fetch new images after a theme
change instead of reusing cached ones.
"""
from __future__ import annotations

import html
import re
from typing import Callable, List, Optional, Tuple

from ..theme import Theme, rgba

SizeOf = Callable[[str], Tuple[int, int]]          # "shot:name" -> natural (w, h)

CALLOUTS = {   # kind -> (icon, theme token)
    "TIP": ("sparkles", "accent"),
    "NOTE": ("info", "text_muted"),
    "IMPORTANT": ("alert", "warning"),
    "TRY IT": ("play", "success"),
}

_ESCAPE = re.compile(r"\\([\\\[\]|*`{}#>!-])")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")


def split_resource(url: str) -> Tuple[str, str]:
    """'shot:overview@760~dark' -> ('shot:overview@760', 'dark')."""
    base, _, tag = url.partition("~")
    return base, tag


def stylesheet(theme: Theme) -> str:
    return f"""
        body {{ color: {theme.text}; }}
        h1 {{ font-size: 21pt; font-weight: 700; margin-top: 0px; margin-bottom: 2px; }}
        h2 {{ font-size: 14.5pt; font-weight: 700; margin-top: 24px; margin-bottom: 6px; }}
        h3 {{ font-size: 11.5pt; font-weight: 700; margin-top: 16px; margin-bottom: 4px; }}
        p {{ margin-top: 4px; margin-bottom: 8px; line-height: 145%; }}
        li {{ margin-bottom: 5px; line-height: 140%; }}
        a {{ color: {theme.accent}; text-decoration: none; font-weight: 600; }}
        .muted {{ color: {theme.text_muted}; }}
        .caption {{ color: {theme.text_muted}; font-size: 9pt; }}
        .eyebrow {{ color: {theme.accent}; font-size: 8.5pt; font-weight: 700; }}
    """


class Renderer:
    def __init__(self, theme: Theme, image_width: int, size_of: SizeOf):
        self.t = theme
        self.width = image_width
        self.size_of = size_of

    def res(self, url: str) -> str:
        """Resource URL tagged with the theme (see module docstring)."""
        return f"{url}~{self.t.name}"

    # ------------------------------------------------------------- inline
    def inline(self, text: str) -> str:
        t = self.t
        literals: List[str] = []

        def protect(match) -> str:
            literals.append(match.group(1))
            return chr(0xE000 + len(literals) - 1)

        s = html.escape(_ESCAPE.sub(protect, text), quote=False)
        s = re.sub(r"\[\[(.+?)\]\]", lambda m: (
            f'<span style="background-color:{t.raised}; color:{t.text}; font-family:monospace; '
            f'font-weight:600;">&#8202;{m.group(1)}&#8202;</span>'), s)
        s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
        s = re.sub(r"`(.+?)`", lambda m: (
            f'<span style="font-family:monospace; color:{t.text}; background-color:{rgba(t.accent, 0.12)};">'
            f'&#8202;{m.group(1)}&#8202;</span>'), s)
        s = re.sub(r"\{icon:([\w-]+)\}", lambda m: (
            f'<img src="{self.res("icon:" + m.group(1))}" width="16" height="16" style="vertical-align: middle;">'), s)
        s = re.sub(r"\{n:(\d+)\}", lambda m: (
            f'<img src="{self.res("badge:" + m.group(1))}" width="20" height="20" style="vertical-align: middle;">'), s)
        s = re.sub(r"\{color:([0-9A-Fa-f]{6})\}", lambda m: (
            f'<img src="{self.res("swatch:" + m.group(1))}" width="16" height="16" style="vertical-align: middle;">'), s)
        for k, literal in enumerate(literals):
            s = s.replace(chr(0xE000 + k), html.escape(literal, quote=False))
        return s

    # -------------------------------------------------------------- blocks
    def heading(self, level: int, text: str) -> str:
        return f"<h{level}>{self.inline(text)}</h{level}>"

    def paragraph(self, lines: List[str]) -> str:
        return f"<p>{self.inline(' '.join(lines))}</p>"

    def steps(self, items: List[Tuple[int, str]]) -> str:
        rows = "".join(
            f'<tr><td width="36" valign="top" style="padding-top:1px;">'
            f'<img src="{self.res(f"step:{n}")}" width="24" height="24"></td>'
            f'<td valign="top" style="padding-bottom:9px; padding-top:3px;">{self.inline(text)}</td></tr>'
            for n, text in items)
        return f'<table cellspacing="0" cellpadding="0" style="margin-top:6px; margin-bottom:8px;">{rows}</table>'

    def bullets(self, items: List[str]) -> str:
        return "<ul style='margin-top:2px; margin-bottom:8px;'>" + "".join(
            f"<li>{self.inline(text)}</li>" for text in items) + "</ul>"

    def callout(self, kind: str, text: str) -> str:
        icon_name, token = CALLOUTS.get(kind, CALLOUTS["NOTE"])
        color = getattr(self.t, token)
        soft = rgba(color if token != "text_muted" else self.t.text_faint, 0.13)
        return (f'<table width="100%" cellspacing="0" cellpadding="11" style="margin-top:8px; margin-bottom:12px;">'
                f'<tr><td style="background-color:{soft};">'
                f'<img src="{self.res(f"icon:{icon_name}:{token}")}" width="16" height="16" '
                f'style="vertical-align: middle;">'
                f'&nbsp;<b style="color:{color};">{kind}</b>&nbsp;&nbsp;{self.inline(text)}</td></tr></table>')

    def image(self, caption: str, source: str) -> str:
        natural_w, natural_h = self.size_of(source)
        is_shot = source.startswith("shot:")
        width = min(self.width, natural_w) if is_shot else self.width
        height = max(1, int(round(natural_h * width / max(natural_w, 1))))
        img = f'<img src="{self.res(f"{source}@{width}")}" width="{width}" height="{height}">'
        if is_shot:   # click to see it full size
            img = f'<a href="zoom:{source[5:]}">{img}</a>'
            caption = f"{caption}  ·  *click to enlarge*" if caption else "*Click to enlarge*"
        caption_html = f'<p align="center" class="caption">{self.inline(caption)}</p>' if caption else ""
        return (f'<p align="center" style="margin-top:12px; margin-bottom:2px; line-height:100%;">{img}</p>'
                f'{caption_html}')

    def table(self, rows: List[List[str]]) -> str:
        t = self.t
        out = ['<table width="100%" cellspacing="0" cellpadding="7" style="margin-top:6px; margin-bottom:12px;">']
        for r, cells in enumerate(rows):
            if r == 0:
                if not any(cells):          # "| | |" header: no header row
                    continue
                out.append("<tr>" + "".join(
                    f'<td style="background-color:{t.surface_alt}; color:{t.text_muted}; font-weight:700;">'
                    f'{self.inline(c)}</td>' for c in cells) + "</tr>")
            else:
                bg = t.surface if r % 2 else t.window
                out.append("<tr>" + "".join(
                    f'<td valign="top" style="background-color:{bg};"'
                    f'{" width=32" if c.startswith(("{n:", "{color:")) else ""}>{self.inline(c)}</td>'
                    for c in cells) + "</tr>")
        out.append("</table>")
        return "".join(out)

    # ------------------------------------------------------------- chapter
    def chapter(self, eyebrow: str, title: str, summary: str, body: str,
                previous: Optional[Tuple[str, str]] = None, following: Optional[Tuple[str, str]] = None) -> str:
        out = [f'<p class="eyebrow">{html.escape(eyebrow.upper())}</p>', f"<h1>{html.escape(title)}</h1>"]
        if summary:
            out.append(f'<p class="muted" style="font-size:11.5pt; margin-bottom:14px;">{self.inline(summary)}</p>')
        out += self.blocks(body)
        out.append(self.footer(previous, following))
        return "".join(out)

    def footer(self, previous, following) -> str:
        left = f'&#8249;&nbsp; <a href="chapter:{previous[0]}">{html.escape(previous[1])}</a>' if previous else ""
        right = f'<a href="chapter:{following[0]}">{html.escape(following[1])}</a> &nbsp;&#8250;' if following else ""
        return (f'<table width="100%" cellspacing="0" cellpadding="10" style="margin-top:26px;">'
                f'<tr><td style="background-color:{self.t.surface};" align="left">{left}</td>'
                f'<td style="background-color:{self.t.surface};" align="right">{right}</td></tr></table>')

    def blocks(self, body: str) -> List[str]:
        lines = body.strip("\n").splitlines()
        out: List[str] = []
        paragraph: List[str] = []

        def flush() -> None:
            if paragraph:
                out.append(self.paragraph(paragraph))
                paragraph.clear()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                flush()
                i += 1
            elif line.startswith("### "):
                flush()
                out.append(self.heading(3, line[4:]))
                i += 1
            elif line.startswith("## "):
                flush()
                out.append(self.heading(2, line[3:]))
                i += 1
            elif re.match(r"\d+\.\s", line):
                flush()
                items = []
                while i < len(lines) and re.match(r"\d+\.\s", lines[i].strip()):
                    number, text = lines[i].strip().split(".", 1)
                    items.append((int(number), text.strip()))
                    i += 1
                out.append(self.steps(items))
            elif line.startswith("- "):
                flush()
                items = []
                while i < len(lines) and lines[i].strip().startswith("- "):
                    items.append(lines[i].strip()[2:])
                    i += 1
                out.append(self.bullets(items))
            elif line.startswith(">"):
                flush()
                text = line.lstrip("> ").strip()
                i += 1
                while i < len(lines) and lines[i].strip().startswith(">"):
                    text += " " + lines[i].strip().lstrip("> ").strip()
                    i += 1
                kind, _, rest = text.partition(":")
                kind = kind.strip().upper()
                if kind not in CALLOUTS:
                    kind, rest = "NOTE", text
                out.append(self.callout(kind, rest.strip()))
            elif line.startswith("!["):
                flush()
                match = re.match(r"!\[(.*?)\]\(([\w:-]+)\)", line)
                if match:
                    out.append(self.image(match.group(1), match.group(2)))
                i += 1
            elif line.startswith("|"):
                flush()
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    row = lines[i].strip()[1:]
                    row = row[:-1] if row.endswith("|") and not row.endswith("\\|") else row
                    cells = [c.strip() for c in _CELL_SPLIT.split(row)]
                    if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):   # skip |---| separators
                        rows.append(cells)
                    i += 1
                out.append(self.table(rows))
            else:
                paragraph.append(line)
                i += 1
        flush()
        return out


def plain_text(body: str) -> str:
    """Searchable text of a chapter (markup removed)."""
    text = re.sub(r"!\[(.*?)\]\(.*?\)", r"\1", body)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\{(icon|n|color):[\w-]+\}", "", text)
    text = _ESCAPE.sub(r"\1", text)
    return re.sub(r"[#*`>\[\]|]", " ", text)
