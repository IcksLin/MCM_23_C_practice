from __future__ import annotations

import re
import sys
from pathlib import Path


MATH_REPLACEMENTS = {
    "λ": r"\lambda{}",
    "β": r"\beta{}",
    "γ": r"\gamma{}",
    "η": r"\eta{}",
    "π": r"\pi{}",
    "Π": r"\Pi{}",
    "Δ": r"\Delta{}",
    "Ω": r"\Omega{}",
    "Γ": r"\Gamma{}",
    "ε": r"\varepsilon{}",
    "ω": r"\omega{}",
    "ξ": r"\xi{}",
    "ζ": r"\zeta{}",
    "τ": r"\tau{}",
    "κ": r"\kappa{}",
    "α": r"\alpha{}",
    "ρ": r"\rho{}",
    "χ": r"\chi{}",
    "Σ": r"\sum{}",
    "∑": r"\sum{}",
    "≤": r"\leq{}",
    "≥": r"\geq{}",
    "≠": r"\neq{}",
    "∈": r"\in{}",
    "±": r"\pm{}",
    "²": r"^{2}",
}


def normalize_math(value: str) -> str:
    for source, target in MATH_REPLACEMENTS.items():
        value = value.replace(source, target)
    value = re.sub(r"_([A-Za-z0-9]+)", r"_{\1}", value)
    return value


def escape_plain(value: str) -> str:
    placeholders: dict[str, str] = {}

    def protect_math(match: re.Match[str]) -> str:
        key = f"@@MATH{len(placeholders)}@@"
        placeholders[key] = match.group(0)
        return key

    value = re.sub(r"\$[^$]+\$", protect_math, value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = value.replace("→", r"\ensuremath{\rightarrow}")
    value = value.replace("↓", r"\ensuremath{\downarrow}")
    for source, target in MATH_REPLACEMENTS.items():
        value = value.replace(source, rf"\ensuremath{{{target}}}")
    for key, original in placeholders.items():
        value = value.replace(key, original)
    return value


def inline(value: str) -> str:
    code_values: list[str] = []

    def protect_code(match: re.Match[str]) -> str:
        key = f"@@CODE{len(code_values)}@@"
        code_values.append(match.group(1))
        return key

    value = re.sub(r"`([^`]+)`", protect_code, value)
    value = escape_plain(value)
    value = re.sub(r"\*\*([^*]+)\*\*", lambda m: r"\textbf{" + m.group(1) + "}", value)
    value = re.sub(r"\*([^*]+)\*", lambda m: r"\emph{" + m.group(1) + "}", value)
    for idx, code in enumerate(code_values):
        key = f"@@CODE{idx}@@"
        is_math = bool(re.search(r"[λβγηπωξζτκαρχΣ∑≤≥≠∈±²\\]", code))
        if is_math:
            rendered = r"\(" + normalize_math(code) + r"\)"
        else:
            rendered = r"\texttt{" + escape_plain(code) + "}"
        value = value.replace(key, rendered)
    return value


def split_table_row(line: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


def column_spec(count: int) -> str:
    widths = {
        2: [0.26, 0.68],
        3: [0.20, 0.36, 0.38],
        4: [0.17, 0.27, 0.27, 0.23],
        5: [0.12, 0.20, 0.24, 0.24, 0.14],
        6: [0.05, 0.15, 0.27, 0.16, 0.25, 0.08],
        7: [0.10, 0.17, 0.15, 0.13, 0.13, 0.13, 0.11],
    }
    selected = widths.get(count, [0.94 / count] * count)
    return "@{}" + "".join(r">{\raggedright\arraybackslash}p{" + f"{w:.2f}" + r"\linewidth}" for w in selected) + "@{}"


def render_table(rows: list[list[str]], table_number: int) -> list[str]:
    header = rows[0]
    body = rows[2:]
    spec = column_spec(len(header))
    output = [
        r"\begin{longtable}{" + spec + "}",
        rf"\caption{{思路表第 {table_number} 表}}\label{{tab:thought-{table_number}}}\\",
        r"\toprule",
    ]
    output.append(" & ".join(r"\textbf{" + inline(cell) + "}" for cell in header) + r" \\")
    output.extend([r"\midrule", r"\endfirsthead", r"\toprule"])
    output.append(" & ".join(r"\textbf{" + inline(cell) + "}" for cell in header) + r" \\")
    output.extend([r"\midrule", r"\endhead"])
    for row in body:
        output.append(" & ".join(inline(cell) for cell in row) + r" \\")
    output.extend([r"\bottomrule", r"\end{longtable}"])
    return output


def convert(source: Path, target: Path) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    body: list[str] = []
    index = 0
    list_kind: str | None = None
    in_math = False
    table_number = 0

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            body.append(rf"\end{{{list_kind}}}")
            list_kind = None

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped == r"\[":
            close_list()
            in_math = True
            body.append(r"\[")
            index += 1
            continue
        if in_math:
            body.append(line)
            if stripped == r"\]":
                in_math = False
            index += 1
            continue
        if stripped.startswith("```"):
            close_list()
            code: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            body.extend([r"\begin{center}", r"\begin{minipage}{0.78\linewidth}", r"\small\raggedright"])
            for code_line in code:
                body.append(inline(code_line) + r"\par")
            body.extend([r"\end{minipage}", r"\end{center}"])
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[index + 1].strip()):
            close_list()
            table_number += 1
            table_lines: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(split_table_row(lines[index]))
                index += 1
            body.extend(render_table(table_lines, table_number))
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            close_list()
            level = len(heading.group(1))
            title = inline(heading.group(2))
            if level == 1:
                body.append(r"\title{" + title + "}")
                body.append(r"\author{}")
                body.append(r"\date{}")
                body.append(r"\maketitle")
                body.append(r"\begin{abstract}")
                body.append("本文将农作物种植策略问题一至问题三的数据处理、数学模型、求解流程、验证方法、现有结果及证据边界汇总为统一思路表，供后续工程实现与论文写作核对使用。")
                body.append(r"\keywords{农作物种植；混合整数线性规划；情景分析；CVaR；t-Copula}")
                body.append(r"\end{abstract}")
            else:
                command = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
                body.append(rf"\{command}{{{title}}}")
            index += 1
            continue
        if stripped.startswith("> "):
            close_list()
            body.extend([r"\begin{quote}\small", inline(stripped[2:]), r"\end{quote}"])
            index += 1
            continue
        numbered = re.match(r"^\d+\.\s+(.*)$", stripped)
        bullet = re.match(r"^-\s+(.*)$", stripped)
        if numbered or bullet:
            desired = "enumerate" if numbered else "itemize"
            if list_kind != desired:
                close_list()
                body.append(rf"\begin{{{desired}}}")
                list_kind = desired
            body.append(r"\item " + inline((numbered or bullet).group(1)))
            index += 1
            continue
        close_list()
        if not stripped:
            body.append("")
        else:
            body.append(inline(stripped))
        index += 1

    close_list()
    preamble = r"""\documentclass[UTF8,11pt,a4paper]{ctexart}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,longtable,array}
\usepackage[a4paper,landscape,left=1.4cm,right=1.4cm,top=1.5cm,bottom=1.5cm]{geometry}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\newcommand{\keywords}[1]{\par\noindent\textbf{关键词：}#1}
\setlength{\headheight}{14pt}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.25em}
\setlength{\LTpre}{0.35em}
\setlength{\LTpost}{0.55em}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.22}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{2024 年高教社杯 C 题：农作物种植策略}
\fancyhead[R]{问题 1--3 总思路表}
\fancyfoot[C]{\thepage}
\setcounter{secnumdepth}{0}
\begin{document}
"""
    ending = "\n\\end{document}\n"
    target.write_text(preamble + "\n".join(body) + ending, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: render_thought_table_tex.py SOURCE.md TARGET.tex")
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
