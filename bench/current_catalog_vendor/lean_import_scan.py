import re

def lean_code_mask(text: str, *, strings: bool = True) -> str:
    """Blank comments/strings while preserving offsets and line structure.

    Ordinary escaped strings and Lean raw strings (r"..." or r#"..."#) are
    lexical literals, so their ``triton {`` text and braces are not syntax.
    Macro quotations remain code; only their literal strings are masked.
    """
    out, i, n = list(text), 0, len(text)

    def blank(start, end):
        for index in range(start, end):
            if text[index] != '\n':
                out[index] = ' '

    while i < n:
        if text.startswith('--', i):
            end = text.find('\n', i)
            end = n if end < 0 else end
            blank(i, end)
            i = end
        elif text.startswith('/-', i):
            start, depth = i, 1
            i += 2
            while i < n and depth:
                if text.startswith('/-', i):
                    depth += 1
                    i += 2
                elif text.startswith('-/', i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth:
                raise ValueError('Unterminated Lean block comment')
            blank(start, i)
        else:
            character = re.match(r"'(?:\\(?:u\{[0-9A-Fa-f]+\}|.)|[^'\\\n])'", text[i:]) if (
                text[i] == "'" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] in "_'"))
            ) else None
            if character:
                if strings:
                    blank(i, i + character.end())
                i += character.end()
                continue
            raw = re.match(r'r(#+)?"', text[i:]) if (
                text[i] == 'r' and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_'))
            ) else None
            if raw:
                closing = '"' + (raw[1] or '')
                end = text.find(closing, i + raw.end())
                if end < 0:
                    raise ValueError('Unterminated Lean raw string')
                end += len(closing)
                if strings:
                    blank(i, end)
                i = end
            elif text[i] == '"':
                start = i
                i += 1
                while i < n:
                    if text[i] == "\\":
                        i += 2
                    elif text[i] == '"':
                        i += 1
                        break
                    else:
                        i += 1
                else:
                    raise ValueError('Unterminated Lean string')
                if strings:
                    blank(start, min(i, n))
            else:
                i += 1
    return ''.join(out)
