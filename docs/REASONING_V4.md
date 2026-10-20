# Reasoning v4

Reasoning v4 makes the arithmetic hidden inside the linear-equation `collect`
phase explicit. V3 showed the collected equation and the result of combining
terms, but did not show the additions or subtractions that produced those
values.

For example, solving `2x + 3 = 9` is rendered as:

```text
step:collect:2x+(3)=0x+(9)
call:subtract(2,0) -> 2
call:subtract(9,3) -> 6
move:2x=6
call:divide(6,2) -> 3
x=3
```

The first call explicitly computes the net variable coefficient; the second
computes the net constant. Each call uses exact rational arithmetic and is
checked during annotation verification. V3 traces and their behavior remain
unchanged. Other v3 reasoning traces retain their existing rendering in v4.

The annotation API selects this trace format with
`annotate_row(row, protocol="reasoning_v4")`. The separate annotation preset is
`uv run python scripts/annotate_reasoning.py --config-name reasoning_v4`.
It writes the v4 bundle to its own output path. V4 uses the existing character
completion framing, tokenizer, call executor, scorer, and explanation
transcriber interfaces.
