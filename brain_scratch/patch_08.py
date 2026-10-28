"""Fix typo in 08_train_hybrid.py: line 406 uses config.CF_N_FACTORS instead of 0.6 for alpha display."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('src/08_train_hybrid.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the typo in the log message
old = 'logger.info(f"\\n=== Xay dung Hybrid (alpha={config.CF_N_FACTORS}) ===")'
new = 'logger.info(f"\\n=== Xay dung Hybrid (alpha=0.6) ===")'

# Try exact match first
if old in content:
    content = content.replace(old, new)
    print("Fixed alpha typo (exact match)")
else:
    # Search for the pattern
    idx = content.find('CF_N_FACTORS}) ===')
    if idx != -1:
        # Find line start
        line_start = content.rfind('\n', 0, idx) + 1
        line_end = content.find('\n', idx)
        old_line = content[line_start:line_end]
        new_line = '    logger.info(f"\\n=== Xay dung Hybrid (alpha=0.6) ===")'
        content = content[:line_start] + new_line + content[line_end:]
        print(f"Fixed line: {old_line!r} -> {new_line!r}")
    else:
        print("Pattern not found - checking nearby lines")
        idx2 = content.find('Xay dung Hybrid')
        if idx2 != -1:
            print(repr(content[idx2-5:idx2+80]))

with open('src/08_train_hybrid.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done")
