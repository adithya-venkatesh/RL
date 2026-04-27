#!/usr/bin/env python3

with open('main_with_vanilla.py', 'r') as f:
    content = f.read()

# Replace all [Exp {experiment_num}] with [Exp {experiment_num + 1}]
import re
content = re.sub(
    r'\[Exp \{experiment_num\}\]',
    '[Exp {experiment_num + 1}]',
    content
)

with open('main_with_vanilla.py', 'w') as f:
    f.write(content)

print("✓ Fixed experiment number display")
print("✓ Now shows [Exp 4] for EXPERIMENT 4/5")
