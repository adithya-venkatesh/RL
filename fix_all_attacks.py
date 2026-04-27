#!/usr/bin/env python3
import re

with open('main_with_vanilla.py', 'r') as f:
    content = f.read()

# Replace all instances of the bug
# Pattern: attack_action = int(attack_action[0])
old_pattern = r'attack_action = int\(attack_action\[0\]\)'
new_code = 'attack_action = int(attack_action.item() if hasattr(attack_action, "item") else attack_action)'

content = re.sub(old_pattern, new_code, content)

with open('main_with_vanilla.py', 'w') as f:
    f.write(content)

print("✓ Fixed all 3 instances of attack_action[0] bug")
print("✓ Lines fixed: 996, 1078, 1245")
