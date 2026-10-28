"""Patch 11_recommend.py to:
1. Fix svd.trainset.to_raw_iid(iid)
2. Fix hybrid rating score calculation (direct linear combination on star scale)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('src/11_recommend.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix all_items inner ID mapping
old_items = 'all_items = set(int(iid) for iid in svd.trainset.all_items()) - seen_movies'
new_items = 'all_items = set(int(svd.trainset.to_raw_iid(iid)) for iid in svd.trainset.all_items()) - seen_movies'
if old_items in content:
    content = content.replace(old_items, new_items)
    print("Patched all_items in 11_recommend.py")
else:
    print("old_items not found")

# 2. Fix hybrid score
old_h = '''            cf_n = (cf_p - config.MIN_RATING) / (config.MAX_RATING - config.MIN_RATING)
            cb_n = max(0.0, min(1.0, cb_p))
            h = alpha * cf_n + (1 - alpha) * cb_n'''
new_h = '''            h = alpha * cf_p + (1.0 - alpha) * cb_p'''
if old_h in content:
    content = content.replace(old_h, new_h)
    print("Patched hybrid score in 11_recommend.py")
else:
    print("old_h not found")

with open('src/11_recommend.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Saved patched src/11_recommend.py")
