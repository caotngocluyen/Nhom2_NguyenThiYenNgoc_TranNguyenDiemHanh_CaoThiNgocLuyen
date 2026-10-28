with open('src/08_train_hybrid.py', 'r', encoding='utf-8') as f:
    c = f.read()
old = 'demo_users = df_train.groupby("userId").size().nlargest(5).index.tolist()'
new = '''user_counts = df_train.groupby("userId").size()
    demo_users = user_counts[user_counts.between(50, 100)].index[:3].tolist()'''
if old in c:
    c = c.replace(old, new)
    with open('src/08_train_hybrid.py', 'w', encoding='utf-8') as f:
        f.write(c)
    print("Replaced demo_users")
else:
    print("old string not found")
