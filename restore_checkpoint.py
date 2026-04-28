import shutil
import os

src_active = 'weights_v6/summarizer_best.pt'
src_safe = 'weights_v5/summarizer_best.pt'
backup = src_active + '.bak'

if os.path.exists(src_active):
    shutil.copy(src_active, backup)
    print('Backed up', src_active, '->', backup)
else:
    print('No existing active best to back up:', src_active)

if not os.path.exists(src_safe):
    raise SystemExit('Safe checkpoint not found: ' + src_safe)

shutil.copy(src_safe, src_active)
print('Restored', src_safe, '->', src_active)
