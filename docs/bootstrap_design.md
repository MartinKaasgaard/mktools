
# mktools bootstrap design

## Decision: public core, private overlay

The recommended design is:

- **Public**: `mktools.common.bootstrap`
- **Private**: personal / organizational overlays that contain:
  - Drive folder conventions
  - internal storage mount points
  - secret file locations
  - private package indexes or Git URLs
  - any token-bearing installer logic

### Why the core should stay public

A public bootstrap core is the better default because:

1. It avoids a secret dependency just to start a notebook.
2. It keeps Colab bootstrap simple: install `mktools`, import bootstrap, run.
3. The core code is not sensitive when it only contains generic logic.
4. The private pieces change more often and are user- or organization-specific.

### What should stay private

Keep these out of the public repo:

- personal Google Drive paths like `MyDrive/Education/...`
- private GitHub package URLs
- token names or secret retrieval conventions tied to one account
- internal NFS / DGX mount points
- Kaggle credential locations if they reveal private structure

Instead, supply them through:
- environment variables
- an ignored JSON overlay file
- a separate private overlay repo or gist
- a Colab userdata-backed config cell

## Bootstrap goals

This implementation is:

- **typed**
- **environment-aware**
- **capability-driven**
- **less notebook-cell-order dependent**
- **mktools-centric**

## How it fits the workflow

1. Install `mktools` in the first notebook cell.
2. Import `mktools.common.bootstrap`.
3. Run one `bootstrap_environment(...)` call.
4. Use `mktools.kfile`, `mktools.kio`, and `mktools.kstat`.

## Overlay JSON example

```json
{
  "persistent_root": "/content/drive/MyDrive/Education/Walsh-DBA-2024/Courses/Walsh-2025",
  "common_code_root": "/content/drive/MyDrive/Education/CommonCode/python",
  "kaggle_home": "/content/drive/MyDrive/Education/CommonCode/KaggleHome"
}
```

That file should remain private or ignored.

## Important notebook pattern

The bootstrap lives inside `mktools`, so the notebook still needs a tiny installer cell first.
That is expected and is the cleanest pattern for Colab.
