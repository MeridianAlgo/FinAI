# Git LFS Setup Guide

This guide will help you set up Git Large File Storage (LFS) for handling large model files in your FinAI repository.

## Why Git LFS?

As your model trains and grows, the `.pt` and `.pkl` files can become very large (100MB+). Git LFS is designed to handle large files efficiently by storing them separately from your Git repository, preventing repository bloat.

## Installation

### Windows
```powershell
# Using Chocolatey
choco install git-lfs

# Or download installer from:
# https://git-lfs.github.com/
```

### macOS
```bash
brew install git-lfs
```

### Linux
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# Fedora/RHEL
sudo dnf install git-lfs
```

## Setup Steps

### 1. Initialize Git LFS in Your Repository

```bash
cd c:\Users\ishaa\OneDrive\Desktop\finai\FinAI
git lfs install
```

### 2. Track Large Files

The `.gitattributes` file has already been created to track:
- `*.pt` - PyTorch model files
- `*.pkl` - Pickle files (tokenizers, etc.)
- `*.pth` - PyTorch checkpoint files
- `*.bin` - Binary model files
- `*.h5` - HDF5 model files
- Other large data formats

### 3. Migrate Existing Large Files to LFS

If you already have large model files committed to Git, migrate them:

```bash
# Track the files
git lfs track "*.pt" "*.pkl"

# Migrate existing files
git lfs migrate import --include="*.pt,*.pkl" --everything

# Push the migrated files
git push --force
```

### 4. Verify LFS is Working

```bash
# Check which files are tracked by LFS
git lfs ls-files

# Check LFS status
git lfs status

# See LFS file info
git lfs ls-files -s
```

## GitHub Actions Integration

The workflow is already configured to:
1. **Checkout with LFS**: `lfs: true` in checkout action
2. **Pull LFS files**: `git lfs pull` before training
3. **Push LFS files**: `git lfs push` after training
4. **Track new files**: Automatically tracks model files with LFS

## File Size Monitoring

The workflow now reports model file sizes in:
- Commit messages
- Pull request descriptions
- Email notifications

Example: "Model size: 57M"

## Best Practices

### 1. Always Use LFS for Large Files
Files over 50MB should always be tracked with LFS:
```bash
git lfs track "large_file.pt"
git add .gitattributes large_file.pt
git commit -m "Add large model file"
```

### 2. Check File Sizes Before Committing
```bash
# Check file size
du -h models/finai_gpt.pt

# If over 50MB, ensure it's tracked by LFS
git lfs ls-files | grep finai_gpt.pt
```

### 3. Clean Up Old LFS Files
Periodically clean up old LFS objects:
```bash
git lfs prune
```

## Troubleshooting

### Issue: "This exceeds GitHub's file size limit of 100 MB"
**Solution**: The file isn't being tracked by LFS. Run:
```bash
git lfs track "*.pt"
git add .gitattributes
git add models/finai_gpt.pt
git commit --amend
```

### Issue: LFS files not downloading
**Solution**: Manually pull LFS files:
```bash
git lfs pull
```

### Issue: Slow pushes
**Solution**: LFS files are large. This is normal. You can check progress:
```bash
GIT_TRACE=1 git push
```

### Issue: LFS bandwidth limit exceeded
GitHub has LFS bandwidth limits:
- Free: 1 GB/month
- Pro: 50 GB/month

**Solution**: 
- Reduce training frequency
- Use GitHub Pro
- Consider alternative storage (S3, etc.)

## Monitoring LFS Usage

Check your LFS storage and bandwidth:
1. Go to your repository on GitHub
2. Settings → Billing → Git LFS Data
3. View storage and bandwidth usage

## Current Setup

Your repository is now configured to:
- ✅ Track all model files (`.pt`, `.pkl`) with LFS
- ✅ Automatically handle large files in CI/CD
- ✅ Report model sizes in updates
- ✅ Prevent repository bloat

## Next Steps

1. Run `git lfs install` locally
2. Verify with `git lfs ls-files`
3. Monitor file sizes as model grows
4. Consider upgrading GitHub plan if you hit LFS limits

## Additional Resources

- [Git LFS Documentation](https://git-lfs.github.com/)
- [GitHub LFS Guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
- [LFS Pricing](https://docs.github.com/en/billing/managing-billing-for-git-large-file-storage)
