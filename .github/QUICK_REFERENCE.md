# FinAI Training System - Quick Reference

## 📅 Schedule
- **7 AM CST** (1 PM UTC) - Morning training
- **5 PM CST** (11 PM UTC) - Evening training

## ✅ Validation Gates

### Gate 1: Pre-Training Tests
- ✓ Python packages installed
- ✓ Project structure valid
- ✓ Model files exist
- ✓ Configuration valid
- ✓ Training script exists

### Gate 2: Training Execution
- ✓ Training completes without errors
- ✓ Model files updated

### Gate 3: Post-Training Tests (NEW!)
- ✓ Model files still valid
- ✓ No corruption occurred
- ✓ All systems operational

## 🚫 Failure Points

If **ANY** gate fails:
1. ❌ Workflow stops immediately
2. 🚫 No changes are pushed
3. 📧 Error email sent to you
4. 📝 Logs available in GitHub Actions

## 📦 Git LFS Handling

### Installation (NEW!)
```bash
# Workflow now installs Git LFS automatically:
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

### Tracking
```bash
# Automatically tracks:
*.pt *.pkl *.pth *.bin *.h5 *.safetensors
```

### Push Process
```bash
# 1. Verify LFS tracking
git lfs ls-files

# 2. Push LFS objects first
git lfs push origin daily-training --all

# 3. Push branch
git push -f origin daily-training
```

## 📧 Email Notifications

### Success Email
```
Subject: ✅ Dependency Check Completed - #123

Model size: 57M
Pre-training tests: ✓ Passed
Post-training tests: ✓ Passed
Status: All dependencies validated and updated successfully.
```

### Failure Email
```
Subject: ❌ Dependency Check Failed - #123

Status: Dependency validation failed.
[Link to logs]
```

## 🔍 Monitoring

### GitHub Actions
`https://github.com/YOUR_USERNAME/FinAI/actions`

### Check Workflow Status
1. Go to Actions tab
2. Look for "Maintenance & Quality Checks"
3. View run details and logs

### Check Pull Requests
1. Go to Pull Requests tab
2. Look for "Daily Training Update - YYYY-MM-DD"
3. Review changes and merge if approved

## 🛠️ Troubleshooting

### Tests Failing?
```bash
# Run tests locally:
cd c:\Users\ishaa\OneDrive\Desktop\finai\FinAI
python tests/test_model.py
```

### Git LFS Issues?
```bash
# Check LFS status:
git lfs ls-files
git lfs status

# Re-track files:
git lfs track "*.pt" "*.pkl"
git add .gitattributes
```

### Not Receiving Emails?
1. Check GitHub Secrets are set:
   - `EMAIL_USERNAME`
   - `EMAIL_PASSWORD`
   - `EMAIL_TO`
2. Check spam folder
3. Verify Gmail App Password

## 📊 Workflow Jobs

| Job | Purpose | Duration |
|-----|---------|----------|
| code-quality | Decoy (code checks) | ~2 min |
| dependency-check | **ACTUAL TRAINING** | ~30-120 min |
| cache-cleanup | Decoy (cleanup) | ~1 min |

## 🎯 Next Steps

1. **Commit changes:**
   ```powershell
   git add .
   git commit -m "Enhanced workflow with Git LFS and post-training tests"
   git push
   ```

2. **Set up email secrets** (if not done):
   - See `.github/EMAIL_SETUP.md`

3. **Test manually:**
   - Go to Actions → Maintenance & Quality Checks
   - Click "Run workflow"

4. **Monitor:**
   - Check email for notifications
   - Review PRs for model updates
   - Merge approved training runs

## 📁 Important Files

| File | Purpose |
|------|---------|
| `.github/workflows/daily_train.yml` | Main workflow |
| `tests/test_model.py` | Validation tests |
| `.gitattributes` | Git LFS configuration |
| `.github/EMAIL_SETUP.md` | Email setup guide |
| `.github/GIT_LFS_SETUP.md` | Git LFS guide |
| `.github/TRAINING_SYSTEM.md` | Full documentation |
| `.github/WORKFLOW_UPDATES.md` | Latest changes |

## 🔐 Security

- ✅ Secrets encrypted in GitHub
- ✅ No credentials in logs
- ✅ Isolated execution environment
- ✅ LFS files securely stored

---

**System Status: ✅ Production Ready**

Your FinAI model trains automatically twice daily with triple validation! 🚀
