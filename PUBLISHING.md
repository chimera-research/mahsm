# Publishing mahsm to PyPI

This guide explains how to publish mahsm to PyPI using GitHub Actions.

---

## Prerequisites

### 1. PyPI Account Setup

1. Create accounts on:
   - **[PyPI](https://pypi.org/account/register/)** (production)
   - **[Test PyPI](https://test.pypi.org/account/register/)** (testing)

2. **Enable 2FA** on both accounts (required for publishing)

### 2. Configure Trusted Publishing (Recommended)

Trusted Publishing uses GitHub's OIDC provider to authenticate without needing API tokens.

#### On PyPI:
1. Go to https://pypi.org/manage/account/publishing/
2. Click **"Add a new publisher"**
3. Fill in:
   - **PyPI Project Name**: `mahsm`
   - **Owner**: `chimera-research`
   - **Repository name**: `mahsm`
   - **Workflow name**: `publish.yml`
   - **Environment name**: (leave blank)
4. Click **"Add"**

#### On Test PyPI:
1. Go to https://test.pypi.org/manage/account/publishing/
2. Repeat the same steps as above

---

## Publishing Methods

### Method 1: GitHub Release (Automatic - Recommended)

**This is the recommended method for production releases.**

1. **Update version** in `pyproject.toml`:
   ```toml
   [project]
   name = "mahsm"
   version = "0.1.0"  # Increment this
   ```

2. **Commit and push** version bump:
   ```bash
   git add pyproject.toml
   git commit -m "chore: bump version to 0.1.0"
   git push origin main
   ```

3. **Create a GitHub Release**:
   ```bash
   # Via GitHub CLI
   gh release create v0.1.0 \
     --title "mahsm v0.1.0" \
     --notes "Initial release with core features"

   # Or via GitHub UI
   # Go to: https://github.com/chimera-research/mahsm/releases/new
   ```

4. **Monitor the workflow**:
   - Go to: https://github.com/chimera-research/mahsm/actions
   - The "Publish to PyPI" workflow will trigger automatically
   - Wait for it to complete (usually < 2 minutes)

5. **Verify publication**:
   - Check: https://pypi.org/project/mahsm/
   - Test install: `pip install mahsm==0.1.0`

### Method 2: Manual Workflow Dispatch (Testing)

**Use this for testing on Test PyPI before a real release.**

1. Go to: https://github.com/chimera-research/mahsm/actions/workflows/publish.yml

2. Click **"Run workflow"**

3. Select options:
   - **Branch**: `main` (or your branch)
   - **Publish to Test PyPI**: ✅ (checked for testing)

4. Click **"Run workflow"**

5. **Verify on Test PyPI**:
   - Check: https://test.pypi.org/project/mahsm/
   - Test install:
     ```bash
     pip install --index-url https://test.pypi.org/simple/ mahsm
     ```

---

## Version Numbering

mahsm uses [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH

- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes (backwards compatible)
```

### Examples:
- `0.1.0` → `0.1.1`: Bug fix
- `0.1.0` → `0.2.0`: New feature
- `0.1.0` → `1.0.0`: Breaking change or stable release

---

## Troubleshooting

### Error: "Project name already exists"

If the project name is taken on PyPI:

1. **Option A**: Request name transfer (if inactive)
   - Email: admin@pypi.org
   - Include: reason, your PyPI username

2. **Option B**: Use a different name
   - Update `name` in `pyproject.toml`
   - Update all references in docs/README

### Error: "403 Forbidden" during publish

**Cause**: Trusted Publishing not configured correctly.

**Fix**:
1. Verify workflow name matches exactly: `publish.yml`
2. Check repository owner/name match
3. Ensure workflow has `id-token: write` permission (already configured)

### Error: "Version already exists"

**Cause**: Trying to re-upload same version.

**Fix**:
1. Bump version in `pyproject.toml`
2. Create new release

### Workflow doesn't trigger

**Check**:
1. Workflow file exists at `.github/workflows/publish.yml`
2. Release was created (not just a tag)
3. Check Actions tab for errors

---

## Best Practices

### Before Publishing

1. ✅ **Run all tests**:
   ```bash
   python tests/test_core.py
   python tests/test_graph_integration.py
   ```

2. ✅ **Update CHANGELOG** (if you have one):
   ```markdown
   ## [0.1.0] - 2025-10-31
   ### Added
   - Initial release
   - Core `@ma.dspy_node` decorator
   - Langfuse integration
   ```

3. ✅ **Update documentation**:
   - README.md mentions new version
   - QUICKSTART.md reflects latest features

4. ✅ **Test on Test PyPI first**:
   - Use manual workflow dispatch
   - Install and test: `pip install --index-url https://test.pypi.org/simple/ mahsm`

### After Publishing

1. ✅ **Announce the release**:
   - Tweet / social media
   - Post in relevant communities

2. ✅ **Monitor for issues**:
   - Watch GitHub Issues
   - Check PyPI download stats

---

## Local Testing (Without Publishing)

To test the build process locally:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Check the built package
twine check dist/*

# Optional: Test install locally
pip install dist/mahsm-0.1.0-py3-none-any.whl
```

---

## Security Notes

1. **Never commit API tokens** to the repository
2. **Use Trusted Publishing** (OIDC) instead of API tokens
3. **Enable 2FA** on PyPI accounts
4. **Review workflow logs** before publishing to production

---

## References

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions PyPI Publish](https://github.com/marketplace/actions/pypi-publish)
- [Python Packaging Guide](https://packaging.python.org/)

---

**Happy Publishing!** 🚀
