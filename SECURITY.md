# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do Not Open a Public Issue

Please **do not** open a public GitHub issue for security vulnerabilities.

### 2. Report Privately

Send a detailed report to the repository maintainers via:

- GitHub Security Advisories (preferred)
- Direct message to repository owner

### 3. Include in Your Report

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)
- Your contact information

### 4. Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: 1-7 days
  - High: 7-30 days
  - Medium: 30-90 days
  - Low: Best effort

## Security Best Practices

### For Users

1. **Keep Dependencies Updated**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Use Virtual Environments**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Protect API Keys**
   - Never commit API keys to git
   - Use environment variables or secrets
   - Rotate keys regularly

4. **Validate Input Data**
   - Sanitize user inputs
   - Validate file uploads
   - Check dataset sources

5. **Monitor Dependencies**
   ```bash
   pip install safety
   safety check
   ```

### For Contributors

1. **Code Review**
   - All PRs require review
   - Security-sensitive changes need extra scrutiny

2. **Dependency Management**
   - Pin dependency versions
   - Review dependency updates
   - Check for known vulnerabilities

3. **Secrets Management**
   - Use GitHub Secrets for CI/CD (Settings → Secrets and variables → Actions)
   - Never hardcode credentials in code
   - Use `.env` files for local development (already in `.gitignore`)
   - Use `.env.example` as a template (safe to commit)
   - Never commit `.env` files to the repository
   - Rotate tokens/keys if accidentally exposed

4. **Input Validation**
   - Validate all external inputs
   - Sanitize file paths
   - Check data types and ranges

## Known Security Considerations

### Model Training

- **Data Poisoning**: Validate dataset sources
- **Model Extraction**: Limit API access if deploying
- **Resource Exhaustion**: Set timeouts and limits

### API Keys

- **Comet ML API Key**: Store in GitHub Secrets
- **Hugging Face Token**: Optional, store securely if used

### File Operations

- **Path Traversal**: Validate file paths
- **Arbitrary File Write**: Restrict write locations
- **Large Files**: Implement size limits

### Dependencies

We regularly update dependencies to patch security vulnerabilities. Check `requirements.txt` for current versions.

## Security Updates

Security updates are released as:

1. **Patch Versions** (1.0.x) for minor fixes
2. **GitHub Security Advisories** for critical issues
3. **Release Notes** documenting security fixes

## Disclosure Policy

- We follow responsible disclosure
- Security fixes are released before public disclosure
- Credit given to reporters (if desired)

## Security Checklist for Deployments

- [ ] Use HTTPS for all connections
- [ ] Implement rate limiting
- [ ] Set up monitoring and logging
- [ ] Use strong authentication
- [ ] Keep dependencies updated
- [ ] Regular security audits
- [ ] Backup critical data
- [ ] Implement access controls

## Automated Security

We use:

- **Dependabot**: Automatic dependency updates
- **GitHub Security Scanning**: Code analysis
- **Secret Scanning**: Prevent credential leaks

## Contact

For security concerns, contact:
- GitHub Security Advisories (preferred)
- Repository maintainers

## Acknowledgments

We thank security researchers who responsibly disclose vulnerabilities.

---

Last Updated: December 2024
