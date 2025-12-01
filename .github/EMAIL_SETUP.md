# Email Notification Setup Guide

This guide will help you set up email notifications for the FinAI training workflow.

## Required GitHub Secrets

You need to add the following secrets to your GitHub repository:

### 1. Navigate to Repository Settings
1. Go to your repository on GitHub
2. Click on **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**

### 2. Add the Following Secrets

#### EMAIL_USERNAME
- **Name**: `EMAIL_USERNAME`
- **Value**: Your email address (e.g., `your-email@gmail.com`)

#### EMAIL_PASSWORD
- **Name**: `EMAIL_PASSWORD`
- **Value**: Your email app password (see below for Gmail setup)

#### EMAIL_TO
- **Name**: `EMAIL_TO`
- **Value**: The email address where you want to receive notifications (can be the same as EMAIL_USERNAME)

## Gmail Setup (Recommended)

If you're using Gmail, you need to create an **App Password**:

### Step 1: Enable 2-Factor Authentication
1. Go to your Google Account settings: https://myaccount.google.com/
2. Navigate to **Security**
3. Enable **2-Step Verification** if not already enabled

### Step 2: Generate App Password
1. Go to https://myaccount.google.com/apppasswords
2. Select **Mail** as the app
3. Select **Other (Custom name)** as the device
4. Enter "FinAI Training Bot" as the name
5. Click **Generate**
6. Copy the 16-character password (this is your `EMAIL_PASSWORD`)

### Step 3: Add to GitHub Secrets
Use the generated app password as the value for `EMAIL_PASSWORD` secret.

## Alternative Email Providers

### Outlook/Hotmail
- **Server**: `smtp.office365.com`
- **Port**: `587`
- Update the workflow file to use these settings

### Yahoo Mail
- **Server**: `smtp.mail.yahoo.com`
- **Port**: `587`
- Update the workflow file to use these settings

### Custom SMTP Server
If you're using a different email provider, update these lines in `daily_train.yml`:
```yaml
server_address: smtp.your-provider.com
server_port: 587  # or 465 for SSL
```

## Testing the Setup

After adding the secrets, you can test the workflow by:
1. Going to **Actions** tab in your GitHub repository
2. Selecting **Daily FinAI Training** workflow
3. Clicking **Run workflow** → **Run workflow**
4. Check your email for the notification

## Troubleshooting

### Not Receiving Emails?
1. Check that all three secrets are correctly set
2. Verify your app password is correct
3. Check your spam/junk folder
4. Review the workflow logs in GitHub Actions for error messages

### Gmail Blocking Sign-in?
- Make sure you're using an App Password, not your regular password
- Ensure 2-Factor Authentication is enabled
- Check Google's security alerts for blocked sign-in attempts

## Schedule Information

The workflow is now scheduled to run:
- **7 AM CST** (1 PM UTC) - Morning training
- **5 PM CST** (11 PM UTC) - Evening training

You'll receive an email notification after each training run completes (success or failure).
