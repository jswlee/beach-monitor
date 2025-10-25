# Deployment Guide for Beach Monitor

This guide covers deploying the Beach Monitor chat UI to Streamlit Community Cloud.

## Prerequisites

1. GitHub account
2. Streamlit Community Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
3. API keys ready:
   - OpenAI API key
   - YouTube API key
   - AWS credentials (for S3 model access)

## Option 1: Streamlit Community Cloud (Recommended - 5 minutes) ⭐

### Step 1: Push to GitHub

Ensure your code is pushed to GitHub:

```bash
git add .
git commit -m "Add beach/water classification and deployment config"
git push origin main
```

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository: `beach-monitor`
4. Set main file path: `main.py`
5. Click "Advanced settings"

### Step 3: Configure Secrets

In the "Secrets" section, paste your environment variables in TOML format:

```toml
OPENAI_API_KEY = "sk-your-actual-key-here"
YOUTUBE_API_KEY = "your-youtube-key-here"
AWS_ACCESS_KEY_ID = "your-aws-key-here"
AWS_SECRET_ACCESS_KEY = "your-aws-secret-here"
S3_BUCKET_NAME = "your-bucket-name"
S3_MODEL_KEY = "path/to/model.pt"
```

### Step 4: Deploy

Click "Deploy!" and wait 2-3 minutes for the app to build and start.

Your app will be live at: `https://[your-app-name].streamlit.app`

### Updating the App

Any push to your main branch will automatically redeploy the app.

---

## Option 2: Hugging Face Spaces (Alternative - 10 minutes)

### Step 1: Create a Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose "Streamlit" as the SDK
4. Name your space (e.g., "beach-monitor")

### Step 2: Upload Files

Upload your project files or connect your GitHub repo.

### Step 3: Configure Secrets

In Space settings, add your secrets as environment variables:
- `OPENAI_API_KEY`
- `YOUTUBE_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `S3_BUCKET_NAME`
- `S3_MODEL_KEY`

### Step 4: Create app.py

Create an `app.py` file that imports and runs your main:

```python
from main import main
if __name__ == "__main__":
    main()
```

---

## Option 3: Railway.app (Production - 15 minutes)

### Step 1: Install Railway CLI

```bash
npm install -g @railway/cli
railway login
```

### Step 2: Initialize Project

```bash
railway init
railway link
```

### Step 3: Add Environment Variables

```bash
railway variables set OPENAI_API_KEY=sk-your-key
railway variables set YOUTUBE_API_KEY=your-key
railway variables set AWS_ACCESS_KEY_ID=your-key
railway variables set AWS_SECRET_ACCESS_KEY=your-key
railway variables set S3_BUCKET_NAME=your-bucket
railway variables set S3_MODEL_KEY=path/to/model.pt
```

### Step 4: Deploy

```bash
railway up
```

Your app will be deployed with a custom domain.

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Ensure all dependencies are in `requirements.txt` and `packages.txt` (for system packages).

### Issue: "API key not found"

**Solution**: Double-check that secrets are properly configured in your deployment platform's settings.

### Issue: OpenCV errors

**Solution**: The `packages.txt` file includes necessary system libraries. If issues persist, add:
```
libsm6
libxext6
libxrender-dev
```

### Issue: Out of memory

**Solution**: 
- Streamlit Cloud free tier has 1GB RAM limit
- Consider upgrading or using Railway/Render for more resources
- Disable location classification for lower memory usage:
  ```python
  detector = BeachDetector(enable_location_classification=False)
  ```

### Issue: Slow response times

**Solution**: 
- GPT-4V calls take 2-5 seconds
- Consider caching results
- Show loading spinner to improve UX (already implemented)

---

## Cost Estimates

### Streamlit Community Cloud
- **Free tier**: Unlimited public apps
- **Limitations**: 1GB RAM, shared CPU

### Hugging Face Spaces
- **Free tier**: CPU instances
- **Paid**: $0.60/hour for GPU (if needed)

### Railway.app
- **Free tier**: $5 credit/month
- **Paid**: ~$5-10/month for hobby projects

### API Costs (per 1000 requests)
- **GPT-4V**: ~$10 (at $0.01 per image)
- **YouTube API**: Free (10,000 requests/day)
- **AWS S3**: Minimal (~$0.01 for model downloads)

---

## Monitoring

### Streamlit Cloud
- Built-in logs and metrics dashboard
- View at: App menu → "Manage app" → "Logs"

### Custom Monitoring
Add to your app:
```python
import streamlit as st

# Track usage
if 'request_count' not in st.session_state:
    st.session_state.request_count = 0

st.session_state.request_count += 1
st.sidebar.metric("Total Requests", st.session_state.request_count)
```

---

## Security Best Practices

1. **Never commit API keys** - Use secrets management
2. **Use environment variables** - Keep `.env` in `.gitignore`
3. **Rotate keys regularly** - Especially if exposed
4. **Monitor usage** - Set up billing alerts for APIs
5. **Rate limiting** - Consider adding rate limits for public apps

---

## Next Steps

After deployment:

1. **Test thoroughly** - Try various queries
2. **Monitor costs** - Check OpenAI usage dashboard
3. **Gather feedback** - Share with users
4. **Iterate** - Add features based on usage patterns

For issues or questions, check the logs in your deployment platform's dashboard.
