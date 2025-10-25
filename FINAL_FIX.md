# Final Fix - YouTube Search Query

## What Changed

### Problem
- Code was trying to use direct YouTube URLs
- You want to always search YouTube for the latest livestream
- This ensures you always get the current live stream, not an old URL

### Solution
Simplified the code to **always search YouTube** using the query string.

---

## Files Modified

### 1. `beach_cv_tool/extract_url.py`
**Removed:** URL detection logic (unnecessary complexity)  
**Result:** Clean, simple function that always searches YouTube

```python
def get_youtube_livestream_url(search_query: str) -> str | None:
    """
    Searches YouTube using the official API and returns the URL of the first result.
    """
    # Always searches - no URL detection
```

### 2. `config.yaml`
**Changed:** `stream_url` from direct URL to search query

```yaml
camera:
  stream_url: "LIVE 24/7 4K MAUI LIVE CAM WhalerCondo.net"  # Search query
```

---

## How It Works Now

```
1. Config provides search query: "LIVE 24/7 4K MAUI LIVE CAM WhalerCondo.net"
   ↓
2. BeachCapture calls get_youtube_livestream_url(query)
   ↓
3. Function searches YouTube API for the query
   ↓
4. Returns URL of top result (latest livestream)
   ↓
5. YouTubeCapture uses that URL to capture frames
```

---

## Benefits

✅ **Always current** - Gets the latest livestream, not an old URL  
✅ **Simple** - No complex URL detection logic  
✅ **Reliable** - YouTube search finds the active stream  
✅ **Flexible** - Easy to change search query in config  

---

## Test It

```bash
# Test the search function directly
python beach_cv_tool/extract_url.py

# Should output:
# ✅ Found video: [Video Title]
# YouTube Stream URL: https://www.youtube.com/watch?v=...
```

```bash
# Run the full app
python main.py

# Should now work without errors
```

---

## Cost Note

**YouTube API Usage:**
- 1 search per app startup = 100 quota units
- Daily quota: 10,000 units = 100 searches/day
- **Free tier is plenty for your use case**

To reduce API calls, the search only happens once when BeachCapture initializes.

---

## Summary

The code is now **simpler and more reliable**:
- ❌ Removed: Complex URL detection
- ❌ Removed: Direct URL support
- ✅ Kept: Simple YouTube search
- ✅ Result: Always gets the current livestream
