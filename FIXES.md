# Bug Fixes Applied

## Issue 1: Config File Encoding Error ✅ FIXED

**Error:**
```
'charmap' codec can't decode byte 0x81 in position 957: character maps to <undefined>
```

**Root Cause:**
- Windows defaults to 'charmap' encoding when opening files
- The config.yaml contains Unicode character ʻ in "Kāʻanapali"
- 'charmap' codec cannot decode this character

**Fix:**
- Added `encoding='utf-8'` parameter to `open()` in `beach_cv_tool/capture.py`
- Line 31: `with open(config_path, 'r', encoding='utf-8') as f:`

**File Changed:**
- `beach_cv_tool/capture.py`

---

## Issue 2: YouTube URL Extraction Failing ✅ FIXED

**Error:**
```
Error getting stream URL: expected string or bytes-like object, got 'NoneType'
❌ No video results found for 'https://www.youtube.com/watch?v=c3HxYfhhFu8'.
```

**Root Cause:**
- `get_youtube_livestream_url()` was designed to search YouTube
- Config provides a direct YouTube URL, not a search query
- Function tried to search for the URL string, found nothing, returned None

**Fix:**
- Modified function to detect if input is already a YouTube URL
- If it's a URL, return it directly (no API call needed)
- If it's a search query, use YouTube API to search
- Added URL detection: `if "youtube.com/watch" in search_query or "youtu.be/" in search_query`

**Benefits:**
- No unnecessary YouTube API calls
- Works with both direct URLs and search queries
- Saves API quota

**File Changed:**
- `beach_cv_tool/extract_url.py`

---

## Architecture Clarification ✅ DOCUMENTED

**Question:**
"Are you using my fine-tuned model to get bounding boxes and only asking the LLM how many of those people boxes are in the water vs beach?"

**Answer:**
Yes! The architecture is:

1. **Your fine-tuned YOLO model** → Detects people and boats → Returns bounding boxes
2. **GPT-4V (VLM)** → Takes those bounding boxes → Classifies each person as "beach" or "water"

**Why This Works:**
- YOLO is already trained on your beach data (accurate detection)
- VLM is good at spatial reasoning (beach vs water classification)
- No redundant detection (efficient and cost-effective)
- Numbers always match (same boxes used for both)

**Cost:**
- YOLO: Free (your model, your infrastructure)
- VLM: ~$0.01 per image (only when people detected)

**Files Updated:**
- `beach_cv_tool/location_classifier.py` - Added architecture comment
- `ARCHITECTURE.md` - Created detailed architecture documentation

---

## Summary of Changes

### Files Modified:
1. ✅ `beach_cv_tool/capture.py` - UTF-8 encoding fix
2. ✅ `beach_cv_tool/extract_url.py` - Direct URL support
3. ✅ `beach_cv_tool/location_classifier.py` - Architecture clarification

### Files Created:
1. ✅ `ARCHITECTURE.md` - Complete architecture documentation
2. ✅ `FIXES.md` - This file

---

## Testing Checklist

- [ ] Run `python main.py` - Should start without encoding errors
- [ ] Query: "How busy is the beach?" - Should capture and analyze
- [ ] Check logs - Should see "✅ Using direct YouTube URL"
- [ ] Verify counts - Should show beach/water breakdown
- [ ] Check cost - Only charged when people detected

---

## Next Steps

1. **Test locally:**
   ```bash
   python main.py
   ```

2. **Verify the flow:**
   - Config loads without encoding errors ✅
   - YouTube URL is used directly (no search) ✅
   - YOLO detects people ✅
   - VLM classifies locations ✅
   - Results show beach vs water counts ✅

3. **Deploy when ready:**
   - Follow `DEPLOYMENT.md` for Streamlit Cloud
   - Add secrets in Streamlit dashboard
   - Push to GitHub and deploy

---

## Known Limitations

1. **YouTube API Key**: Only needed if using search queries (not needed for direct URLs)
2. **VLM Cost**: ~$0.01 per image with people (can be disabled if needed)
3. **Response Time**: 2-5 seconds due to VLM API call (show loading spinner)

---

## Support

If you encounter issues:
1. Check logs for specific error messages
2. Verify all API keys are set in `.env` or Streamlit secrets
3. Ensure `encoding='utf-8'` is used for all file reads on Windows
4. Test with a direct YouTube URL first before using search queries
