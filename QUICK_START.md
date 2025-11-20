# Quick Start Guide - WhatsApp Analyzer

## What's New

The application has been significantly improved with:
- ✨ **Professional UI** - Clean, modern design
- 🎯 **Better answers** - Natural language without message citations
- 🔍 **Optional citations** - Click to see source messages
- 🛡️ **Robust error handling** - Better error messages and recovery
- 📚 **More context** - Increased from 5 to 50 messages by default

## Getting Started

### 1. Export Your WhatsApp Chat

**For Group Chats:**
1. Open WhatsApp → Select group
2. Menu (⋮) → More → Export Chat → Without Media
3. Save the `.txt` file

**For Direct Messages:**
1. Open WhatsApp → Select contact
2. Menu (⋮) → More → Export Chat → Without Media
3. Save the `.txt` file

### 2. Upload & Index

1. Click **📁 Upload Chat** in the sidebar
2. Select your exported `.txt` file
3. Wait for the success message
4. Click **🚀 Index Chat Data** to process
5. Wait for indexing to complete

### 3. Explore Your Chat

#### **📊 Analytics Tab**
- View message statistics
- See most active users
- Explore word clouds and activity patterns
- Select specific users for detailed analysis

#### **❓ Q&A Tab**
- Ask natural language questions about your chat
- Get AI-powered answers with optional source citations
- Adjust retrieval settings (5-100 messages)
- View question history
- Toggle citations visibility

#### **💭 Sentiment Tab**
- Analyze emotional tone of conversations
- View sentiment over time
- Analyze by user or overall
- See emotional patterns and insights

#### **📚 Topics Tab**
- Extract main discussion themes
- See topic distribution
- Understand what people talk about most
- Analyze topic sentiment

## Tips for Better Results

### Questions to Ask

✅ **Good questions:**
- "What does Didi talk about the most?"
- "When is the deadline?"
- "What did we decide about the project?"
- "Who is the most active person?"
- "What was discussed on [date]?"

❌ **Avoid:**
- Very vague questions
- Questions about future events
- Questions about information not in the chat

### Improving Answers

1. **Select a date range** to focus analysis on specific periods
2. **Increase retrieved messages** using the slider (up to 100)
3. **Adjust context size** based on chat length
4. **Use specific keywords** when asking questions
5. **View citations** for source verification

## Understanding the Metrics

- **Confidence**: How confident the model is in the answer (0-100%)
- **Sources**: Number of chat messages used to generate the answer
- **Retrieved**: How many messages were searched from the database

## Troubleshooting

### Chat data won't upload
- Check file format (must be `.txt`)
- Ensure proper WhatsApp export format
- Try re-exporting the chat

### Answers seem incomplete
- Increase "messages to retrieve" slider
- Try a more specific question
- Check your date range selection

### Slow response
- Reduce "messages to retrieve" slider
- Check your internet connection
- Try a simpler question

### No results found
- Verify your date range
- Check if data was indexed properly
- Try different keywords

## Features Explained

### RAG (Retrieval-Augmented Generation)
The system finds relevant messages from your chat and uses AI to answer questions. This ensures answers are based on actual conversations.

### Sentiment Analysis
Analyzes emotional tone (positive, neutral, negative) of messages over time to track conversation mood.

### Topic Modeling
Uses AI to automatically identify main discussion themes in your conversations.

### Vector Database
Stores embeddings of messages for fast semantic search, allowing questions like "What did we talk about XYZ?"

## Configuration Options

### RAG Settings
- **Date Range**: Filter to specific time periods
- **Messages to Retrieve**: 5 (fast) to 100 (comprehensive)
- **Show Citations**: Optional toggle to view source messages

### Analysis Settings
- **Analysis Type**: Overall, By User, or By Topic
- **Date Range**: Custom or preset ranges
- **Number of Topics**: 2-10 topics for extraction

## Privacy & Data

- All processing happens locally on your machine
- Chat data is not stored or transmitted
- Vector embeddings are cached locally for speed
- You can delete the index anytime to clear data

## Performance Tips

1. **Indexing**: First-time indexing may take a few minutes
2. **Large chats**: 10k+ messages may be slower - consider date ranges
3. **Repeated queries**: Use Q&A history to avoid re-indexing
4. **Analytics**: Large date ranges with many users may take longer

## Getting Help

If something doesn't work:
1. Check the error message (usually shown in red boxes)
2. Try re-indexing your data
3. Use a smaller date range to test
4. Check that your WhatsApp export is valid

## Advanced Features

### Custom Date Ranges
- Select "Custom" to choose specific start/end dates
- Useful for analyzing specific events or time periods

### User-Specific Analysis
- Select a user from the dropdown to focus on their conversations
- Great for understanding individual communication patterns

### Export Results
- Copy sentiment analysis as text
- Download topic reports
- Screenshot visualizations

---

**Enjoy analyzing your conversations! 🎉**
