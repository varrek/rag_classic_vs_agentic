# Troubleshooting Guide

This guide addresses common issues you might encounter when running the RAG Comparison application.

## Common Issues

### Linux inotify Watch Limit Errors

**Symptoms:**
- Error message: `OSError: [Errno 28] inotify watch limit reached`
- Application crashes during startup or when loading files

**Solution:**
1. Run the included script to increase the inotify watch limit:
   ```bash
   sudo bash increase_inotify_watches.sh
   ```
2. Alternatively, manually increase the limit:
   ```bash
   echo fs.inotify.max_user_watches=65536 | sudo tee -a /etc/sysctl.conf
   sudo sysctl -p
   ```

### UI Not Updating After Processing

**Symptoms:**
- Processing completes, but results aren't displayed
- Questions remain disabled after processing
- UI appears frozen or unresponsive

**Solutions:**
1. Click the Reset button to clear application state
2. Try a different sample question
3. Restart the application using the restart script:
   ```bash
   ./restart_streamlit.sh
   ```
4. Check the terminal for error messages or logs

### PyTorch/CUDA Errors

**Symptoms:**
- Errors mentioning CUDA, torch, or out of memory
- Application crashes during similarity computation or evaluation

**Solutions:**
1. The application is configured to use CPU mode by default:
   ```python
   torch.set_grad_enabled(False)
   torch.set_num_threads(1)
   ```
2. If you encounter memory issues, try:
   - Restarting the application
   - Closing other memory-intensive applications
   - Reducing the number of retrieved documents (modify `k` in `get_retriever`)

### Knowledge Base Creation Fails

**Symptoms:**
- Error during knowledge base creation
- Progress bar hangs or fails to complete

**Solutions:**
1. Ensure your OpenAI API key is valid and has sufficient credits
2. Check that you have files in the `data` directory
3. Try creating a smaller knowledge base by reducing the number of files

### OpenAI API Key Issues

**Symptoms:**
- "API key not found" or authentication errors
- Retrievals or answer generation failing

**Solutions:**
1. Set the API key through the application UI when prompted
2. Export the API key in your terminal:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
3. Check if your API key has expired or has usage limits

### Streamlit-Specific Issues

**Symptoms:**
- "Address already in use" errors
- Multiple Streamlit processes running
- Application runs but isn't accessible in browser

**Solutions:**
1. Kill all existing Streamlit processes:
   ```bash
   pkill -f streamlit
   ```
2. Use a different port:
   ```bash
   streamlit run app.py --server.port 8506
   ```
3. Check if another application is using the default port (8501)

## Advanced Troubleshooting

### Check Logs

The application creates logs in:
- Terminal output
- `logs/streamlit_app.log` (if configured)

Review these logs for error messages or warnings.

### Debug Session State

Add debug print statements to inspect session state:
```python
st.write("Session State:", st.session_state)
```

### Test OpenAI API Connection

Verify your API connection works:
```bash
curl -X POST https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Check System Resources

Monitor system resources:
```bash
# Monitor memory usage
watch -n 1 free -m

# Check CPU usage
top -u $(whoami)

# View Streamlit processes
ps aux | grep streamlit
```

## Still Having Issues?

If you've tried the solutions above and still have problems:

1. Create an issue in the repository with:
   - Clear description of the problem
   - Steps to reproduce
   - Terminal output/logs
   - System information (OS, Python version)

2. Try running the simplified test app (`simple_test.py`) to isolate the issue:
   ```bash
   streamlit run simple_test.py
   ``` 