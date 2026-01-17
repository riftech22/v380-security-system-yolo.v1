#!/bin/bash

###############################################################################
# Debug Script untuk Telegram Bot di Server
###############################################################################

echo ""
echo "╔═════════════════════════════════════════════════════════════╗"
echo "║           🔍 TELEGRAM BOT DEBUG SCRIPT                          ║"
echo "╚═════════════════════════════════════════════════════════════╝"
echo ""

# Check 1: Check running processes
echo "📌 Step 1: Check Running Processes"
echo "=========================================="
echo "Checking for web_server.py processes:"
ps aux | grep -E "web_server.py|python" | grep -v grep | awk '{print "  PID:", $2, "CMD:", $11, $12, $13, $14, $15}'
echo ""

# Check 2: Check logs directory
echo "📌 Step 2: Check Logs Directory"
echo "=========================================="
if [ -d "logs" ]; then
    echo "✓ Logs directory exists"
    echo "  Files in logs:"
    ls -lh logs/ 2>/dev/null || echo "  (empty or permission denied)"
else
    echo "❌ Logs directory NOT found!"
    echo "  Creating logs directory..."
    mkdir -p logs
    echo "  ✓ Logs directory created"
fi
echo ""

# Check 3: Find log files
echo "📌 Step 3: Find Log Files"
echo "=========================================="
echo "Searching for log files:"
find . -name "*.log" -type f 2>/dev/null | head -10
echo ""

# Check 4: Check recent log entries
echo "📌 Step 4: Check Recent Log Entries"
echo "=========================================="
for log_file in logs/*.log 2>/dev/null; do
    if [ -f "$log_file" ]; then
        echo ""
        echo "📄 File: $log_file"
        echo "  Size: $(du -h "$log_file" | cut -f1)"
        echo "  Last 10 lines:"
        tail -10 "$log_file" | sed 's/^/    /'
    fi
done

# If no log files found
if ! ls logs/*.log 2>/dev/null; then
    echo "❌ No log files found in logs/ directory"
    echo ""
    echo "Possible reasons:"
    echo "  - Logging is disabled"
    echo "  - Logs written to stdout/stderr"
    echo "  - Logs in different location"
fi
echo ""

# Check 5: Check systemd journal
echo "📌 Step 5: Check Systemd Journal"
echo "=========================================="
echo "Checking systemd logs for security-system:"
journalctl -u security-system-v380 -n 50 --no-pager 2>/dev/null | tail -20 || echo "  Not running as systemd service"
echo ""

# Check 6: Test Telegram manually
echo "📌 Step 6: Test Telegram API Manually"
echo "=========================================="

# Get credentials
BOT_TOKEN=$(grep "TELEGRAM_BOT_TOKEN" config.py | cut -d'"' -f2)
CHAT_ID=$(grep "TELEGRAM_CHAT_ID" config.py | cut -d'"' -f2 | tr -d '"')

if [ -z "$BOT_TOKEN" ] || [ -z "$CHAT_ID" ]; then
    echo "❌ Telegram credentials not found in config.py"
    echo ""
    exit 1
fi

echo "Testing getMe..."
RESPONSE=$(curl -s "https://api.telegram.org/bot${BOT_TOKEN}/getMe")
if echo "$RESPONSE" | grep -q '"ok":true'; then
    echo "  ✅ Bot API accessible"
    echo "  Bot: $(echo "$RESPONSE" | grep -o '"username":"[^"]*' | cut -d'"' -f4)"
else
    echo "  ❌ Bot API not accessible"
    echo "  Response: $RESPONSE"
fi
echo ""

echo "Testing getUpdates (check for pending messages)..."
RESPONSE=$(curl -s "https://api.telegram.org/bot${BOT_TOKEN}/getUpdates")
if echo "$RESPONSE" | grep -q '"ok":true'; then
    echo "  ✅ Can read messages"
    UPDATE_COUNT=$(echo "$RESPONSE" | grep -o '"update_id":[0-9]*' | wc -l)
    echo "  Pending updates: $UPDATE_COUNT"
else
    echo "  ❌ Cannot read messages"
    echo "  Response: $RESPONSE"
fi
echo ""

# Check 7: Send test message to check bot
echo "📌 Step 7: Send Test Message to Bot"
echo "=========================================="
MESSAGE="🔧 *Debug Test*

This is a debug test message to verify bot can send messages.

Time: $(date)"

RESPONSE=$(curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
    -H "Content-Type: application/json" \
    -d "{\"chat_id\": \"$CHAT_ID\", \"text\": \"$MESSAGE\", \"parse_mode\": \"Markdown\"}")

if echo "$RESPONSE" | grep -q '"ok":true'; then
    echo "  ✅ Bot can send messages"
    echo "  Message ID: $(echo "$RESPONSE" | grep -o '"message_id":[0-9]*' | cut -d':' -f2)"
    echo ""
    echo "  Check your Telegram app for this message!"
else
    echo "  ❌ Bot cannot send messages"
    echo "  Response: $RESPONSE"
fi
echo ""

# Check 8: Kill duplicate processes
echo "📌 Step 8: Check for Duplicate Processes"
echo "=========================================="
PROCESS_COUNT=$(ps aux | grep "web_server.py" | grep -v grep | wc -l)
if [ $PROCESS_COUNT -gt 1 ]; then
    echo "⚠️  Found $PROCESS_COUNT web_server.py processes (should be 1)"
    echo ""
    echo "Do you want to stop all web_server processes and restart? (y/n)"
    read -r -t 10 response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Stopping all web_server processes..."
        pkill -f "web_server.py"
        sleep 2
        echo "✓ All processes stopped"
        echo ""
        echo "Run ./start_both_servers.sh to restart"
    fi
else
    echo "✓ No duplicate processes found"
fi
echo ""

# Summary
echo "╔═════════════════════════════════════════════════════════════╗"
echo "║                    DEBUG COMPLETE                           ║"
echo "╚═════════════════════════════════════════════════════════════╝"
echo ""

echo "📋 Recommendations:"
echo "1. Check logs above for Telegram initialization messages"
echo "2. Look for 'Telegram Bot enabled' or 'Telegram enabled'"
echo "3. If Telegram is not enabled, check config.py credentials"
echo "4. If multiple processes found, stop all and restart"
echo "5. Test Telegram commands: /start, /status, /arm"
echo ""
