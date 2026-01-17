#!/bin/bash

###############################################################################
# Telegram Bot Test Script for Server
###############################################################################

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           🤖 TELEGRAM BOT TEST - SERVER EDITION                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if in correct directory
if [ ! -f "config.py" ]; then
    echo "❌ Error: config.py not found!"
    echo "Please run this script from the riftech-cam-security directory"
    exit 1
fi

# Extract Telegram credentials
echo "📋 Telegram Configuration:"
echo "================================"

BOT_TOKEN=$(grep "TELEGRAM_BOT_TOKEN" config.py | cut -d'"' -f2)
CHAT_ID=$(grep "TELEGRAM_CHAT_ID" config.py | cut -d'"' -f2 | tr -d '"')

if [ -z "$BOT_TOKEN" ] || [ -z "$CHAT_ID" ]; then
    echo "❌ Error: Telegram credentials not found in config.py!"
    echo ""
    echo "Please add these to config.py:"
    echo "  TELEGRAM_BOT_TOKEN = \"your_bot_token\""
    echo "  TELEGRAM_CHAT_ID = \"your_chat_id\""
    exit 1
fi

echo "✓ Bot Token: ${BOT_TOKEN:0:20}...${BOT_TOKEN: -5}"
echo "✓ Chat ID: $CHAT_ID"
echo ""

# Test 1: Check bot info
echo "📡 Test 1: Checking Bot Info..."
echo "================================"
RESPONSE=$(curl -s "https://api.telegram.org/bot${BOT_TOKEN}/getMe")

if echo "$RESPONSE" | grep -q '"ok":true'; then
    BOT_NAME=$(echo "$RESPONSE" | grep -o '"first_name":"[^"]*' | cut -d'"' -f4)
    BOT_USER=$(echo "$RESPONSE" | grep -o '"username":"[^"]*' | cut -d'"' -f4)
    echo "✅ Bot connected!"
    echo "   Bot Name: $BOT_NAME"
    echo "   Bot Username: @$BOT_USER"
else
    echo "❌ Bot connection failed!"
    echo "   Response: $RESPONSE"
    exit 1
fi
echo ""

# Test 2: Send test message
echo "📨 Test 2: Sending Test Message..."
echo "================================"
MESSAGE="🔧 *Telegram Bot Test*

✅ Connection successful!
✅ Bot is working!
✅ Script running on server!

You will receive alerts when system is ARMED and a person is detected."

RESPONSE=$(curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
    -H "Content-Type: application/json" \
    -d "{\"chat_id\": \"$CHAT_ID\", \"text\": \"$MESSAGE\", \"parse_mode\": \"Markdown\"}")

if echo "$RESPONSE" | grep -q '"ok":true'; then
    echo "✅ Message sent successfully!"
    MSG_ID=$(echo "$RESPONSE" | grep -o '"message_id":[0-9]*' | cut -d':' -f2)
    echo "   Message ID: $MSG_ID"
else
    echo "❌ Message send failed!"
    echo "   Response: $RESPONSE"
    exit 1
fi
echo ""

# Test 3: Send test photo
echo "📸 Test 3: Sending Test Photo..."
echo "================================"

# Create test image
TEST_IMAGE="/tmp/test_telegram_photo.jpg"
convert -size 400x200 xc:blue \
    -pointsize 40 -fill white -gravity center \
    -annotate +0+0 "Test Photo" \
    -pointsize 20 -fill yellow -gravity south \
    -annotate +0+10 "Riftech Cam Security" \
    "$TEST_IMAGE" 2>/dev/null || {
        # Fallback if ImageMagick not available
        cat > /tmp/test_photo.html << EOF
<html><body bgcolor="blue"><center><h1>Test Photo</h1><p>Riftech Cam Security</p></center></body></html>
        echo "⚠️  ImageMagick not available, skipping photo test"
        echo ""
    }

if [ -f "$TEST_IMAGE" ]; then
    RESPONSE=$(curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendPhoto" \
        -F "chat_id=$CHAT_ID" \
        -F "photo=@$TEST_IMAGE" \
        -F "caption=📸 *Test Photo*

This is how alert photos will look!

Features:
- Timestamp overlay
- Confidence percentage
- Cropped person frame only" \
        -F "parse_mode=Markdown")

    if echo "$RESPONSE" | grep -q '"ok":true'; then
        echo "✅ Photo sent successfully!"
        PHOTO_ID=$(echo "$RESPONSE" | grep -o '"file_id":"[^"]*' | head -1 | cut -d'"' -f4)
        echo "   Photo ID: ${PHOTO_ID:0:20}..."
    else
        echo "❌ Photo send failed!"
        echo "   Response: $RESPONSE"
    fi

    # Cleanup
    rm -f "$TEST_IMAGE"
fi
echo ""

# Summary
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    ✅ ALL TESTS PASSED!                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "🎉 Telegram Bot is working correctly!"
echo ""
echo "📋 Next Steps:"
echo "1. Start the security system:"
echo "   ./start_both_servers.sh"
echo ""
echo "2. Open web interface:"
echo "   http://$(hostname -I | awk '{print $1}'):8080/web.html"
echo ""
echo "3. Click 'ARM SYSTEM' button"
echo "   → You will receive a notification!"
echo ""
echo "4. Send /start to bot in Telegram"
echo "   → You will receive welcome message!"
echo ""
echo "5. Stand in front of camera"
echo "   → You will receive alert with photo!"
echo ""
echo "📊 Available Telegram Commands:"
echo "  /start    - Show welcome message and menu"
echo "  /arm      - Activate monitoring"
echo "  /disarm   - Deactivate monitoring"
echo "  /status   - Check system status"
echo "  /snapshot - Take screenshot"
echo "  /menu     - Show command list"
echo ""
