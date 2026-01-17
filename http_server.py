#!/usr/bin/env python3
"""Simple HTTP server untuk melayani web interface."""

import http.server
import socketserver
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PORT = 8080
DIRECTORY = os.getcwd()  # Gunakan direktori kerja saat ini

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler untuk proper MIME types dan CORS."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        """Add CORS headers."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        # Get requested path
        path = self.path.lstrip('/')
        
        # Default to web.html if root
        if not path or path == '/':
            path = 'web.html'
        
        # Check if file exists
        file_path = os.path.join(DIRECTORY, path)
        
        if not os.path.exists(file_path):
            self.send_error(404, "File Not Found")
            return
        
        # Serve file
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Set proper MIME type
            if path.endswith('.html'):
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(content)
            elif path.endswith('.css'):
                self.send_response(200)
                self.send_header('Content-type', 'text/css; charset=utf-8')
                self.end_headers()
                self.wfile.write(content)
            elif path.endswith('.js'):
                self.send_response(200)
                self.send_header('Content-type', 'application/javascript; charset=utf-8')
                self.end_headers()
                self.wfile.write(content)
            elif path.endswith('.svg'):
                self.send_response(200)
                self.send_header('Content-type', 'image/svg+xml')
                self.end_headers()
                self.wfile.write(content)
            elif path.endswith('.jpg') or path.endswith('.jpeg'):
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()
                self.wfile.write(content)
            elif path.endswith('.png'):
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(200)
                self.send_header('Content-type', 'application/octet-stream')
                self.end_headers()
                self.wfile.write(content)
                
        except Exception as e:
            logging.error(f"Error serving {path}: {e}")
            self.send_error(500, "Internal Server Error")
    
    def log_message(self, format, *args):
        """Custom logging."""
        logging.info(f"{self.address_string()} - {format % args}")


def main():
    """Start HTTP server."""
    try:
        with socketserver.TCPServer(("0.0.0.0", PORT), CustomHTTPRequestHandler) as httpd:
            logging.info(f"[HTTP] Server started on http://0.0.0.0:{PORT}")
            logging.info(f"[HTTP] Serving directory: {DIRECTORY}")
            logging.info(f"[HTTP] Access web interface at: http://YOUR_SERVER_IP:{PORT}/web.html")
            logging.info(f"[HTTP] Press Ctrl+C to stop")
            httpd.serve_forever()
    except KeyboardInterrupt:
        logging.info("\n[HTTP] Server stopped by user")
    except OSError as e:
        if e.errno == 98:  # Address already in use
            logging.error(f"[HTTP] Port {PORT} already in use!")
            logging.error(f"[HTTP] Stop other process using port {PORT} or use different port")
        else:
            logging.error(f"[HTTP] Error: {e}")
    except Exception as e:
        logging.error(f"[HTTP] Fatal error: {e}")


if __name__ == "__main__":
    main()
