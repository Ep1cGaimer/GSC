import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from serving.local_server import app

if __name__ == '__main__':
    print("Starting server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)