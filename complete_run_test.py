import os
import uuid
import urllib.request
import urllib.error
import numpy as np
import cv2

# Verify server
url = 'http://127.0.0.1:5000'
try:
    resp = urllib.request.urlopen(url, timeout=10)
    print('SERVER_UP', resp.status)
except Exception as exc:
    print('SERVER_DOWN', repr(exc))
    raise SystemExit(1)

# Create temp image
upload_dir = os.path.join('static', 'uploads')
os.makedirs(upload_dir, exist_ok=True)
image_path = os.path.join(upload_dir, 'test_upload.jpg')
image = 255 * np.ones((256, 256, 3), dtype=np.uint8)
cv2.putText(image, 'TEST', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8, cv2.LINE_AA)
cv2.imwrite(image_path, image)

# Build multipart form data
boundary = '----WebKitFormBoundary' + uuid.uuid4().hex
body_lines = []
body_lines.append(f'--{boundary}')
body_lines.append('Content-Disposition: form-data; name="image"; filename="test_upload.jpg"')
body_lines.append('Content-Type: image/jpeg')
body_lines.append('')
body = '\r\n'.join(body_lines).encode('utf-8') + b'\r\n'
with open(image_path, 'rb') as f:
    body += f.read()
body += b'\r\n' + f'--{boundary}--\r\n'.encode('utf-8')

request = urllib.request.Request(url + '/detect', data=body, method='POST')
request.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
request.add_header('Content-Length', str(len(body)))

try:
    response = urllib.request.urlopen(request, timeout=60)
    content = response.read(512).decode('utf-8', errors='replace')
    print('DETECT_STATUS', response.status)
    print(content)
except urllib.error.HTTPError as exc:
    print('DETECT_HTTP_ERROR', exc.code)
    print(exc.read(512).decode('utf-8', errors='replace'))
except Exception as exc:
    print('DETECT_ERROR', repr(exc))
