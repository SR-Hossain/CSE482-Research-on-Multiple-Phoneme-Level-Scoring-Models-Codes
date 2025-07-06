import json
import requests

# URL of the endpoint
url = 'http://localhost:8000/upload-audio'

file_path = '/mnt/Download/Downloads/2024_06_02_13_12_05.wav'

# The transcript string
transcript = 'This is a demo transcript.'
# transcript = 'rain pours down heavily Lightning flashes brightly trees sway wildly Streets flood quickly People stay indoors windows rattle hard'

# Open the file in binary mode
with open(file_path, 'rb') as f:
    response = requests.post(url, files={'audio': (file_path, f)}, data={'transcript': transcript})

# Print the response from the server
print(json.dumps(eval(response.text), indent=4, ensure_ascii=False))


