```bash
chmod +x install.sh
./install.sh
```

Once the dependencies are installed, you can start the FastAPI server by executing the following command:
```bash
uvicorn main:app --reload
```

The FastAPI server will start running on `http://localhost:8000`. You can access the API endpoints by opening this URL in your web browser or sending HTTP requests to it using tools like cURL or Postman.

Run **send_audio.py** to send post request and get gop scores as json.