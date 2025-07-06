## 🐳 Montreal Forced Aligner (MFA) — Docker Setup and Usage Guide


### Step 1: Install Docker

After installing, test with:

```bash
docker --version
```

---

### Step 2: Pull the MFA Docker Image

```bash
docker pull montrealcorpus/tools:latest
```

This pulls the latest official MFA Docker image. You can verify with:

```bash
docker images
```

---

### Step 3: Prepare Your Data Directory Structure (following basr18 dataset)


**Notes**:
* Audio files should be **16kHz mono WAV**.
* Transcriptions should match file names.
* Dictionary should cover all words in transcripts.

---

### Step 4: Run the MFA Docker Container Bash

### Step 5: Download the relevant dictionary and acoustic models.

### Step 6: Align using the dictionary and acoustic model mentioning the input directory and output directory