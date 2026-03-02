# PodBot - AI Voice Tutor

PodBot is a real-time, high-speed AI voice tutor powered by Gemini 2.5 Flash. It features live transcription, natural male voice interaction, and topic-based learning.

## Deployment Instructions

### 1. Export the Code
- Use the **Export to GitHub** button in the AI Studio interface (if available).
- Alternatively, download the project as a ZIP file.

### 2. Push to GitHub
If you downloaded the ZIP:
1. Initialize a new git repository: `git init`
2. Add all files: `git add .`
3. Commit: `git commit -m "Initial commit"`
4. Create a new repo on GitHub and follow the instructions to push:
   ```bash
   git remote add origin https://github.com/your-username/podbot.git
   git branch -M main
   git push -u origin main
   ```

### 3. Host on Vercel
1. Go to [Vercel](https://vercel.com) and click **New Project**.
2. Import your GitHub repository.
3. Vercel should automatically detect the **Vite** framework.
4. **Environment Variables**:
   - Add a new environment variable named `VITE_GEMINI_API_KEY`.
   - Paste your Gemini API Key as the value.
5. Click **Deploy**.

## Local Development
1. Install dependencies: `npm install`
2. Run the dev server: `npm run dev`
3. Open `http://localhost:3000`

## Features
- **Live Transcription**: Real-time text display of your voice.
- **Natural Voice**: High-quality male voice (Puck) at 24kHz.
- **Ultra-Low Latency**: Optimized for fast back-and-forth conversation.
- **Topic-Based**: Works with any subject you choose.
