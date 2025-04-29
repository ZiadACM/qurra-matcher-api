from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.audio_processor import AudioProcessor
from app.matcher import QuranMatcher
import os, tempfile, logging
from app.logger import AppLogger

logger = AppLogger.get_logger(__name__)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
matcher = QuranMatcher()

@app.post("/match-recitation")
async def match_recitation(audio_file: UploadFile):
    """Process uploaded audio and return matching verses"""
    temp_path = None
    wav_path = None
    try:
        # save upload
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            await audio_file.seek(0)
            tmp.write(await audio_file.read())
            temp_path = tmp.name
        # convert and transcribe
        wav_path = AudioProcessor.convert_to_wav(temp_path)
        transcription = matcher.transcribe_audio(wav_path)
        matches = matcher.find_matches(transcription)
        return {"original_transcription": transcription, "quran_matches": matches}
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        for path in (temp_path, wav_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as clean_err:
                    logger.warning(f"Failed to delete temp file {path}: {clean_err}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
