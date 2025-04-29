import subprocess, tempfile, os, logging
from app.logger import AppLogger

logger = AppLogger.get_logger(__name__)

class AudioProcessor:
    @staticmethod
    def convert_to_wav(input_file, output_dir=None):
        """Convert any audio file to 16kHz mono WAV using ffmpeg"""
        try:
            if output_dir is None:
                output_dir = tempfile.mkdtemp()

            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "processed_audio.wav")

            # FFmpeg command to convert to mono, 16kHz WAV
            command = [
                "ffmpeg",
                "-y",                   # Overwrite output file if exists
                "-i", input_file,      # Input file
                "-ac", "1",            # Mono
                "-ar", "16000",        # Sample rate 16kHz
                output_path
            ]

            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed:\n{e.stderr.decode()}")
            raise RuntimeError("Audio conversion failed: unsupported format or ffmpeg error")