import torch, json, re, requests, logging, torchaudio

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from rapidfuzz import process, fuzz
from app.logger import AppLogger

logger = AppLogger.get_logger(__name__)

class QuranMatcher:
    def __init__(self):
        self.model_id = "tarteel-ai/whisper-base-ar-quran"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing model on {self.device.upper()} device")

        # load Whisper model
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id).to(self.device)

        # load and prepare verses
        self.quran_data = self._load_quran_data()
        self.all_verses = self._prepare_verse_database()
        # cache normalized strings for matching
        self.normalized_verses = [v["normalized"] for v in self.all_verses]
        logger.info(f"Loaded {len(self.all_verses)} verses")

    def _load_quran_data(self):
        """Load Quran data from remote or fallback to local"""
        try:
            quran_url = "https://cdn.jsdelivr.net/npm/quran-json@3.1.2/dist/quran.json"
            resp = requests.get(quran_url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch Quran data remotely: {e}, using local backup")
            try:
                with open('quran_backup.json', 'r', encoding='utf-8') as f:
                    return json.load(f)
            except FileNotFoundError:
                raise RuntimeError("Quran data unavailable locally")

    def _prepare_verse_database(self):
        """Flatten and normalize all verses"""
        verses = []
        for chapter in self.quran_data:
            for verse in chapter.get("verses", []):
                orig = verse.get("text", "")
                verses.append({
                    "surah_num": chapter.get("id"),
                    "surah_name": chapter.get("name"),
                    "ayah_num": verse.get("id"),
                    "original": orig,
                    "normalized": self._normalize_text(orig)
                })
        return verses

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize Arabic text for robust matching"""
        # remove diacritics
        diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]')
        text = diacritics.sub('', text)
        # remove tatweel
        text = text.replace('ـ', '')
        # remove punctuation (keep only Arabic letters and spaces)
        text = re.sub(r'[^؀-ۿ\s]', '', text)

        # standardize variants
        variants = {
            'آ': 'ا', 'أ': 'ا', 'إ': 'ا', 'ٱ': 'ا',
            'ى': 'ي', 'ئ': 'ي', 'ؤ': 'و',
            'ة': 'ه', 'ﷲ': 'الله'
        }
        for old, new in variants.items():
            text = text.replace(old, new)

        # collapse whitespace
        return re.sub(r'\s+', ' ', text).strip()

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text using Whisper model"""
        try:
            wav, sr = torchaudio.load(audio_path)
            wav = wav.mean(dim=0)
            inputs = self.processor.feature_extractor(wav, sampling_rate=sr, return_tensors='pt')
            input_feats = inputs.input_features.to(self.device)
            pred_ids = self.model.generate(input_feats)
            return self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError("Audio transcription failed")

    def find_matches(self, transcription: str, top_n: int = 5, score_threshold: int = 30) -> list:
        """Find top matching Quran verses via fuzzy matching"""
        clean = self._normalize_text(transcription)
        logger.info(f"Normalized transcription: {clean}")

        # get fuzzy scores against precomputed normalized verses
        results = process.extract(clean, self.normalized_verses,
                                  scorer=fuzz.token_set_ratio,
                                  limit=top_n * 3)
        # sort by score descending
        sorted_res = sorted(results, key=lambda x: x[1], reverse=True)

        matches = []
        seen = set()
        for choice, score, idx in sorted_res:
            if score < score_threshold:
                break
            verse = self.all_verses[idx]
            key = (verse['surah_num'], verse['ayah_num'])
            if key in seen:
                continue
            seen.add(key)
            matches.append({
                'surah': verse['surah_name'],
                'ayah_number': verse['ayah_num'],
                'text': verse['original'],
                'confidence': f"{score}%"
            })
            if len(matches) >= top_n:
                break
        return matches