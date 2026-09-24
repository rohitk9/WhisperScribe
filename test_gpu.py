from faster_whisper import WhisperModel
import time

print("Testing faster_whisper on GPU (CUDA)...")
try:
    start_time = time.time()
    print("Initializing WhisperModel('base', device='cuda', compute_type='float16')...")
    model = WhisperModel("base", device="cuda", compute_type="float16")
    print(f"Model loaded successfully in {time.time() - start_time:.2f} seconds.")
    print("CUDA is working correctly for faster_whisper! Your GPU is ready to be used.")
except Exception as e:
    print(f"Error loading model on GPU: {e}")
    import traceback
    traceback.print_exc()
