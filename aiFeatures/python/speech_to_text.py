import speech_recognition as sr
import threading
import time
from queue import Queue, Empty

# Global variables for better control
stop_listening = False
recognition_active = False
result_queue = Queue()

def speech_to_text():
    """Converts speech to text using SpeechRecognition with better error handling"""
    global stop_listening, recognition_active
    
    stop_listening = False
    recognition_active = True
    
    recognizer = sr.Recognizer()
    
    # Adjust recognizer settings for better performance
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.phrase_threshold = 0.3
    
    try:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise... Please wait.", flush=True)
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            if stop_listening:
                recognition_active = False
                return ""
            
            print("Listening... Speak now!", flush=True)
            
            # Listen with timeout and phrase limit
            audio = recognizer.listen(
                source, 
                timeout=10,  # Total timeout
                phrase_time_limit=8  # Max phrase length
            )
            
            # Check if stop was requested during listening
            if stop_listening:
                recognition_active = False
                return ""
            
            print("Processing speech... Please wait.", flush=True)
            
            # Try Google Speech Recognition first
            try:
                query = recognizer.recognize_google(audio, language="en-US")
                print(f"Recognized: {query}", flush=True)
                recognition_active = False
                return query.strip()
                
            except sr.UnknownValueError:
                # Try alternative recognition if Google fails
                try:
                    query = recognizer.recognize_sphinx(audio)
                    print(f"Recognized (Sphinx): {query}", flush=True)
                    recognition_active = False
                    return query.strip()
                except:
                    pass
                    
                print("Sorry, I couldn't understand what you said. Please try again.", flush=True)
                recognition_active = False
                return ""
                
    except sr.WaitTimeoutError:
        print("Listening timeout - no speech detected.", flush=True)
        recognition_active = False
        return ""
        
    except sr.RequestError as e:
        print(f"Could not request results from speech recognition service; {e}", flush=True)
        recognition_active = False
        return ""
        
    except Exception as e:
        print(f"An error occurred during speech recognition: {e}", flush=True)
        recognition_active = False
        return ""

def speech_to_text_async():
    """Asynchronous version that puts result in queue"""
    global result_queue
    
    def recognition_worker():
        result = speech_to_text()
        result_queue.put(result)
    
    # Start recognition in separate thread
    recognition_thread = threading.Thread(target=recognition_worker, daemon=True)
    recognition_thread.start()
    
    return recognition_thread

def get_recognition_result(timeout=1):
    """Get result from async recognition"""
    global result_queue
    
    try:
        result = result_queue.get(timeout=timeout)
        return result
    except Empty:
        return None

def stop_speech_recognition():
    """Stops the speech recognition process"""
    global stop_listening, recognition_active
    
    stop_listening = True
    print("Speech recognition stop requested...", flush=True)
    
    # Wait a moment for recognition to stop
    timeout = 5
    while recognition_active and timeout > 0:
        time.sleep(0.1)
        timeout -= 0.1
    
    if not recognition_active:
        print("Speech recognition stopped successfully.", flush=True)
        return True
    else:
        print("Speech recognition may still be active.", flush=True)
        return False

def is_recognition_active():
    """Check if speech recognition is currently active"""
    global recognition_active
    return recognition_active

def speech_to_text_with_feedback(feedback_callback=None):
    """Speech recognition with status feedback for UI integration"""
    global stop_listening, recognition_active
    
    def status_update(message):
        if feedback_callback:
            feedback_callback(message)
        else:
            print(message, flush=True)
    
    stop_listening = False
    recognition_active = True
    
    recognizer = sr.Recognizer()
    
    # IMPROVED: Better microphone settings
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0  # Slightly longer pause
    recognizer.phrase_threshold = 0.3
    
    try:
        with sr.Microphone() as source:
            status_update("🎤 Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1.5)
            
            if stop_listening:
                recognition_active = False
                return ""
            
            status_update("🔴 Listening... Speak clearly!")
            
            # IMPROVED: Better timeout handling
            audio = recognizer.listen(
                source, 
                timeout=12,  # Slightly longer timeout
                phrase_time_limit=10  # Allow longer phrases
            )
            
            if stop_listening:
                recognition_active = False
                return ""
            
            status_update("🔄 Processing speech...")
            
            # Try Google Speech Recognition
            query = recognizer.recognize_google(audio, language="en-US")
            status_update(f"✅ Recognized: {query}")
            
            recognition_active = False
            return query.strip()
            
    except sr.WaitTimeoutError:
        status_update("⏰ Timeout - No speech detected. Please try again.")
        recognition_active = False
        return ""
        
    except sr.UnknownValueError:
        status_update("❓ Could not understand speech. Please speak clearly and try again.")
        recognition_active = False
        return ""
        
    except sr.RequestError as e:
        status_update(f"❌ Service error: {str(e)}")
        recognition_active = False
        return ""
        
    except Exception as e:
        status_update(f"❌ Unexpected error: {str(e)}")
        recognition_active = False
        return ""

# Test Run
if __name__ == "__main__":
    print("Starting speech recognition test...")
    
    def test_callback(message):
        print(f"Status: {message}")
    
    result = speech_to_text_with_feedback(test_callback)
    print(f"Final result: '{result}'")