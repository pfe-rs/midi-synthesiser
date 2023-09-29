import numpy as np
import scipy.signal as signal
import sounddevice as sd
import threading

samplerate = 44.1e3

def release_window(t, k = 0.01):
    value = np.sqrt(1 - 1/k * t**2)
    return np.where(value > 0, value, 0)

def falloff_window(t):
    return (1 - 1/(1 + np.exp(-(1 * t - 1)))) / (1 - 1/(1 + np.exp(-(1 * 0 - 1))))

piano_window_defaults = (40, 2 * np.pi * 30)

def piano_response(t, b, wn):
    return np.exp(-b * t) * np.sin(wn * t + 1.5 * np.pi) + 1

timbre_defaults=[1.0,.3,.8,.2,.4,.2,.2,.1,.1]

def midi_to_freq(code):
    return 2**((code-69)/12.0) * 440

def make_note(f0, coeffs = timbre_defaults):
    return f0 * (np.arange(len(coeffs)) + 1)

class Nota:
    
    def __init__(self, freqs, coeffs = timbre_defaults, curr_phases = None, curr_time = 0, curr_release_time = 0, released = False) -> None:
        self.freqs = freqs
        self.coeffs = coeffs
        
        if curr_phases is not None:
            self.curr_phases = curr_phases
        else:
            self.curr_phases = np.zeros(len(freqs))
            
        self.curr_time = curr_time
        self.curr_release_time = curr_release_time
        self.released = released
        
        pass
    
def make_buffer(notes, samplerate = samplerate, buffer_size = 1024, ditter = 0.01):
    dt = 1 / samplerate
    ts = np.linspace(0, buffer_size * dt, buffer_size + 1)
    buffer = np.zeros(buffer_size + 1)
    
    new_notes = {}
    
    for freq, nota in notes.items():
                
        curr_phases = np.zeros(len(nota.curr_phases))
        
        for i in range(len(nota.freqs)):
        
            ps = nota.curr_phases[i] + 2 * np.pi * nota.freqs[i] * ts
            ps = np.random.uniform(1 - ditter / 100, 1 + ditter / 100, len(ps)) * ps
        
            if nota.released:
                buffer += nota.coeffs[i] * np.sin(ps) * piano_response(ts + nota.curr_time, *piano_window_defaults) \
                                                      * falloff_window(ts + nota.curr_time) \
                                                      * release_window(ts + nota.curr_release_time)
            else:
                buffer += nota.coeffs[i] * np.sin(ps) * piano_response(ts + nota.curr_time, *piano_window_defaults) \
                                                      * falloff_window(ts + nota.curr_time)
                             
            curr_phases[i] = ps[-1] % (2 * np.pi)
            
        curr_time = (buffer_size + 1) * dt + nota.curr_time
        if nota.released:
            curr_release_time = (buffer_size + 1) * dt + nota.curr_release_time
        
            if release_window(curr_release_time) > 0:
                new_notes[freq] = Nota(nota.freqs, nota.coeffs, curr_phases, curr_time, curr_release_time, nota.released)
        else:
            new_notes[freq] = Nota(nota.freqs, nota.coeffs, curr_phases, curr_time, nota.curr_release_time, nota.released)
        
    return (buffer[:-1], new_notes)


    (buffer, notes) = make_buffer(notes, samplerate, buffer_size, ditter)
        
keys = {
    "G" : 392.00,
    "H" : 440.00,
    "J" : 493.88,
}

from pynput import keyboard

def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()

# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()


event = threading.Event()

def callback(outdata, frames, time, status):
    global current_frame
    if status:
        print(status)
    chunksize = min(len(data) - current_frame, frames)
    outdata[:chunksize] = data[current_frame:current_frame + chunksize]
    if chunksize < frames:
        outdata[chunksize:] = 0
        raise sd.CallbackStop()

print(sd.query_devices())
"""
stream = sd.OutputStream(
    samplerate=fs, device=args.device, channels=data.shape[1],
    callback=callback, finished_callback=event.set)
with stream:
    event.wait()  # Wait until playback is finished
"""