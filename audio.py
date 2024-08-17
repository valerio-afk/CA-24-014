import numpy as np
import pyaudio



class AudioDriver:

    def __init__(this, n,low,high,sample_rate):
        this._audio_waves = {}
        for i in range(n):
            f = ((i / (n-1)) * (high - low)) + low

            t = np.linspace(0, 1, int(sample_rate), endpoint=False)


            this._audio_waves[i] = np.sin(2 * np.pi * f * t) * np.sin(2 * np.pi * f * t)


        this._current_wave = None

        p = pyaudio.PyAudio()
        this._idx = 0

        this._stream = p.open(format=pyaudio.paFloat32,rate=sample_rate,channels=1,output=True,stream_callback=this._play)

    def _play(this, in_data, frame_count, *args, **kwargs):

        wave = np.zeros((frame_count,)) if this._current_wave is None else this._current_wave

        output = np.take(wave,np.asarray(range(this._idx,this._idx+frame_count)), mode='wrap')
        this._idx += frame_count



        output = output.astype(np.float32).tobytes()

        return (output, pyaudio.paContinue)


    def mix(this,*args):
        if (len(args)==0):
            this._current_wave = None
        else:

            wave = np.zeros((len(this._audio_waves[0])))

            for val in args:
                wave += this._audio_waves[val]

            this._current_wave = wave





        this._idx = 0


    def stop(this):

        if (this._stream is not None):
            this._stream.stop_stream()
            this._stream.close()

        this._stream = None
