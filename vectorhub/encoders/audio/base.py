from ...import_utils import *

if is_all_dependency_installed('audio-encoder'):
    import librosa
    import soundfile as sf

import tempfile
import shutil
import os
from urllib.request import urlopen, Request
from urllib.parse import quote
import io
import numpy as np

from ...base import Base2Vec, catch_vector_errors

class BaseAudio2Vec(Base2Vec):
    def read(self, audio: str, new_sampling_rate: int = 16000):
        """An method to specify the read method to read the data.
        """
        if type(audio) == str:
            if 'http' in audio:
                fd, fp = tempfile.mkstemp()
                os.write(fd, urlopen(Request(quote(audio, safe=':/?*=\''),
                                             headers={'User-Agent': "Magic Browser"})).read())
                if '.mp3' in audio:
                    data, sampling_rate = librosa.load(fp, dtype='float32')
                else:
                    data, sampling_rate = sf.read(fp, dtype='float32')
                os.close(fd)
            else:
                data, sampling_rate = sf.read(audio, dtype='float32')
        elif type(audio) == bytes:
            data, sampling_rate = sf.read(io.BytesIO(audio), dtype='float32')
        elif type(audio) == io.BytesIO:
            data, sampling_rate = sf.read(audio, dtype='float32')
        return np.array(librosa.resample(data.T, sampling_rate, new_sampling_rate))
    
    @catch_vector_errors
    def bulk_encode(self, audios, vector_operation='mean'):
        return [self.encode(c, vector_operation) for c in audios]
