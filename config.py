#!/usr/bin/env python3

import torch
import rir_generator as rir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C = 343  # Sound velocity
FS = int(16e3)  # Sample rate [KHz]
SIGNAL_LEN_SECS = 0.6  # Fixed length of audio signals [s]
SIGNAL_LEN = int(SIGNAL_LEN_SECS * FS)  # Fixed length of audio signals [samples]
FRAME_LENGTH = 512 # K
NUMBER_OF_FRAMES = 96 # L

NUM_MICS = 8  # Number of microphones
ROOM_DIMENSIONS = (6.0, 6.0, 2.4)  # Room dimensions [x y z] (m)
MIC_ARRAY_POS = ((2.87, 1.0, 1.5), (2.9, 1.0, 1.5), (2.93, 1.0, 1.5), (2.96, 1.0, 1.5),
                 (3.04, 1.0, 1.5), (3.07, 1.0, 1.5), (3.1, 1.0, 1.5), (3.13, 1.0, 1.5))  # Mics' positions
MIC_ARRAY_CENTER = (3.0, 1.0, 1.5)
SHOULD_REVERSE_MICROPHONES = True
ANGLE_RES = 15  # Circular source setup's angle resolution [deg]
ANGLE_LOW = 0  # Circular source setup's lowest possible angle [deg]
ANGLE_HIGH = 180  # Circular source setup's highest possible angle [deg]
NUM_CLASSES = ((ANGLE_HIGH - ANGLE_LOW) // ANGLE_RES) + 1
SPEAKER_HEIGHT = MIC_ARRAY_CENTER[2] #torch.tensor(1.75)  # Height of each sound source [m]
EPS = 1e-9  # Epsilon to avoid 0 values in numerical calculations

ENABLE_HPF = True  # Enable HPF for RIRs
DIM = 3  # Room dimension (3 dimensional room)
ORDER = -1  # Reflection order (-1 is maximal order)
MTYPE = rir.mtype.omnidirectional  # Microphone type
NSAMPLE_COEF = 0.8  # Proportion between nsample and maximal (discrete) reverb time
MAX_REVERB_TIME = 0.4  # Maximal reverb time in project [s]
NSAMPLE = round(NSAMPLE_COEF * MAX_REVERB_TIME * FS)  # RIR length
ORIENTATION = (0, 0)  # Microphone orientation
REVERB_TAIL_LENGTH = int(FS * 0.16)

NFFT = FRAME_LENGTH  # FFT length in STFT
OVERLAP = 0.75  # Frame overlap in STFT
HOP_LENGTH = int((1 - OVERLAP) * NFFT)  # Hop length in STFT

TRAIN_REVERB_TIMES = (0.2, 0.3, 0.4)  # Training reverb times (T60) [s]
TRAIN_RAD_MEAN = 1.5  # Training mean speaker radius [m]
TRAIN_RAD_VAR = 0.3  # Training variance in speaker radius [m^2]
TRAIN_SIR_LOW = -2  # Lowest possible training signal-to-interference ratio [dB]
TRAIN_SIR_HIGH = 2  # Highest possible training signal-to-interference ratio [dB]
TEST_REVERB_TIMES = (0.16, 0.36)

TEST_RADII = (1.0, 2.0)  # Allowed radii for test set
# SIGNAL_LEN = NUMBER_OF_FRAMES * HOP_LENGTH + int(NFFT * (1 - OVERLAP)) # Fixed length of audio signals [samples]
MAX_WORKERS = 10