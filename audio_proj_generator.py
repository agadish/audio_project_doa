import rir_generator as rir
import os
import torch
import torchaudio
from torchaudio.transforms import Resample, Spectrogram
from typing import List, Tuple, Dict
import pandas as pd
from torch.utils.data import Dataset
import scipy
from dataclasses import dataclass
from functools import cached_property

from config import ANGLE_HIGH, ANGLE_LOW, ANGLE_RES, C, DIM, ENABLE_HPF, EPS, HOP_LENGTH, MIC_ARRAY_CENTER, MIC_ARRAY_POS, MTYPE, NFFT, NSAMPLE, NUM_MICS, ORDER, ORIENTATION, SIGNAL_LEN, SPEAKER_HEIGHT, TEST_RADII, TEST_REVERB_TIMES, TRAIN_RAD_MEAN, TRAIN_RAD_VAR, TRAIN_REVERB_TIMES, TRAIN_SIR_HIGH, TRAIN_SIR_LOW, FS, L


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMIT_RIR_PATH = 'timit_rir'

def load_source_signals(source_dir='source_signals/LibriSpeech/dev-clean', batch_size=64, normalize=True):
    """
    Load batch_size random pairs of source signals of different speakers.
    Truncates or pads source signals as needed, to achieve the desired signal length SIGNAL_LEN.
    :param source_dir: Directory containing speaker folders, each containing text folders with audio files
    :param batch_size: Number of samples in the batch
    :param normalize: If true, normalize each signal to zero mean to unit variance
    :return: Tensor of source signals of shape (batch_size, 2, signal_length)
    """

    # Get list of speaker folders
    speakers = os.listdir(source_dir)
    num_speakers = len(speakers)

    # Initialize list to store source signals
    batch_signals = []

    # Iterate over batch size
    for _ in range(batch_size):
        # Randomly select two different speakers
        selected_speakers = torch.randint(num_speakers, (2,))
        while selected_speakers[0] == selected_speakers[1]:
            selected_speakers = torch.randint(num_speakers, (2,))

        # Initialize list to store signals for each speaker
        speaker_signals = []

        # Iterate over selected speakers
        for speaker_idx in selected_speakers:
            speaker = speakers[speaker_idx]
            speaker_dir = os.path.join(source_dir, speaker)
            text_folders = os.listdir(speaker_dir)
            text_folder = torch.randint(len(text_folders), (1,)).item()
            audio_files = [f for f in os.listdir(os.path.join(speaker_dir, text_folders[text_folder]))
                           if f.endswith('.flac')]

            # Randomly select an audio file
            audio_file = torch.randint(len(audio_files), (1,)).item()

            # Load audio file
            waveform, sample_rate = torchaudio.load(
                os.path.join(speaker_dir, text_folders[text_folder], audio_files[audio_file]))

            # If waveform is too long, select a random segment of SIGNAL_LEN samples
            if waveform.size(1) > SIGNAL_LEN:
                start_idx = torch.randint(waveform.size(1) - SIGNAL_LEN + 1, (1,)).item()
                waveform = waveform[:, start_idx:start_idx + SIGNAL_LEN]

            # If waveform is too short, pad to SIGNAL_LEN samples
            elif waveform.size(1) < SIGNAL_LEN:
                padding = torch.zeros(1, SIGNAL_LEN - waveform.size(1))
                waveform = torch.cat((waveform, padding), dim=1)

            # Resample if needed
            if sample_rate != FS:
                waveform = torchaudio.transforms.Resample(sample_rate, FS)(waveform)

            # Append waveform to list
            speaker_signals.append(waveform)

        # Stack signals for each speaker along the second dimension
        batch_signals.append(torch.stack(speaker_signals))

    # Stack signals for each sample along the first dimension
    batch_signals = torch.stack(batch_signals).squeeze(dim=2)

    # If Normalize is True, normalize each signal to zero mean and unit variance
    if normalize:
        batch_signals = ((batch_signals - torch.mean(batch_signals, dim=2).unsqueeze(-1)) /
                         torch.std(batch_signals, dim=2).unsqueeze(-1))

    return batch_signals


def generate_coords(num_scenarios, radii, variance):
    """
    Generate pairs of coordinates and pairs of DOAs for given number
    of scenarios, radii, and microphone array center.

    Args:
    - num_scenarios (int): Number of scenarios to generate pairs of coordinates for.
    - radii (list or torch.Tensor): List or tensor of radii.
    - variance (float): Variance of Gaussian noise to perturb radius.

    Returns:
    - coordinates (torch.Tensor): Tensor of shape (num_scenarios, 2, 2) containing
      pairs of (x, y) coordinates for each scenario.
    - DOAs (torch.Tensor): Tensor of shape (num_scenarios, 2) containing pairs of DOAs.
    """

    # Generate random angles
    angles = torch.randint(ANGLE_LOW, int((ANGLE_HIGH - ANGLE_LOW) / ANGLE_RES) + 1,
                           size=(num_scenarios, 2)) * ANGLE_RES

    # Convert angles to radians
    angles_rad = angles.float() * (torch.pi / 180.0)

    # If only one radius given, set it for all samples
    if radii.numel() == 1:
        radii_tensor = torch.full(size=(num_scenarios,), fill_value=radii.item(), dtype=torch.float32)
    # Else, pick a radius for each sample
    else:
        radii_tensor = torch.tensor([radii[i % len(radii)] for i in range(num_scenarios)], dtype=torch.float32)

    # Perturb radius with Gaussian noise
    radii_perturbed = radii_tensor.unsqueeze(1) + torch.randn(num_scenarios, 2) * variance

    # Calculate x and y coordinates for each sample
    x_coords = radii_perturbed * torch.cos(angles_rad) + MIC_ARRAY_CENTER[0]
    y_coords = radii_perturbed * torch.sin(angles_rad) + MIC_ARRAY_CENTER[1]

    coordinates = (torch.stack((x_coords, y_coords), dim=-1))

    return coordinates, angles


def generate_rir(source_position, reverb_time):
    """
    Generates RIRs of all microphones, for a given source position (x, y) and reverb time.
    """
    # Concat speaker height to source position
    source_position = torch.cat((source_position, torch.tensor((SPEAKER_HEIGHT,))), dim=0)
    # Generate RIRs of all microphones
    result_rirs = rir.generate(
        c=C,  # Sound velocity (m/s)
        fs=FS,  # Sample frequency (samples/s)
        r=MIC_ARRAY_POS,  # Receiver position(s) [x y z] (m)
        s=source_position,  # Source position [x y z] (m)
        L=L,  # Room dimensions [x y z] (m)
        reverberation_time=reverb_time,  # Reverberation time (s)
        nsample=NSAMPLE,  # Number of output samples
        mtype=MTYPE,  # Microphone type
        order=ORDER,  # Reflection order
        dim=DIM,  # Room dimension
        orientation=ORIENTATION,  # Microphone orientation
        hp_filter=ENABLE_HPF  # High-pass filter
    ).transpose()

    return torch.tensor(result_rirs)


def generate_rir_batch(source_positions, reverb_times):
    """
    For each scenario, generates pairs of RIRs of all microphones, for a given source position and reverb time.
    :param source_positions: Tensor of source positions of shape (num_scenarios, 2, 2)
    :param reverb_times: T60 reverberation times to be uniformly chosen from for each scenario
    :return: Tensor of size (num_scenarios, 2, NUM_MICS, NSAMPLE)
    """
    num_scenarios = source_positions.shape[0]
    reverb_times = torch.tensor(reverb_times)

    # Create tensor of output shape
    rirs = torch.zeros((num_scenarios, 2, NUM_MICS, NSAMPLE))

    # For each scenario, generate RIRs
    for scenario_idx in range(num_scenarios):
        reverb_idx = torch.randint(0, len(reverb_times), (1,)).item()
        for speaker_idx in range(2):
            rirs[scenario_idx, speaker_idx] = generate_rir(
                source_positions[scenario_idx, speaker_idx], reverb_times[reverb_idx])

    return rirs


def conv_rirs(scenario_signals, rirs):
    """
    For each scenario, convolve pairs of speech signal with corresponding RIRs.
    :param scenario_signals: Tensor of speech signal pairs of shape (num_scenarios, 2, signal_len)
    :param rirs: Tensor of RIRs of shape (num_scenarios, 2, NUM_MICS, NSAMPLE)
    :output: Tensor of shape (num_scenarios, 2, NUM_MICS, conv_len)
    """
    num_scenarios, _, signal_len = scenario_signals.size()
    rir_len = rirs.size(3)  # Should equal NSAMPLE

    # Reshape signals and RIRs to convolve along the last dimension
    signals_reshaped = scenario_signals.view(num_scenarios * 2, 1, signal_len).expand(-1, NUM_MICS, -1)
    rirs_reshaped = rirs.view(num_scenarios * 2, NUM_MICS, -1)

    # Create FFTConvolve instance
    fft_convolve = torchaudio.transforms.FFTConvolve(mode='full')

    # Perform convolution
    conv_result = fft_convolve(signals_reshaped, rirs_reshaped)

    # Reshape the result tensor
    conv_len = signal_len + rir_len - 1
    convolved_signals = conv_result.view(num_scenarios, 2, NUM_MICS, conv_len)

    return convolved_signals


def mix_rirs(perceived_signals, interfere=True):
    """
    For each scenario, for each mic, mixes perceived signals pair.
    :param perceived_signals: Tensor of pairs of signals perceived at different mics.
           shape: (num_scenarios, 2, NUM_MICS, conv_len)
    :param interfere: If true, mixes perceived pairs with random sirs in [TRAIN_SIR_LOW, TRAIN_SIR_HIGH]
    :output: Tensor of shape (num_scenarios, NUM_MICS, conv_len)
    """
    num_scenarios, _, _, conv_len = perceived_signals.size()

    # Generate random SIRs if interference is required
    if interfere:
        # Generate random SIRs for each scenario and mic within the specified range
        sirs = torch.rand(num_scenarios, NUM_MICS) * (TRAIN_SIR_HIGH - TRAIN_SIR_LOW) + TRAIN_SIR_LOW

        # Ensure that SIRs are in the correct range and convert to linear scale
        sirs = torch.pow(10, sirs.float() / 10)

        # Expand dimensions for broadcasting
        sirs = sirs.unsqueeze(2)

        # Mix the signals with interference
        mixed_signals = perceived_signals[:, 0, :, :] + perceived_signals[:, 1, :, :] / sirs
    else:
        # Mix the signals without interference
        mixed_signals = perceived_signals.sum(dim=1)  # Sum the signals from both speakers

    return mixed_signals


def calculate_rtf(mic_signals, discard_dc=True):
    """
    Calculates RTF for each microphone signal w.r.t corresponding reference mic signal.
    :param mic_signals: Tensor of signals perceived at different mics, for multiple scenarios.
                        Shape: (NUM_SCENARIOS, NUM_MICS, CONV_LENGTH)
    :param discard_dc: If True, DC component is discarded.
    :return:    1. Tensor of (real and imaginary parts of) RTFs for each scenario.
                Shape: (NUM_SCENARIOS, NUM_TIME_FRAMES, NUM_FREQUENCY_BINS, 2 * (NUM_MICS - 1))
                2. Magnitude tensor of the reference microphone.
                Shape: (NUM_SCENARIOS, NUM_TIME_FRAMES, NUM_FREQUENCY_BINS)
    """

    # Create Spectrogram instance
    spectrogram = Spectrogram(
        n_fft=NFFT,
        win_length=NFFT,
        hop_length=HOP_LENGTH,
        power=None  # Complex spectrum
    )

    # Compute STFTs - first microphone is used as reference
    ref_mic = spectrogram(mic_signals[:, 0]).unsqueeze(dim=1)
    # If discard_dc is True, discard DC component
    if discard_dc:
        ref_mic = ref_mic[:, :, 1:, :]
    ref_mic = ref_mic.permute(0, 3, 2, 1)
    non_ref_mics = spectrogram(mic_signals[:, 1:].reshape(-1, mic_signals.shape[2]))[:, 1:, :]
    non_ref_mics = non_ref_mics.reshape(mic_signals.shape[0],
                                        non_ref_mics.shape[2],
                                        non_ref_mics.shape[1],
                                        NUM_MICS - 1)

    # Compute RTFs via division
    rtf = non_ref_mics / (ref_mic + EPS)

    # Complex values -> real and imaginary parts
    rtf = torch.cat((torch.real(rtf), torch.imag(rtf)), dim=-1)

    # Get magnitude tensor of the reference microphone
    # ref_stft = ref_mic.squeeze(axis=-1).abs() # XXX: old
    ref_stft = ref_mic.squeeze(axis=-1)
    print(rtf.shape)
    return rtf, ref_stft


def calculate_target(signals, doas, discard_dc):
    """
    Calculates target DOAs for NUM_SCENARIOS pairs of clean audio signals.
    :param signals: Tensor of pairs of clean audio signals. Shape: (NUM_SCENARIOS, 2, SIGNAL_LEN)
    :param doas: Tensor of pairs of speaker DOAs. Shape: (NUM_SCENARIOS, 2)
    :param discard_dc: If True, DC component is discarded.
    :return: Tensor of DOAs (a DOA for each TF bin in each scenario).
                Shape: (NUM_SCENARIOS, NUM_TIME_FRAMES, NUM_FREQUENCY_BINS)
    """

    # Create Spectrogram instance
    spectrogram = Spectrogram(
        n_fft=NFFT,
        win_length=NFFT,
        hop_length=HOP_LENGTH,
        power=1  # Magnitude
    )

    # Compute STFTs of all clean signals
    # Shape: (NUM_SCENARIOS, 2, NUM_TIME_FRAMES, NUM_FREQUENCY_BINS)
    stfts = spectrogram(signals).permute(0, 1, 3, 2)

    # If discard_dc is True, discard DC component
    if discard_dc:
        stfts = stfts[:, :, :, 1:]

    # Get dominant speaker in each TF frame
    dominant_speakers = torch.argmax(stfts, dim=1)

    # dominant_doas = doas[torch.arange(dominant_speakers.shape[0])[:, None, None], dominant_speakers]
    dominant_doas = doas[torch.arange(dominant_speakers.size(0)).unsqueeze(1).unsqueeze(2), dominant_speakers]

    return dominant_doas

@dataclass
class SpeakerConf:
    file_path: str
    radius: float
    angle: float
    reverb: float

class TimitRIR:
    def __init__(self, base_dir: str = 'timit_rir'):
        super().__init__()
        self.__base_dir = base_dir

    @cached_property
    def dir_meta_df(self) -> pd.DataFrame:
        files_data: List[SpeakerConf] = [self._dispatch_file_name(f) for f in os.listdir(self.__base_dir)]
        df = pd.DataFrame(files_data)
        return df

    def _dispatch_file_name(self, file_name: str) -> SpeakerConf:
        """
        Dispatch SpeakerConf object form a filename.
        For example:
        Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.160s)_3-3-3-8-3-3-3_1m_060.mat
        -- >
        SpeakerConf(filepath=..., radius=1, angle=60, reverb=0.16)
        """
        reverb, _, radius, angle = file_name.split('_')[-4:]
        reverb = float(reverb[:-2]) # 0.160s) -->  0.160
        radius = float(radius[:-1]) # 1m --> 1
        """
        270.mat --> 180
        Raw data:
                0                                         0
          -45       45              ---->          315          45
        -90            90                       270                90

        We want:
                90
          135       45
        180            0
        """
        angle = (90 - int(angle.split('.')[0])) % 360

        result = SpeakerConf(file_path=os.path.join(self.__base_dir, file_name),
                             radius=radius,
                             angle=angle,
                             reverb=reverb)

        return result
    
    def _load_mat_as_torch(self, mat_path: str) -> torch.tensor:
        mat = scipy.io.loadmat(mat_path)
        numpy_impulse_response = mat['impulse_response']
        result = torch.from_numpy(numpy_impulse_response).float()
        # 48000x8 -> 1x8x48000
        result = result.T.unsqueeze(0)
        return result
    
    def sample(self, n: int, allowed_radii: List[int], returned_reverbs_values: List[float]) -> pd.DataFrame:
        dir_meta_df = self.dir_meta_df
        if allowed_radii:
            dir_meta_df = dir_meta_df[dir_meta_df.radius.isin(allowed_radii)]
        if returned_reverbs_values:
            # Sample equally each reverb - first entry gets the leftover
            nl = n // len(returned_reverbs_values)
            nl_first = nl + n - nl * len(returned_reverbs_values)
            dfs: List[pd.DataFrame] = [dir_meta_df[dir_meta_df.reverb == returned_reverbs_values[0]].sample(n=nl_first)]
            for reverb in returned_reverbs_values[1:]:
                current_df = dir_meta_df[dir_meta_df['reverb'] == reverb].sample(n=nl)
                dfs.append(current_df)

            df = pd.concat(dfs, ignore_index=True)
        else:
            df = dir_meta_df.sample(n=n)

        df['rir'] = df['file_path'].apply(self._load_mat_as_torch)
        return df


def generate_coords_rirs_test(num_scenarios: int, allowed_radii: Tuple[float], returned_reverbs_values: List[float]):
    """
    Generate pairs of coordinates and pairs of DOAs for given number
    of scenarios, radii, and microphone array center.

    Args:
    - num_scenarios (int): Number of scenarios to generate pairs of coordinates for.
    - radii (list or torch.Tensor): List or tensor of radii.
    - variance (float): Variance of Gaussian noise to perturb radius.

    Returns:
    - coordinates (torch.Tensor): Tensor of shape (num_scenarios, 2, 2) containing
      pairs of (x, y) coordinates for each scenario.
    - DOAs (torch.Tensor): Tensor of shape (num_scenarios, 2) containing pairs of DOAs.
    """

    # Generate random angles
    timit_rir_gen = TimitRIR(base_dir=TIMIT_RIR_PATH)
    df = timit_rir_gen.sample(num_scenarios * 2, allowed_radii=allowed_radii, returned_reverbs_values=returned_reverbs_values)
    print(num_scenarios)
    print(f"num scenarios: {num_scenarios}")
    print(f"num scenarios: {num_scenarios}")
    print(f"num scenarios: {num_scenarios}")
    angles = torch.tensor(df.angle.values)
    angles = angles.view(num_scenarios, 2)

    # Convert angles to radians
    angles_rad = angles.float() * (torch.pi / 180.0)

    # Perturb radius with Gaussian noise
    radii_perturbed = torch.tensor(df.radius.values).view(num_scenarios, 2)

    # Calculate x and y coordinates for each sample
    x_coords = radii_perturbed * torch.cos(angles_rad) + MIC_ARRAY_CENTER[0]
    y_coords = radii_perturbed * torch.sin(angles_rad) + MIC_ARRAY_CENTER[1]

    coordinates = (torch.stack((x_coords, y_coords), dim=-1))
    rirs = torch.stack(df.rir.tolist())
    rirs = rirs.view(num_scenarios, 2, NUM_MICS, -1)
 
    return coordinates, angles, rirs


def generate_batch(batch_size=64, test=False, source_dir='source_signals/LibriSpeech/dev-clean',
                   normalize=True, discard_dc=True) -> Dict[str, torch.tensor]:
    """
    Generates a batch for the neural network.
    :param test: If True, generates a batch for test set, otherwise for training set.
    :param source_dir: Directory containing speaker folders, each containing text folders with audio files
    :param batch_size: Number of scenarios in the batch (batch_size == NUM_SCENARIOS)
    :param normalize: If true, normalize each audio signal to zero mean to unit variance
    :param discard_dc: If True, DC component is discarded in STFT
    :return:    1. Tensor of (real and imaginary parts of) RTFs for each scenario.
                Shape: (NUM_SCENARIOS, NUM_TIME_FRAMES, NUM_FREQUENCY_BINS, 2 * (NUM_MICS - 1))
                2. Magnitude tensor of the reference microphone.
                Shape: (NUM_SCENARIOS, NUM_TIME_FRAMES, NUM_FREQUENCY_BINS)
    """
    # Load random pairs of audio signals
    audio_signals = load_source_signals(source_dir=source_dir, normalize=normalize, batch_size=batch_size)

    # If training batch, generate scenarios
    if not test:
        # Generate (x, y) positions for speakers
        source_positions, doas = generate_coords(num_scenarios=batch_size,
                                                 radii=torch.tensor(TRAIN_RAD_MEAN), variance=TRAIN_RAD_VAR)

        # Generate RIRs
        rirs = generate_rir_batch(source_positions=source_positions, reverb_times=TRAIN_REVERB_TIMES)

        # Convolve speaker signals with RIRs
        perceived_signals = conv_rirs(scenario_signals=audio_signals, rirs=rirs)

        # Mix pairs with interference
        mixed_signals = mix_rirs(perceived_signals=perceived_signals, interfere=True)

    # If test batch, use scenarios given in the real RIRs
    else:
        # Generate (x, y) positions for speakers
        # DONE: expand generate_coords to return radii too, or use something else
        source_positions, doas, rirs = generate_coords_rirs_test(num_scenarios=batch_size,
                                                                 allowed_radii=TEST_RADII,
                                                                 returned_reverbs_values=TEST_REVERB_TIMES)


        # DONE: load batch_size/2 RIRs with T60=0.160s and batch_size/2 RIRs with T60=0.160s
        # DONE: note that finally the angles should be in range [0, 180]

        # Convolve speaker signals with RIRs
        perceived_signals = conv_rirs(scenario_signals=audio_signals, rirs=rirs)

        # Mix pairs with no interference
        mixed_signals = mix_rirs(perceived_signals=perceived_signals, interfere=False)

    # Calculate RTFs and reference microphone magnitude tensor
    samples, ref_stft = calculate_rtf(mixed_signals, discard_dc=discard_dc)

    # Calculate target
    target = calculate_target(signals=audio_signals, doas=doas, discard_dc=discard_dc)
    return {'samples': samples, 'ref_stft': ref_stft, 'target': target}


# class AudioProjDataset(Dataset):
#     @classmethod
#     def create_dataset(self, number_of_samples: int = 5, test=False, source_dir='source_signals/LibriSpeech/dev-clean',
#                      normalize=True, discard_dc=True, output_path='data.pt'):
#         data = generate_batch(batch_size=number_of_samples, test=test, source_dir=source_dir, normalize=normalize, discard_dc=discard_dc)
#         with open(output_path, 'wb') as f:
#             torch.save(data, f)

#     def __init__(self, dataset_path: str = 'data.pt'):
#         super().__init__()
#         with open(dataset_path, 'rb') as f:
#             self.batches = torch.load(f)    

#     def __len__(self):
#         return self.batches['samples'].size(0)
    
#     def __getitem__(self, index) -> Tuple:
#         return super().__getitem__(index)

# generate_batch(batch_size=5)
data = generate_batch(batch_size=5)
with open('data.pt', 'wb') as f:
    torch.save(data, f)
