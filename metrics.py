#!/usr/bin/env python3
import torch
from torchaudio.transforms import Spectrogram, InverseSpectrogram
from torchaudio import save
import numpy as np

import config

try:
	# pip install git+https://github.com/sigsep/bsseval@92535ea70e5c0864286ee5f0c5a4fa762de98546
	import bsseval
except ImportError:
	import sys
	sys.path.insert(0, 'bsseval-master')
	import bsseval


stft = Spectrogram(
	n_fft=config.NFFT,
	win_length=config.NFFT,
	hop_length=config.HOP_LENGTH,
	power=None  # Complex spectrum
).to(config.DEVICE)

istft = InverseSpectrogram(
	n_fft=config.NFFT,
	win_length=config.NFFT,
	hop_length=config.HOP_LENGTH,
	onesided=True
).to(config.DEVICE)


def bss_eval(ref, est):
	"""Retuns a tensor of SDR, SIR.
	ref.shape = (n_src, ref_signal_len)
	est.shape = (n_src, est_signal_len)
	return.shape = (n_src, SDR_or_SIR=2)
	"""
	sdr, isr, sir, sar = bsseval.evaluate(
		references=ref.unsqueeze(-1).detach().cpu().numpy(),
		estimates=est.unsqueeze(-1).detach().cpu().numpy(),
		# win=1*44100,
		# hop=1*44100,
		# mode='v4',
		# padding=True
	)
	avg_sdr = torch.tensor(np.mean(sdr, axis=1))
	avg_sir = torch.tensor(np.mean(sir, axis=1))
	return torch.stack((avg_sdr, avg_sir), dim=1)




class SeparatedSource:
	"""Represents the part of the received sound that comes from this
	specific source (angle).
	"""
	def __init__(self, ref_spec, probs):
		# The ref mic's spectrogram, shape=(w, t)
		self.ref_spec = ref_spec
		# The model's output for this angle, shape=(w, t)
		self.probs = probs
	
	def energy(self) -> float:
		return torch.sum(self.probs * abs(self.ref_spec) ** 2).item()
	
	def spec(self):
		mag = abs(self.ref_spec) * self.probs
		phase = torch.angle(self.ref_spec)
		return mag * torch.exp(1j * phase)
	
	def signal(self):
		spec = self.spec()
		
		spec_with_dc = torch.empty(
			(spec.shape[0] + 1, spec.shape[1]),
			dtype=torch.complex64,
			device=config.DEVICE
		)
		spec_with_dc[0] = 0
		spec_with_dc[1:] = spec
		
		return istft(spec_with_dc)
	
	def metrics(self, speaker_signal):
		"""Retuns a tensor of SDR, SIR"""
		# Just to make it the same length as `self.signal()`.
		speaker_spec = stft(speaker_signal)
		t = speaker_spec.shape[1]
		speaker_spec = speaker_spec[:, :(t // 16) * 16]
		speaker_signal = istft(speaker_spec)
		
		# Reshape (siglen,) -> (n_src=1, siglen)
		ref = speaker_signal.unsqueeze(0)
		est = self.signal().unsqueeze(0)
		
		# Reshape result (n_src=1, SDR_or_SIR=2) -> (SDR_or_SIR=2,)
		return bss_eval(ref, est).squeeze(0)
	
	@classmethod
	def speaker_angles(cls, sources):
		"""Returns the 2 angle numbers where the speakers are (most likely)
		located.
		"""
		energies = [src.energy() for src in sources]
		# `argpartition` to get the indices, AKA angle numbers.
		partitioned = np.argpartition(energies, len(sources) - 2)
		max_angles = tuple(partitioned[-2:])
		return max_angles
	
	@classmethod
	def speakers(cls, ref_spec, samp_probs):
		# `samp_probs`'s shape is (angle_count, w, t).
		# `sources[i]` is the separeted source coming from the direction
		# theta_i.
		sources = [
			cls(ref_spec, probs)
			for probs in samp_probs
		]
		
		return [
			sources[angle]
			for angle in cls.speaker_angles(sources)
		]

	def save(self, path):
		save(
			uri=path,
			src=self.signal().unsqueeze(0),
			sample_rate=config.FS
		)


def separated_sample_metrics(ref_spec, samp_probs, speaker_signals_gt):
	"""Retuns the metrics for both speakers of this sample, as a tensor
	with shape=(speaker_num, sdr_or_sir)
	"""
	def zipped_metrics(sources_pred, speakers_gt):
		return torch.stack([
			pred.metrics(gt)
			for pred, gt in zip(sources_pred, speakers_gt)
		]).to(config.DEVICE)
	
	speaker_sources_pred = SeparatedSource.speakers(ref_spec, samp_probs)
	
	possible_metrics = [
		zipped_metrics(speaker_sources_pred, speaker_signals_gt),
		zipped_metrics(speaker_sources_pred[::-1], speaker_signals_gt)
	]
	best_metrics = max(possible_metrics, key=torch.prod)
	
	return best_metrics


def separated_batch_metrics(batch):
	"""Returns a tensor with shape=
	(batch_size, speaker_num, sdr_or_sir).
	"""
	samples = zip(
		batch['ref_stft'],
		batch['probs'],
		batch['perceived_signals'],
	)
	
	return torch.stack([
		separated_sample_metrics(
			ref_spec[1:],  # Remove the DC freq.
			samp_probs,
			speaker_signals
		)
		for ref_spec, samp_probs, speaker_signals in samples
	])


def mixed_batch_metrics(batch):
	"""Returns a tensor with shape=
	(batch_size, speaker_num, sdr_or_sir).
	"""
	# Flatted axes 0, 1:[samp0spk0, samp0spk1, samp1spk0, samp1spk1...]
	refs = batch['perceived_signals']
	refs = refs.view(-1, refs.shape[-1])
	
	# Repeat signals twice [mixed0, mixed0, mixed1, mixed1...]
	ests = batch['mixed_signals'][:, 0].repeat_interleave(2, dim=0)
	
	# Convert back to indexing by [samp_num, speaker_num, sdr_or_sir]
	result = bss_eval(ref=refs, est=ests)
	result = result.view(-1, 2, 2)
	return result


def print_batch_metrics(batch, verbose=False):
	"""Calculates & prints the average metrics of this batch, for each
	speaker. if verbose=True, prints the metrics of each samples as
	well,
	"""
	SAMP_TITLE = "[ SAMPLE {:<5}]======[ SDR ]=====[ SIR ]===================="
	AVG_TITLE  = "[  BATCH AVG  ]======[ SDR ]=====[ SIR ]===================="
	TEMPLATE   = "Speaker {:<5}        {:7.3f}     {:7.3f}"
	
	batch_mix = mixed_batch_metrics(batch)
	batch_sep = separated_batch_metrics(batch)
	
	def print_metrics(mix_metrics, sep_metrics):
		print("MIXED SIGNALS")
		print(TEMPLATE.format(0, *mix_metrics[0]))
		print(TEMPLATE.format(1, *mix_metrics[1]))
		
		print("SEPARATED SIGNALS")
		print(TEMPLATE.format(0, *sep_metrics[0]))
		print(TEMPLATE.format(1, *sep_metrics[1]))
		
	if verbose:
		for samp_num, (mix, sep) in enumerate(zip(batch_mix, batch_sep)):
			print(SAMP_TITLE.format(samp_num))
			print_metrics(mix, sep)
			print()
		print('\n')
	
	avg_mix = batch_mix.mean(dim=0)
	avg_sep = batch_sep.mean(dim=0)
	print(AVG_TITLE)
	print_metrics(avg_mix, avg_sep)


def batch_speaker_signals(batch):
	"""Returns a tensor with shape=(batch_size, speaker_num, siglen)."""
	batch_speakers = torch.stack([
		torch.stack([
			speaker.signal()
			for speaker in SeparatedSource.speakers(ref_spec[1:], samp_probs)
		]) 
		for ref_spec, samp_probs in zip(batch['ref_stft'], batch['probs'])
	])


if __name__ == '__main__':
	# batch = torch.load('samples_test1605.pt', map_location=torch.device('cpu'))
	# batch['mixed_signals'] = batch['perceived_signals'][:, 0]
	
	batch = {
		'ref_stft': torch.rand((3, 257, 1280)),
		'mixed_signals': torch.rand((3, 8, 163760)),
		'perceived_signals': torch.rand((3, 2, 163760)),
		'probs': torch.rand((3, 13, 256, 1280)),
	}
	for key, val in batch.items():
		print(f"{key:<20}{tuple(val.shape)}")
	
	print_batch_metrics(batch, verbose=True)
