#!/usr/bin/env python3
import torch
from torchaudio.transforms import Spectrogram, InverseSpectrogram
import numpy as np

import config

try:
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
).to(config.device)

istft = InverseSpectrogram(
	n_fft=config.NFFT,
	win_length=config.NFFT,
	hop_length=config.HOP_LENGTH,
	onesided=True
).to(config.device)


def bss_eval(ref, est):
	"""Retuns a tensor of SDR, SIR."""
	sdr, isr, sir, sar = bsseval.evaluate(
		references=ref.unsqueeze(-1).numpy(),
		estimates=est.unsqueeze(-1).numpy(),
		# win=1*44100,
		# hop=1*44100,
		# mode='v4',
		# padding=True
	)
	
	return torch.stack([
		torch.tensor(np.mean(sdr, axis=1)),
		torch.tensor(np.mean(sir, axis=1))
	])




class SeparatedSource:
	"""Represents the part of the received sound that comes from this
	specific source (angle).
	"""
	def __init__(self, ref_spec, probs):
		# The ref mic's spectrogram, shape=(w, t)
		self.ref_spec = ref_spec
		# The model's output for this angle, shape=(w, t)
		self.probs = probs
	
	def energy(self):
		return torch.sum(self.probs * abs(self.ref_spec) ** 2)
	
	def spec(self):
		mag = abs(self.ref_spec) * self.probs
		phase = torch.angle(self.ref_spec)
		return mag * torch.exp(1j * phase)
	
	def signal(self):
		spec = self.spec()
		
		spec_with_dc = torch.empty(
			(spec.shape[0] + 1, spec.shape[1]),
			dtype=torch.complex64
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
		
		return bss_eval(
			ref=speaker_signal.unsqueeze(0),
			est=self.signal().unsqueeze(0)
		).squeeze(1)


def speaker_angles(sources):
	"""Returns the 2 angle numbers where the speakers are (most likely)
	located.
	"""
	energies = [src.energy() for src in sources]
	# `argpartition` to get the indices, AKA angle numbers.
	partitioned = np.argpartition(energies, len(sources) - 2)
	max_angles = tuple(partitioned[-2:])
	return max_angles


def zip_metrics(sources_pred, speakers_gt):
	return torch.stack([
		pred.metrics(gt)
		for pred, gt in zip(sources_pred, speakers_gt)
	])


def separated_sample_metrics(ref_spec, samp_probs, speaker_signals_gt):
	"""Retuns the metrics for both speakers of this sample, as a tensor
	with shape=(speaker_num, sdr_or_sir)
	"""
	# `samp_probs`'s shape is (angle_count, w, t).
	# `sources[i]` is the separeted source coming from the direction
	# theta_i.
	sources = [
		SeparatedSource(ref_spec, probs)
		for probs in samp_probs
	]
	speaker_sources_pred = [
		sources[angle]
		for angle in speaker_angles(sources)
	]
	
	possible_metrics = [
		zip_metrics(speaker_sources_pred, speaker_signals_gt),
		zip_metrics(speaker_sources_pred[::-1], speaker_signals_gt)
	]
	best_metrics = max(possible_metrics, key=torch.prod)
	return best_metrics


def separated_batch_metrics(batch):
	"""Returns a tensor with
	shape=(batch_size, speaker_num, sdr_or_sir).
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


def mixed_batch_metrics(batch, mixed_signals):
	"""Returns a tensor with
	shape=(batch_size, speaker_num, sdr_or_sir).
	"""
	# Flatted axes 0, 1:[smp0spk0, smp0spk1, smp1spk0, smp1spk1...]
	refs = batch['perceived_signals']
	refs = refs.view(-1, refs.shape[-1])
	
	# Repeat signals twice [mixed0, mixed0, mixed1, mixed1...]
	mixed_signals = refs[::2]
	ests = mixed_signals.repeat_interleave(2, dim=0)
	
	# Convert back to indexing by [smp_num, speaker_num, t]
	return bss_eval(ref=refs, est=ests).view(-1, 2, 2)


def print_batch_metrics(batch, mixed_signals, verbose=False):
	"""Calculates & prints the average metrics of this batch, for each
	speaker. if verbose=True, prints the metrics of each samples as
	well,
	"""
	SAMP_TITLE = "[ SAMPLE{:5d} ]======[ SDR ]=====[ SIR ]===================="
	AVG_TITLE  = "[ BATCH AVG ]========[ SDR ]=====[ SIR ]===================="
	TEMPLATE   = "Speaker{:5d}:        {:7.3f}     {:7.3f}"
	
	batch_mix = mixed_batch_metrics(batch, mixed_signals)
	batch_sep = separated_batch_metrics(batch)
	
	def print_metrics(mixed_metrics, separated_metrics, title):
		print(title)
		
		print("MIXED SIGNALS")
		for speaker_num, metrics in enumerate(mixed_metrics):
			print(TEMPLATE.format(speaker_num, *metrics))
		
		print("SEPARATED SIGNALS")
		for speaker_num, metrics in enumerate(separated_metrics):
			print(TEMPLATE.format(speaker_num, *metrics))
	
	if verbose:
		samp_num = 0
		for mix, sep in zip(batch_mix, batch_sep):
			print_metrics(mix, sep, SAMP_TITLE.format(samp_num))
			print()
			samp_num += 1
		print('\n')
	
	avg_mix = batch_mix.mean(dim=0)
	avg_sep = batch_sep.mean(dim=0)
	print_metrics(avg_mix, avg_sep, AVG_TITLE)


if __name__ == '__main__':
	batch = torch.load('example_batch2.pt')
	print_batch_metrics(batch, ..., verbose=False)
